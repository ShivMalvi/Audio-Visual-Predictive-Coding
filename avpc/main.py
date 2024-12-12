import os
import time
import random
import warnings

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from arguments import ArgParser
from datasets.music import MUSICMixDataset
from models import ModelBuilder, activate
from utils import AverageMeter, warpgrid, makedirs, output_visuals, calc_metrics, plot_loss_metrics

warnings.filterwarnings("ignore")


def main():
    # arguments
    parser = ArgParser()
    args = parser.parse_train_arguments()
    args.batch_size = args.num_gpus * args.batch_size_per_gpu

    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    # experiment name
    if args.mode == 'train':
        args.id += '-{}mix'.format(args.num_mix)
        if args.log_freq:
            args.id += '-LogFreq'
        args.id += '-{}-{}'.format(args.arch_frame, args.arch_sound)
        args.id += '-frames{}stride{}'.format(args.num_frames, args.stride_frames)
        if args.binary_mask:
            assert args.loss == 'bce', 'Binary Mask should go with BCE loss'
            args.id += '-binary'
        else:
            args.id += '-ratio'
        if args.weighted_loss:
            args.id += '-weightedLoss'
        args.id += '-epoch{}'.format(args.num_epoch)
        args.id += '-step' + '_'.join([str(x) for x in args.lr_steps])

    print('Model ID: {}'.format(args.id))

    # paths to save/load output
    args.vis = os.path.join(args.ckpt, 'visualization/')
    args.log = os.path.join(args.ckpt, 'running_log.txt')

    # initialize the best SDR with a small number
    args.best_sdr = -float("inf")

    args.world_size = args.num_gpus * args.nodes
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # specified by yourself
    os.environ['MASTER_PORT'] = '29500'  # specified by yourself
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    mp.spawn(main_worker, nprocs=args.num_gpus, args=(args,))

    # Manual checkpoint saving
    if args.mode == 'save_checkpoint':
      print("Manually saving checkpoint...")
      builder = ModelBuilder()
      net_frame = builder.build_frame(
          arch=args.arch_frame,
          fc_vis=args.n_fm_visual,
          weights=args.weights_frame)
      net_sound = builder.build_sound(
          arch=args.arch_sound,
          cyc_in=args.cycles_inner,
          fc_vis=args.n_fm_visual,
          n_fm_out=args.n_fm_out)

      # Save the checkpoints
      checkpoint(net_frame, net_sound, args.ckpt)
      print(f"Checkpoint manually saved to {args.ckpt}.")
      return

    # Training or other modes (CUDA initialization happens here)
    mp.spawn(main_worker, nprocs=args.num_gpus, args=(args,))




    

def main_worker(gpu, args):
    rank = args.nr * args.num_gpus + gpu
    dist.init_process_group(
        backend='gloo', init_method='env://', world_size=args.world_size, rank=rank
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)


    # Network builders
    builder = ModelBuilder()
    net_frame = builder.build_frame(
        arch=args.arch_frame,
        fc_vis=args.n_fm_visual,
        weights=args.weights_frame
    )
    net_sound = builder.build_sound(
        arch=args.arch_sound,
        weights=args.weights_sound,
        cyc_in=args.cycles_inner,
        fc_vis=args.n_fm_visual,
        n_fm_out=args.n_fm_out
    )

    if args.device.type == "cuda":
        torch.cuda.set_device(gpu)
        net_frame.cuda(gpu)
        net_sound.cuda(gpu)

    # Wrap the models for Distributed Data Parallel if CUDA is available
    if args.device.type == "cuda":
        netWrapper = NetWrapper(net_frame, net_sound, crit_mask).to(args.device)
        netWrapper = torch.nn.parallel.DistributedDataParallel(netWrapper, device_ids=[gpu])
    else:
        netWrapper = NetWrapper(net_frame, net_sound, crit_mask).to(args.device)


    if gpu == 0:
        # count number of parameters
        n_params_net_frame = sum(p.numel() for p in net_frame.parameters())
        print('#P of net_frame: {}'.format(n_params_net_frame))
        n_params_net_sound = sum(p.numel() for p in net_sound.parameters())
        print('#P of net_sound: {}'.format(n_params_net_sound))
        print('Total #P: {}'.format(n_params_net_frame + n_params_net_sound))

    # loss function
    crit_mask = builder.build_criterion(arch=args.loss)

    torch.cuda.set_device(gpu)
    net_frame.cuda(gpu)
    net_sound.cuda(gpu)

    

    # set up optimizer
    optimizer = create_optimizer(net_frame, net_sound, args)

    args.batch_size_ = int(args.batch_size / args.num_gpus)
    args.batch_size_val_test = 30

    # dataset and loader
    dataset_train = MUSICMixDataset(args.list_train, args, process_stage='train')
    dataset_val = MUSICMixDataset(args.list_val, args, max_sample=args.num_val, process_stage='val')
    dataset_test = MUSICMixDataset(args.list_test, args, max_sample=args.num_test, process_stage='test')

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size_,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_val_test,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size_val_test,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False)

    # history of performance
    history = {
        'train': {'epoch': [], 'loss': []},
        'val': {'epoch': [], 'loss': [], 'sdr': [], 'sir': [], 'sar': []}}

    if gpu == 0:
        args.epoch_iters = len(dataset_train) // args.batch_size
        print('1 Epoch = {} iters'.format(args.epoch_iters))

        # eval first
        evaluate(netWrapper, loader_val, history, 0, args)
        evaluate_testset(netWrapper, loader_test, 0, args)
        print('Evaluation before training is done!')

    # start training
    for epoch in range(args.num_epoch):

        # train for one epoch
        train(netWrapper, loader_train, optimizer, history, epoch + 1, gpu, args)

        # evaluation and visualization
        if epoch % args.eval_epoch == 0 and gpu == 0:
            evaluate(netWrapper, loader_val, history, epoch + 1, args)

            # save checkpoint and test modal on test set
            cur_sdr = history['val']['sdr'][-1]
            print(f"Validation SDR: {cur_sdr}, Best SDR so far: {args.best_sdr}")
            if cur_sdr > args.best_sdr:
                print("New best SDR achieved. Saving checkpoint...")
                args.best_sdr = cur_sdr
                checkpoint(net_frame, net_sound, args.ckpt)

                evaluate_testset(netWrapper, loader_test, epoch + 1, args)

        # Ensure checkpoint directory exists
        if not os.path.exists(args.ckpt):
            print(f"Checkpoint directory '{args.ckpt}' does not exist. Creating it...")
            os.makedirs(args.ckpt)
        else:
            print(f"Checkpoint directory '{args.ckpt}' exists.")

        # adjust learning rate
        if epoch + 1 in args.lr_steps:
            adjust_learning_rate(optimizer, args)
    print("Manually saving checkpoint at the end of training...")
    checkpoint(net_frame, net_sound, args.ckpt)
    print("Manual checkpoint save complete.")


        # Example: Force save checkpoint after training one epoch
    if gpu == 0:
        print("[DEBUG] Saving checkpoint manually for debugging...")
        checkpoint(net_frame, net_sound, args.ckpt)
        print(f"[DEBUG] Checkpoint saved successfully to {args.ckpt}")
 #  if gpu == 0:
        print('\nTraining Done!')


# network wrapper, defines forward pass
class NetWrapper(torch.nn.Module):
    def __init__(self, net_frame, net_sound, crit):
        super(NetWrapper, self).__init__()
        self.net_frame, self.net_sound = net_frame, net_sound
        self.crit = crit

    def forward(self, batch_data, args):
        mag_mix = batch_data['mag_mix']
        mags = batch_data['mags']
        frames = batch_data['frames']
        mag_mix = mag_mix + 1e-10

        N = args.num_mix
        B = mag_mix.size(0)
        T = mag_mix.size(3)

        mag_mix = mag_mix.cuda(non_blocking=True)
        for n in range(N):
            mags[n] = mags[n].cuda(non_blocking=True)
            frames[n] = frames[n].cuda(non_blocking=True)

        # 0.0 warp the spectrogram
        if args.log_freq:
            grid_warp = torch.from_numpy(warpgrid(B, 256, T, warp=True)).cuda(non_blocking=True)
            mag_mix = F.grid_sample(mag_mix, grid_warp, align_corners=False)
            for n in range(N):
                mags[n] = F.grid_sample(mags[n], grid_warp, align_corners=False)

        # 0.1 calculate loss weighting coefficient: magnitude of input mixture
        if args.weighted_loss:
            weight = torch.log1p(mag_mix)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = torch.ones_like(mag_mix)

        # 0.2 ground truth masks are computed after warpping!
        gt_masks = [None for n in range(N)]
        for n in range(N):
            if args.binary_mask:
                # for simplicity, mag_N > 0.5 * mag_mix
                gt_masks[n] = (mags[n] > 0.5 * mag_mix).float()
            else:
                gt_masks[n] = mags[n] / mag_mix
                # clamp to avoid large numbers in ratio masks
                gt_masks[n].clamp_(0., 5.)

        # LOG magnitude
        log_mag_mix = torch.log(mag_mix).detach()

        # 1. forward net_frame
        feat_map_frames = [None for n in range(N)]
        for n in range(N):
            feat_map_frames[n] = self.net_frame.forward_multiframe(frames[n])

        # 2. forward net_sound
        pred_masks = [None for n in range(N)]
        for n in range(N):
            pred_masks[n] = self.net_sound(log_mag_mix, feat_map_frames[n])
            pred_masks[n] = activate(pred_masks[n], args.output_activation)

        # 3. loss
        err_mask = self.crit(pred_masks, gt_masks, weight).reshape(1)

        outputs = {'pred_masks': pred_masks, 'gt_masks': gt_masks,
                   'mag_mix': mag_mix, 'mags': mags, 'weight': weight}

        return err_mask, outputs


# train one epoch
def train(netWrapper, loader, optimizer, history, epoch, gpu, args):
    torch.set_grad_enabled(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    netWrapper.train()

    # main loop
    torch.cuda.synchronize()
    tic = time.perf_counter()
    for i, batch_data in enumerate(loader):
        # measure data time
        torch.cuda.synchronize()
        data_time.update(time.perf_counter() - tic)

        # forward pass
        netWrapper.zero_grad()

        print(f"[DEBUG] Forward pass at iteration {i}")
        err_mask, _ = netWrapper.forward(batch_data, args)
        print(f"[DEBUG] Loss calculated: {err_mask.mean().item()}")

        err = err_mask.mean()

        # backward
        err.backward()
        optimizer.step()
        print(f"[DEBUG] Backward pass completed at iteration {i}")

        # measure total time
        torch.cuda.synchronize()
        batch_time.update(time.perf_counter() - tic)
        tic = time.perf_counter()

        # display
        if i % args.disp_iter == 0 and gpu == 0:
            if i == 0:
                print('\n------------------------------------------------------------------------------')
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_frame: {}, lr_sound: {}, '
                  'loss-simsiam: {:.4f}'
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.lr_frame, args.lr_sound,
                          err.item()))

        if gpu == 0:
            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(err.item())


def evaluate(netWrapper, loader, history, epoch, args):
    if epoch == 0:
        print('\n-----------------------------------------------------------------')
    print('Evaluating at epoch {}...'.format(epoch))
    torch.set_grad_enabled(False)

    # remove previous viz results
    makedirs(args.vis, remove=True)

    # switch to eval mode
    netWrapper.eval()

    # initialize meters
    loss_meter = AverageMeter()
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    eval_num = 0
    valid_num = 0
    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            # forward pass
            eval_num += batch_data['mag_mix'].shape[0]
            err_mask, outputs = netWrapper.forward(batch_data, args)
            err = err_mask.mean()
    
            loss_meter.update(err.item())
    
            total_batch = (11 * args.dup_validset // args.batch_size_val_test)
            if i == 0 or (i + 1) == (total_batch // 2) or (i + 1) == total_batch:
                print('[Eval] iter {}, loss: {:.4f}'.format(i, err.item()))
    
            # calculate metrics
            sdr_mix, sdr, sir, sar, valid_num = calc_metrics(batch_data, outputs, args)
            sdr_mix_meter.update(sdr_mix)
            sdr_meter.update(sdr)
            sir_meter.update(sir)
            sar_meter.update(sar)
            #valid_num += cur_valid_num
    
            # output visualization
            if i == 0:
                output_visuals(batch_data, outputs, args)

    metric_output = ('[Test Summary] Loss: {:.4f}, '
                 'SDR_mixture: {:.4f}, SDR: {:.4f}, SIR: {:.4f}, SAR: {:.4f}').format(
    loss_meter.average(),
    sdr_mix_meter.average(),
    sdr_meter.average(),
    sir_meter.average(),
    sar_meter.average())
    print(metric_output)

    learning_rate = ' lr_sound: {}, lr_frame: {}'.format(args.lr_sound, args.lr_frame)
    with open(args.log, 'a') as F:
        if sdr_meter.average() > args.best_sdr:
            F.write(
                '***************************************************************************************************\n')
            F.write(metric_output + learning_rate + '\n')
            F.write(
                '***************************************************************************************************\n')
        else:
            F.write(metric_output + learning_rate + '\n')

    history['val']['epoch'].append(epoch)
    history['val']['loss'].append(loss_meter.average())
    history['val']['sdr'].append(sdr_meter.average())
    history['val']['sir'].append(sir_meter.average())
    history['val']['sar'].append(sar_meter.average())

    # plot figure
    if epoch > 0:
        print('Plotting figures...')
        plot_loss_metrics(args.ckpt, history)


def evaluate_testset(netWrapper, loader, epoch, args):
    print(f"=========================Test at epoch {epoch}=========================")
    torch.set_grad_enabled(False)
    netWrapper.eval()

    # Initialize metrics
    loss_meter = AverageMeter()
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    eval_num = 0
    valid_num = 0
    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            eval_num += batch_data['mag_mix'].shape[0]

            # Forward pass
            err_mask, outputs = netWrapper.forward(batch_data, args)
            err = err_mask.mean()

            loss_meter.update(err.item())
            if i == 0:
              print(f"Saving visualizations for batch {i}...")
              output_visuals(batch_data, outputs, args)
              print(f'[Test] iter {i}, loss: {err.item()}')

            # Calculate metrics
            sdr_mix, sdr, sir, sar, cur_valid_num = calc_metrics(batch_data, outputs, args)
            sdr_mix_meter.update(sdr_mix)
            sdr_meter.update(sdr)
            sir_meter.update(sir)
            sar_meter.update(sar)
            valid_num += cur_valid_num

    # Log metrics
    metric_output = (f'[Test Summary] Epoch: {epoch}, Loss: {loss_meter.average():.4f}, '
                     f'SDR: {sdr_meter.average():.4f}, SIR: {sir_meter.average():.4f}, SAR: {sar_meter.average():.4f}')
    if valid_num / eval_num < 0.8:
        metric_output += ' ---- Invalid ---- '
    print(metric_output)

    log_file = os.path.join(args.ckpt, 'running_log.txt')
    with open(log_file, 'a') as f:
        f.write(metric_output + '\n')


    # Example usage of output_visuals during evaluation
    if args.mode == 'test':
      for i, batch_data in enumerate(loader_test):
          err_mask, outputs = netWrapper.forward(batch_data, args)
          # Generate visualizations for all or specific batches
          if i == 0:  # You can change this condition to generate for all batches
              output_visuals(batch_data, outputs, args)

    print('=========================Test Finished!=========================')


def output_visuals(batch_data, outputs, args, save_dir='visualization'):
    print("Saving visualization files...")
    save_path = os.path.join(args.ckpt, save_dir)
    os.makedirs(save_path, exist_ok=True)
    print(f"[DEBUG] Output directory: {save_path}")

    for i, (mag_mix, pred_mask, gt_mask) in enumerate(
        zip(outputs['mag_mix'], outputs['pred_masks'], outputs['gt_masks'])):
        
      try:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(mag_mix.cpu().numpy(), aspect='auto', origin='lower')
        ax[0].set_title("Input Mixture")

        ax[1].imshow(pred_mask[0].cpu().numpy(), aspect='auto', origin='lower')
        ax[1].set_title("Predicted Mask")

        ax[2].imshow(gt_mask[0].cpu().numpy(), aspect='auto', origin='lower')
        ax[2].set_title("Ground Truth Mask")

        plt.tight_layout()
        save_file = os.path.join(save_path, f'visual_{i}.png')
        plt.savefig(save_file)
        plt.close(fig)
        print(f"[DEBUG] Saved visualization for sample {i} at {save_file}")

      except Exception as e:
        print(f"[ERROR] Failed to save visualization for sample {i}: {e}")
    
    if args.visualize:
      print("[INFO] Visualization enabled.")
      output_visuals(batch_data, outputs, args, save_dir='visualization')





def create_optimizer(net_frame, net_sound, args):

    params_conv_pc = [p for p in net_sound.UpConvs.parameters()] + \
                     [p for p in net_sound.DownConvs.parameters()] + \
                     [p for p in net_sound.BNUp.parameters()] + \
                     [p for p in net_sound.BNDown.parameters()] + \
                     [p for p in net_sound.BNUp_step.parameters()] + \
                     [p for p in net_sound.BNDown_step.parameters()]
    params_rate_pc = [p for p in net_sound.a0.parameters()] + \
                     [p for p in net_sound.b0.parameters()]

    param_groups = [{'params': net_frame.fc.parameters(), 'lr': args.lr_frame},
                    {'params': params_conv_pc, 'lr': args.lr_sound},
                    {'params': params_rate_pc, 'lr': args.lr_sound, 'weight_decay': 0}]

    return torch.optim.AdamW(param_groups, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)


def checkpoint(net_frame, net_sound, save_path, epoch=None):
    print(f"[DEBUG] Saving checkpoint to {save_path}...")
    os.makedirs(save_path, exist_ok=True)
    try:
        frame_path = os.path.join(save_path, "frame_best.pth")
        sound_path = os.path.join(save_path, "sound_best.pth")
        torch.save(net_frame.state_dict(), frame_path)
        torch.save(net_sound.state_dict(), sound_path)
        print(f"[DEBUG] Checkpoint saved: {frame_path}, {sound_path}")
        if epoch is not None:
            epoch_dir = os.path.join(save_path, "epoch_checkpoints")
            os.makedirs(epoch_dir, exist_ok=True)
            epoch_frame_path = os.path.join(epoch_dir, f"frame_epoch_{epoch}.pth")
            epoch_sound_path = os.path.join(epoch_dir, f"sound_epoch_{epoch}.pth")
            torch.save(net_frame.state_dict(), epoch_frame_path)
            torch.save(net_sound.state_dict(), epoch_sound_path)
            print(f"[DEBUG] Epoch checkpoint saved: {epoch_frame_path}, {epoch_sound_path}")
    except Exception as e:
        print(f"[ERROR] Checkpoint saving failed: {e}")


        

def adjust_learning_rate(optimizer, args):
    args.lr_sound *= 0.1
    args.lr_frame *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1


if __name__ == '__main__':
    main()
