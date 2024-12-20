import os
import random
import warnings

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from arguments_test import ArgParserTest
from datasets.music import MUSICMixDataset
from models import ModelBuilder, activate
from utils import AverageMeter, warpgrid, makedirs, output_visuals, calc_metrics

warnings.filterwarnings("ignore")


def main():
    # Parse arguments
    parser = ArgParserTest()
    args = parser.parse_test_arguments()  # Correct method to parse test arguments

    # Validate GPU IDs
    available_gpus = torch.cuda.device_count()
    specified_gpus = list(map(int, args.gpu_ids.split(",")))

    if max(specified_gpus) >= available_gpus:
        print(f"[ERROR] Specified GPU IDs {args.gpu_ids} exceed available GPUs (count: {available_gpus}).")
        args.gpu_ids = "0"  # Fallback to the first available GPU
        args.num_gpus = 1
        print(f"[INFO] Defaulting to GPU ID: {args.gpu_ids}")

    # Set up device configuration
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        print(f"[INFO] Using GPU(s): {args.gpu_ids}")
    else:
        args.device = torch.device("cpu")
        args.gpu_ids = "-1"
        args.num_gpus = 0
        print("[INFO] CUDA not available. Running on CPU.")

    # Additional argument setup
    args.vis = os.path.join(args.ckpt, 'visualization/')
    args.log = os.path.join(args.ckpt, 'test_log.txt')

    # World size for distributed processing
    args.batch_size = args.num_gpus * args.batch_size_per_gpu
    args.world_size = args.num_gpus * args.nodes

    print(f"[INFO] Batch Size: {args.batch_size}")
    print(f"[INFO] World Size: {args.world_size}")

    # Environment setup for multiprocessing
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    # Start multiprocessing workers
    print(f"[INFO] Spawning {args.num_gpus} worker processes...")
    mp.spawn(main_worker, nprocs=args.num_gpus, args=(args,))


def main_worker(gpu, args):
    rank = args.nr * args.num_gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Network builders
    builder = ModelBuilder()
    net_frame = builder.build_frame(arch=args.arch_frame, fc_vis=args.n_fm_visual, weights='')
    net_sound = builder.build_sound(arch=args.arch_sound, weights='', cyc_in=args.cycles_inner, fc_vis=args.n_fm_visual, n_fm_out=args.n_fm_out)

    if gpu == 0:
        # Count number of parameters
        n_params_net_frame = sum(p.numel() for p in net_frame.parameters())
        print('#P of net_frame: {}'.format(n_params_net_frame))
        n_params_net_sound = sum(p.numel() for p in net_sound.parameters())
        print('#P of net_sound: {}'.format(n_params_net_sound))
        print('Total #P: {}'.format(n_params_net_frame + n_params_net_sound))

    # Loss function
    crit_mask = builder.build_criterion(arch=args.loss)

    torch.cuda.set_device(gpu)
    net_frame.cuda(gpu)
    net_sound.cuda(gpu)

    # Wrap model
    netWrapper = NetWrapper(net_frame, net_sound, crit_mask)
    netWrapper = torch.nn.parallel.DistributedDataParallel(netWrapper, device_ids=[gpu])
    netWrapper.to(args.device)

    # Load well-trained model
    map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
    net_frame.load_state_dict(torch.load(args.weights_frame, map_location=map_location))
    net_sound.load_state_dict(torch.load(args.weights_sound, map_location=map_location))

    args.batch_size_ = int(args.batch_size / args.num_gpus)
    args.batch_size_val_test = 30

    # Dataset and loader
    dataset_test = MUSICMixDataset(args.list_test, args, process_stage='test')

    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size_val_test,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False)

    if gpu == 0:
        evaluate(netWrapper, loader_test, args)


class NetWrapper(torch.nn.Module):
    def __init__(self, net_frame, net_sound, crit):
        super(NetWrapper, self).__init__()
        self.net_frame, self.net_sound = net_frame, net_sound
        self.crit = crit

    def forward(self, batch_data, args):
        mag_mix = batch_data['mag_mix'].cuda(non_blocking=True)
        mags = [m.cuda(non_blocking=True) for m in batch_data['mags']]
        frames = [f.cuda(non_blocking=True) for f in batch_data['frames']]

        N = args.num_mix
        gt_masks = [(m / mag_mix).clamp(0., 5.) for m in mags]
        feat_map_frames = [self.net_frame.forward_multiframe(f) for f in frames]

        pred_masks = [activate(self.net_sound.forward_test_stage(torch.log(mag_mix), f, args.cycs_in_test), args.output_activation) for f in feat_map_frames]
        err_mask = self.crit(pred_masks, gt_masks, torch.ones_like(mag_mix))

        outputs = {'pred_masks': pred_masks, 'gt_masks': gt_masks, 'mag_mix': mag_mix, 'mags': mags}
        return err_mask, outputs



def evaluate(netWrapper, loader, args):
    torch.set_grad_enabled(False)

    # Remove previous visualization results
    makedirs(args.vis, remove=True)
    print("\n--- Visualization Directory Content ---")
    print(os.listdir(args.vis))

    # Switch to eval mode
    netWrapper.eval()

    # Initialize meters
    loss_meter = AverageMeter()
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    try:
        with torch.no_grad():
            for i, batch_data in enumerate(loader):
                # Forward pass
                err_mask, outputs = netWrapper.forward(batch_data, args)
                err = err_mask.mean()
                loss_meter.update(err.item())

                # Log intermediate loss
                if i % 10 == 0 or i == len(loader) - 1:
                    print(f"[Eval] Iter {i}, Loss: {err.item():.4f}")
                    log_path = os.path.join(args.ckpt, 'test_log.txt')
                    with open(log_path, 'a') as log_file:
                        log_file.write(f"[Batch {i}] Loss: {err.item():.4f}\n")

                # Calculate metrics
                sdr_mix, sdr, sir, sar, valid_num = calc_metrics(batch_data, outputs, args)
                sdr_mix_meter.update(sdr_mix)
                sdr_meter.update(sdr)
                sir_meter.update(sir)
                sar_meter.update(sar)

                # Visualization
                if i < 3:  # Generate visualizations for the first 3 batches
                    print(f"Generating visualizations for batch {i}...")
                    output_visuals(batch_data, outputs, args)

            # Final metrics
            metric_output = (
                f"[Test Summary] Loss: {loss_meter.average():.4f}, "
                f"SDR_mix: {sdr_mix_meter.average():.4f}, SDR: {sdr_meter.average():.4f}, "
                f"SIR: {sir_meter.average():.4f}, SAR: {sar_meter.average():.4f}"
            )
            print(metric_output)

            # Save final metrics
            log_path = os.path.join(args.ckpt, 'test_log.txt')
            with open(log_path, 'a') as log_file:
                log_file.write(f"{metric_output}\n")
            print(f"[DEBUG] Metrics logged in {log_path}")

    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")

        

# Ensure `args` is passed correctly in the function and is properly initialized
def log_metrics(loader, netWrapper, args):
    """
    Log evaluation metrics for the test loader.
    """
    try:
        # Set up the log path using args.ckpt
        log_path = os.path.join(args.ckpt, 'test_log.txt')
        
        # Open the log file in append mode
        with open(log_path, 'a') as log_file:
            # Iterate through the test loader
            for i, batch_data in enumerate(loader):
                # Forward pass to calculate metrics
                err_mask, outputs = netWrapper.forward(batch_data, args)
                err = err_mask.mean()

                # Calculate SDR, SIR, SAR, etc.
                sdr_mix, sdr, sir, sar, valid_num = calc_metrics(batch_data, outputs, args)

                # Log metrics
                log_file.write(
                    f"[Eval] Batch {i}, Loss: {err.item():.4f}, "
                    f"SDR: {sdr:.4f}, SIR: {sir:.4f}, SAR: {sar:.4f}, Valid: {valid_num}\n"
                )

                print(f"[DEBUG] Metrics logged for batch {i} in {log_path}")

        print("[DEBUG] All metrics logged successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to log metrics: {e}")



if __name__ == '__main__':
    main()
