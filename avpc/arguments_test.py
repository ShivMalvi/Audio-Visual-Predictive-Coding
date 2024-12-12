"""
Settings for Argument Parsing.
"""

import argparse

pretrained_weights_frame = "./models/pretrained_models/frame_best.pth"
pretrained_weights_sound = "./models/pretrained_models/sound_best.pth"

class ArgParserTest(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        # Misc arguments
        self.parser.add_argument('--mode', default='eval',
                                 help="train/eval")
        self.parser.add_argument('--seed', default=1234, type=int,
                                 help='manual seed')
        self.parser.add_argument('--ckpt', default='./test',
                                 help='folder to output checkpoints')
        self.parser.add_argument('--disp_iter', type=int, default=400,
                                 help='frequency to display')
        self.parser.add_argument('--eval_epoch', type=int, default=1,
                                 help='frequency to evaluate')
        self.parser.add_argument('--log', default=None,
                                 help='the file to store the training log')

        # Model related arguments
        self.parser.add_argument('--id', default='',
                                 help="a name for identifying the model")
        self.parser.add_argument('--num_mix', default=2, type=int,
                                 help="number of sounds to mix")
        self.parser.add_argument('--arch_sound', default='pcnetlr',
                                 help="architecture of net_sound")
        self.parser.add_argument('--arch_frame', default='resnet18fc',
                                 help="architecture of net_frame")
        self.parser.add_argument('--weights_frame', default=pretrained_weights_frame,
                                 help="weights to finetune net_frame")
        self.parser.add_argument('--weights_sound', default=pretrained_weights_sound,
                                 help="weights to finetune net_sound")
        self.parser.add_argument('--num_frames', default=3, type=int,
                                 help='number of frames')
        self.parser.add_argument('--stride_frames', default=24, type=int,
                                 help='sampling stride of frames')
        self.parser.add_argument('--output_activation', default='sigmoid',
                                 help="activation on the output")
        self.parser.add_argument('--binary_mask', default=1, type=int,
                                 help="whether to use binary masks")
        self.parser.add_argument('--mask_thres', default=0.5, type=float,
                                 help="threshold in the case of binary masks")
        self.parser.add_argument('--loss', default='bce',
                                 help="loss function to reconstruct target mask")
        self.parser.add_argument('--weighted_loss', default=1, type=int,
                                 help="weighted loss")
        self.parser.add_argument('--log_freq', default=1, type=int,
                                 help="log frequency scale")

        # SimIter related arguments
        self.parser.add_argument('--cycles_inner', default=4, type=int,
                                 help='number of inner cycles to update representations in PC')
        self.parser.add_argument('--cycs_in_test', default=4, type=int,
                                 help='number of inner cycles to update representations in PC at test stage')
        self.parser.add_argument('--n_fm_visual', default=16, type=int,
                                 help='number of visual feature maps predicted in PC')
        self.parser.add_argument('--n_fm_out', default=1, type=int,
                                 help='number of output feature maps in PC')

        # Distributed Data Parallel
        self.parser.add_argument('--gpu_ids', default='0,1', type=str)
        self.parser.add_argument('--num_gpus', default=2, type=int,
                                 help='number of GPUs to use within a node')
        self.parser.add_argument('--batch_size_per_gpu', default=32, type=int,
                                 help='input batch size')
        self.parser.add_argument('--workers', default=8, type=int,
                                 help='number of data loading workers')
        self.parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                                 help='number of nodes for distributed training')
        self.parser.add_argument('-nr', '--nr', default=0, type=int,
                                 help='node rank for distributed training')

        # Data related arguments
        self.parser.add_argument('--audLen', default=65535, type=int,
                                 help='sound length')
        self.parser.add_argument('--audRate', default=11025, type=int,
                                 help='sound sampling rate')
        self.parser.add_argument('--stft_frame', default=1022, type=int,
                                 help="STFT frame length")
        self.parser.add_argument('--stft_hop', default=256, type=int,
                                 help="STFT hop length")
        self.parser.add_argument('--imgSize', default=224, type=int,
                                 help='size of input frame')
        self.parser.add_argument('--frameRate', default=8, type=float,
                                 help='video frame sampling rate')
        self.parser.add_argument('--num_val', default=-1, type=int,
                                 help='number of images to evaluate')
        self.parser.add_argument('--num_test', default=-1, type=int,
                                 help='number of images to test')
        self.parser.add_argument('--list_train', default='./data/train516.csv')
        self.parser.add_argument('--list_val', default='./data/val11.csv')
        self.parser.add_argument('--list_test', default='./data/test11.csv')
        self.parser.add_argument('--dup_trainset', default=100, type=int,
                                 help='duplicate so that one epoch has more iterations')
        self.parser.add_argument('--dup_validset', default=10, type=int,
                                 help='duplicate so that validation results are more meaningful')
        self.parser.add_argument('--dup_testset', default=10, type=int,
                                 help='duplicate so that test results are more meaningful')

        # Optimization related arguments


        self.parser.add_argument('--visualize', action='store_true', help='Save visualizations during evaluation')
        
        self.parser.add_argument('--num_epoch', default=100, type=int,
                                 help='epochs to train for')
        self.parser.add_argument('--lr_frame', default=1e-4, type=float, help='learning rate for frame net')
        self.parser.add_argument('--lr_sound', default=1e-3, type=float, help='learning rate for sound net')
        self.parser.add_argument('--lr_steps', nargs='+', type=int, default=[40, 80],
                                 help='steps to drop learning rate in epochs')
        self.parser.add_argument('--beta1', default=0.9, type=float,
                                 help='momentum for SGD, beta1 for Adam')
        self.parser.add_argument('--weight_decay', default=1e-2, type=float,
                                 help='weights regularizer')

    def parse_test_arguments(self):
        args = self.parser.parse_args()

        print("Input arguments:")
        for key, val in vars(args).items():
            print("{:16} {}".format(key, val))

        return args
