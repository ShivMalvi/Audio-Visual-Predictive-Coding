"""
Settings for Argument Parsing.
"""

import argparse
import os

# Default paths for pretrained weights
pretrained_weights_frame = "/content/Audio-Visual-Predictive-Coding/avpc/results/epoch_checkpoints/epoch_1.pth/frame_best.pth"
pretrained_weights_sound = "/content/Audio-Visual-Predictive-Coding/avpc/results/epoch_checkpoints/epoch_1.pth/sound_best.pth"

class ArgParserTest:
    def __init__(self):
        # Initialize ArgumentParser once in the constructor
        self.parser = argparse.ArgumentParser(description="Argument Parser for Testing Audio-Visual Predictive Coding")

        # Misc arguments
        self.parser.add_argument('--mode', default='eval', help="train/eval")
        self.parser.add_argument('--seed', default=1234, type=int, help='manual seed')
        self.parser.add_argument('--ckpt', default='./test', help='folder to output checkpoints')
        self.parser.add_argument('--disp_iter', type=int, default=400, help='frequency to display')
        self.parser.add_argument('--eval_epoch', type=int, default=1, help='frequency to evaluate')
        self.parser.add_argument('--log', default=None, help='the file to store the training log')

        # Model related arguments
        self.parser.add_argument('--id', default='', help="a name for identifying the model")
        self.parser.add_argument('--num_mix', default=1, type=int, help="number of sounds to mix")
        self.parser.add_argument('--arch_sound', default='pcnetlr', help="architecture of net_sound")
        self.parser.add_argument('--arch_frame', default='resnet18fc', help="architecture of net_frame")
        # Replace None with actual paths
        self.parser.add_argument('--weights_frame', default='/content/Audio-Visual-Predictive-Coding/avpc/results/epoch_checkpoints/epoch_1.pth/frame_best.pth', help="weights to finetune net_frame")
        self.parser.add_argument('--weights_sound', default='/content/Audio-Visual-Predictive-Coding/avpc/results/epoch_checkpoints/epoch_1.pth/sound_best.pth', help="weights to finetune net_sound")
        self.parser.add_argument('--num_frames', default=3, type=int, help='number of frames')
        self.parser.add_argument('--stride_frames', default=24, type=int, help='sampling stride of frames')
        self.parser.add_argument('--output_activation', default='sigmoid', help="activation on the output")
        self.parser.add_argument('--binary_mask', default=1, type=int, help="Use binary masks (1: True, 0: False)")
        self.parser.add_argument('--mask_thres', default=0.5, type=float, help="Threshold for binary masks")
        self.parser.add_argument('--loss', default='bce', help="Loss function for reconstruction")
        self.parser.add_argument('--weighted_loss', default=1, type=int, help="Enable weighted loss (1: True, 0: False)")
        self.parser.add_argument('--log_freq', default=1, type=int, help="Log frequency scale (1: True, 0: False)")

        # PCNet arguments
        self.parser.add_argument('--cycles_inner', default=4, type=int, help='Number of inner cycles for Predictive Coding network')
        self.parser.add_argument('--cycs_in_test', default=4, type=int, help='Number of cycles at test stage')
        self.parser.add_argument('--n_fm_visual', default=16, type=int, help='Number of visual feature maps')
        self.parser.add_argument('--n_fm_out', default=1, type=int, help='Number of output feature maps')

        # Distributed Data Parallel arguments
        self.parser.add_argument('--gpu_ids', default='0,1', type=str, help='Comma-separated list of GPU IDs')
        self.parser.add_argument('--num_gpus', default=2, type=int, help='Number of GPUs to use')
        self.parser.add_argument('--batch_size_per_gpu', default=32, type=int, help='Batch size per GPU')
        self.parser.add_argument('--workers', default=8, type=int, help='Number of worker threads for data loading')
        self.parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N', help='Number of compute nodes')
        self.parser.add_argument('-nr', '--nr', default=0, type=int, help='Node rank for multi-node setup')

        # Dataset and data loading arguments
        self.parser.add_argument('--audLen', default=65535, type=int, help='Audio length')
        self.parser.add_argument('--audRate', default=11025, type=int, help='Audio sampling rate')
        self.parser.add_argument('--stft_frame', default=1022, type=int, help="STFT frame length")
        self.parser.add_argument('--stft_hop', default=256, type=int, help="STFT hop length")
        self.parser.add_argument('--imgSize', default=224, type=int, help='Size of input video frame')
        self.parser.add_argument('--frameRate', default=8, type=float, help='Video frame sampling rate')
        self.parser.add_argument('--num_val', default=-1, type=int, help='Number of validation samples')
        self.parser.add_argument('--num_test', default=-1, type=int, help='Number of test samples')
        self.parser.add_argument('--batch_size_val_test', default=30, type=int, help='Batch size for validation and testing')
        self.parser.add_argument('--list_train', default='/content/Audio-Visual-Predictive-Coding/avpc/data/train516.csv', help='Path to train dataset CSV')
        self.parser.add_argument('--list_val', default='/content/Audio-Visual-Predictive-Coding/avpc/data/val11.csv', help='Path to validation dataset CSV')
        self.parser.add_argument('--list_test', default='/content/Audio-Visual-Predictive-Coding/avpc/data/test11.csv', help='Path to test dataset CSV')


        # Add this argument if it's missing
        self.parser.add_argument('--dup_testset', default=1, type=int, help='Duplicate test set samples for evaluation')

        # Paths for audio and video directories
        self.parser.add_argument('--audio_dir', default='/path/to/correct/audio_directory/', help='Path to the audio files directory')
        self.parser.add_argument('--video_dir', default='./MUSIC_dataset/visual/', help='Path to the video files directory')

    def parse_test_arguments(self):
        """
        Parses command-line arguments for testing.
        """
        args = self.parser.parse_args()
        print("Input arguments:")
        for key, val in vars(args).items():
            print(f"{key:16} {val}")
        return args
