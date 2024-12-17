"""
A base class for constructing PyTorch MUSIC dataset.
"""

import csv
import random
import librosa
import numpy as np
from PIL import Image
import os

import torch
import torchaudio
import torch.utils.data as torchdata
from torchvision import transforms

from . import video_transforms as video_trans


class BaseDataset(torchdata.Dataset):
    def __init__(self, list_sample, opt, max_sample=-1, process_stage='train'):
        # params
        self.num_frames = opt.num_frames
        self.stride_frames = opt.stride_frames
        self.frameRate = opt.frameRate
        self.imgSize = opt.imgSize
        self.audRate = opt.audRate
        self.audLen = opt.audLen
        self.audSec = 1. * self.audLen / self.audRate  # about 6s
        self.binary_mask = opt.binary_mask

        # STFT params
        self.log_freq = opt.log_freq
        self.stft_frame = opt.stft_frame
        self.stft_hop = opt.stft_hop
        self.HS = opt.stft_frame // 2 + 1
        self.WS = (self.audLen + 1) // self.stft_hop

        self.process_stage = process_stage
        self.seed = opt.seed
        random.seed(self.seed)

        # initialize video transform
        self._init_vtransform()

        # list_sample can be a python list or a csv file of list
        if isinstance(list_sample, str):
            self.list_sample = []
            for row in csv.reader(open(list_sample, 'r'), delimiter=','):
                if len(row) < 2:
                    continue
                self.list_sample.append(row)
        elif isinstance(list_sample, list):
            self.list_sample = list_sample
        else:
            raise ValueError("Error in list_sample format!")

        # Duplication and shuffle for training/validation
        if process_stage == 'train':
            self.list_sample *= opt.dup_trainset
            random.shuffle(self.list_sample)
        elif process_stage == 'val':
            self.list_sample *= opt.dup_validset
        else:
            self.list_sample *= opt.dup_testset

        if max_sample > 0:
            self.list_sample = self.list_sample[:max_sample]

        num_sample = len(self.list_sample)
        print(f"[INFO] Process stage: {process_stage}")
        print(f"[INFO] Number of samples after duplication: {num_sample}")
        if num_sample == 0:
            raise AssertionError("No valid samples found in the dataset.")

    def __len__(self):
        return len(self.list_sample)

    # video transform funcs
    def _init_vtransform(self):
        transform_list = []
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.process_stage == 'train':
            transform_list.append(video_trans.Resize(int(self.imgSize * 1.1), Image.BICUBIC))
            transform_list.append(video_trans.RandomCrop(self.imgSize))
            transform_list.append(video_trans.RandomHorizontalFlip())
        else:
            transform_list.append(video_trans.Resize(self.imgSize, Image.BICUBIC))
            transform_list.append(video_trans.CenterCrop(self.imgSize))

        transform_list.append(video_trans.ToTensor())
        transform_list.append(video_trans.Normalize(mean, std))
        transform_list.append(video_trans.Stack())
        self.vid_transform = transforms.Compose(transform_list)

    def _load_frames(self, paths):
        frames = []
        for path in paths:
            frames.append(self._load_frame(path))
        frames = self.vid_transform(frames)
        return frames

    def _load_frame(self, path):
        """
        Load a video frame or return a dummy black frame with correct size.
        """
        if not os.path.exists(path):
            print(f"[WARNING] Missing video frame: {path}. Using a dummy black frame.")
            return Image.new("RGB", (self.imgSize, self.imgSize), color=(0, 0, 0))
        try:
            img = Image.open(path).convert('RGB')
            return img
        except Exception as e:
            print(f"[ERROR] Failed to load frame {path}: {e}. Using a dummy black frame.")
            return Image.new("RGB", (self.imgSize, self.imgSize), color=(0, 0, 0))


    def _stft(self, audio):
        spec = librosa.stft(audio, n_fft=self.stft_frame, hop_length=self.stft_hop)
        amp = np.abs(spec)
        phase = np.angle(spec)
        return torch.from_numpy(amp), torch.from_numpy(phase)

    def _load_audio(self, path, start_timestamp):
        """
        Load audio data or return a silent placeholder with correct length.
        """
        if not os.path.exists(path):
            print(f"[WARNING] Missing audio file: {path}. Using silent audio.")
            return np.zeros(self.audLen, dtype=np.float32)
        try:
            audio_raw, rate = librosa.load(path, sr=self.audRate, mono=True)
            start = int(start_timestamp * self.audRate)
            end = start + self.audLen
            return audio_raw[start:end] if len(audio_raw) >= end else np.pad(audio_raw, (0, self.audLen - len(audio_raw)))
        except Exception as e:
            print(f"[ERROR] Failed to load audio {path}: {e}. Using silent audio.")
            return np.zeros(self.audLen, dtype=np.float32)

        # Repeat if audio is too short
        if audio_raw.shape[0] < rate * self.audSec:
            n = int(rate * self.audSec / audio_raw.shape[0]) + 1
            audio_raw = np.tile(audio_raw, n)

        # Resample if needed
        if rate > self.audRate:
            if nearest_resample:
                audio_raw = audio_raw[::rate // self.audRate]
            else:
                audio_raw = librosa.resample(audio_raw, orig_sr=rate, target_sr=self.audRate)

        # Clip audio based on the start_timestamp
        start = int(start_timestamp * self.audRate)
        end = start + self.audLen

        if end > len(audio_raw):
            end = len(audio_raw)
            start = max(0, end - self.audLen)  # Adjust start if end exceeds

        return audio_raw[start:end]
                    
    def __getitem__(self, index):
        sample = self.list_sample[index]
        N = len(sample) // 2
        paths_audio = sample[0:N]
        paths_frame = sample[N:]
        start_clip_sec = random.uniform(0, self.audSec)

        try:
            audios = [self._load_audio(path, start_clip_sec) for path in paths_audio]
            frames = self._load_frames(paths_frame)
            amp_mix, mags, phase_mix = self._mix_n_and_stft(audios)
        except Exception as e:
            print(f"Error processing sample at index {index}: {e}")
            return self.dummy_mix_data(N)

        return amp_mix, mags, frames, audios, phase_mix

    def _mix_n_and_stft(self, audios):
        N = len(audios)
        mags = [None for n in range(N)]

        # Mix audios
        audio_mix = np.asarray(audios).sum(axis=0) / N

        # STFT
        amp_mix, phase_mix = self._stft(audio_mix)
        for n in range(N):
            ampN, _ = self._stft(audios[n])
            mags[n] = ampN.unsqueeze(0)

        for n in range(N):
            audios[n] = torch.from_numpy(audios[n])

        return amp_mix.unsqueeze(0), mags, phase_mix.unsqueeze(0)

    def dummy_mix_data(self, N):
        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        mags = [None for n in range(N)]

        amp_mix = torch.zeros(1, self.HS, self.WS)
        phase_mix = torch.zeros(1, self.HS, self.WS)

        for n in range(N):
            frames[n] = torch.zeros(3, self.num_frames, self.imgSize, self.imgSize)
            audios[n] = torch.zeros(self.audLen)
            mags[n] = torch.zeros(1, self.HS, self.WS)

        return amp_mix, mags, frames, audios, phase_mix
