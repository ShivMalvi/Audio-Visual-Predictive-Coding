"""
Mixed music PyTorch dataset.
"""

import os
import random, csv, torch
import numpy as np
from .base import BaseDataset
from PIL import Image
from torchvision import transforms

class MUSICMixDataset(BaseDataset):
    def __init__(self, list_sample, args, process_stage='train'):
        # Set `num_mix` from args
        self.num_mix = args.num_mix  # Retrieve num_mix from args
        self.args = args
        self.process_stage = process_stage

        # If list_sample is a string, load it as a CSV file
        if isinstance(list_sample, str):
            self.list_sample = []
            for row in csv.reader(open(list_sample, 'r'), delimiter=','):
                if len(row) < 4:  # Ensure there are at least 4 elements
                    print(f"[WARNING] Skipping invalid row: {row}")
                    continue
                self.list_sample.append(row)
            with open(list_sample, "r") as f:
                for line in f:
                    parts = line.strip().split(",")  # Assuming CSV format
                    self.list_sample.append(parts)
        elif isinstance(list_sample, list):
            self.list_sample = list_sample
        else:
            raise ValueError("Error: list_sample should be a file path (str) or list.")

        self.video_ids = [os.path.splitext(os.path.basename(row[0]))[0] for row in self.list_sample]
        print(f"[INFO] Loaded {len(self.video_ids)} video IDs for {process_stage} stage.")

        # Call the parent class initialization
        super().__init__(self.list_sample, args, process_stage=process_stage)

    

    def __getitem__(self, index):
        N = self.args.num_mix  # Number of sources to mix
        audios = [np.zeros(self.audLen, dtype=np.float32) for _ in range(N)]
        frames = [torch.zeros(3, self.imgSize, self.imgSize) for _ in range(N)]  # Placeholder tensors
        infos = []

        # Transform to convert PIL images to tensors
        transform = transforms.Compose([
            transforms.Resize((self.imgSize, self.imgSize)),
            transforms.ToTensor(),
        ])

        # Safeguard index for valid samples
        for _ in range(N):
            if index < len(self.list_sample):
                infos.append(self.list_sample[index])
            else:
                infos.append(("", "", 0, 0))  # Fallback for invalid index

        for n, infoN in enumerate(infos):
            # Safeguard for incomplete data
            if len(infoN) < 4:
                print(f"[WARNING] Malformed data at index {index}: {infoN}")
                path_audioN, path_frameN, count_chunksN, count_framesN = "", "", 0, 0
            else:
                path_audioN, path_frameN, count_chunksN, count_framesN = infoN[:4]  # Safely unpack first 4 elements
            
            if not self.process_stage == 'train':
                random.seed(index + n)
            
            idx_chunk = random.randint(0, max(0, int(count_chunksN) - 2))
            if idx_chunk == 0:
                start_clip_sec = random.randint(10, 20 - 6 - 1)
            else:
                start_clip_sec = random.randint(0, 20 - 6 - 1)
            
            # Define file paths
            video_id = self.video_ids[index]
            audio_path = os.path.join(self.args.audio_dir, f"{video_id}.wav")
            video_path = os.path.join(self.args.video_dir, f"{video_id}.mp4")
            
            try:
                audios[n] = self._load_audio(audio_path, start_timestamp=start_clip_sec)
            except Exception as e:
                print(f"[WARNING] Failed to load audio at {audio_path}: {e}")

            try:
                frame_image = self._load_frame(video_path)  # Load PIL Image
                frames[n] = transform(frame_image)  # Convert to tensor
            except Exception as e:
                print(f"[WARNING] Failed to load video frame at {video_path}: {e}")

        # Mix audio and compute STFT
        mag_mix, mags, phase_mix = self._mix_n_and_stft(audios)

        # Return a dictionary containing audio and frame data
        return {
            'mag_mix': mag_mix,
            'frames': torch.stack(frames),  # Stack tensors for frames
            'mags': mags,
            'audios': audios
        }

    def __len__(self):
        return len(self.video_ids)

    def normalize(self, audio_data, re_factor=0.8):
        """
        Normalize audio data to [-1, 1].

        Args:
            audio_data: Raw audio waveform.
            re_factor: Random scaling factor for training.

        Returns:
            Normalized audio data.
        """
        EPS = 1e-3
        min_data = audio_data.min()
        audio_data -= min_data
        max_data = audio_data.max()
        audio_data /= max_data + EPS
        audio_data -= 0.5
        audio_data *= 2

        if self.process_stage == 'train':
            re_factor = random.random() + 0.5  # Range: 0.5-1.5
            audio_data *= re_factor

        return audio_data.clip(-1, 1)
