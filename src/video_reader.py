# src/video_reader.py
import os
import cv2
import torch
import random
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset


class VideoReader:
    def __init__(self, video_root, frames_per_clip=16, model_type='3d'):
        self.video_root = video_root
        self.frames_per_clip = frames_per_clip
        self.model_type = model_type
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor()
        ])

    def _parse_line(self, line):
        parts = line.strip().split()
        if len(parts) < 4:
            raise ValueError(f"Invalid line format: {line}")

        id_str = parts[0]
        label = int(parts[3])

        id_parts = id_str.split('-')
        if len(id_parts) < 7:
            raise ValueError(f"Invalid ID format: {id_str}")

        start_frame = int(id_parts[5][1:])
        end_frame = int(id_parts[6][1:])
        video_id = '-'.join(id_parts[:3])
        video_path = os.path.join(self.video_root, f"{video_id}.mp4")

        return video_path, start_frame, end_frame, label

    def _read_clip(self, video_path, start_frame, end_frame):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        clips = []
        step = self.frames_per_clip
        for clip_start in range(start_frame, end_frame - step + 1, step):
            frames = []
            for i in range(clip_start, clip_start + step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.transform(frame)
                frames.append(frame)
            if len(frames) == step:
                clip = torch.stack(frames)  # [T, C, H, W]
                if self.model_type == '3d':
                    clip = clip.permute(1, 0, 2, 3)  # [C, T, H, W]
                clips.append(clip)
        cap.release()
        return clips

    def load_sampled_data(self, train_txt, test_txt, train_n=200, val_n=40, test_n=60):
        # 读取 train1.txt
        with open(train_txt, 'r') as f:
            train_lines = f.readlines()
        random.shuffle(train_lines)
        train_lines_selected = train_lines[:train_n]
        val_lines_selected = train_lines[train_n:train_n+val_n]

        # 读取 test1.txt
        with open(test_txt, 'r') as f:
            test_lines = f.readlines()
        random.shuffle(test_lines)
        test_lines_selected = test_lines[:test_n]

        # 加载数据
        train_data = self._load_lines(train_lines_selected)
        val_data = self._load_lines(val_lines_selected)
        test_data = self._load_lines(test_lines_selected)

        return train_data, val_data, test_data

    def _load_lines(self, lines):
        data, labels = [], []
        for line in tqdm(lines):
            try:
                video_path, start, end, label = self._parse_line(line)
                if not os.path.exists(video_path):
                    continue
                clips = self._read_clip(video_path, start, end)
                for clip in clips:
                    data.append(clip)
                    labels.append(label)
            except Exception as e:
                print(f"⚠️ Failed to process line: {line.strip()}, error: {e}")
        return data, labels

class VideoDataset(Dataset):
    def __init__(self, data, labels, device=None):
        self.data = data
        self.labels = labels
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.device:  # 避免预加载整个数据
            x = x.to(self.device)
            y = torch.tensor(y).to(self.device)
        return x, y