# src/video_reader.py
import os
import cv2
import torch
from glob import glob
from tqdm import tqdm
from torchvision import transforms

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
        label = int(parts[3])  # class_index 是第4个字段（从0计数就是 parts[3]）

        id_parts = id_str.split('-')
        if len(id_parts) < 7:
            raise ValueError(f"Invalid ID format: {id_str}")

        participant = id_parts[0]
        record = id_parts[1]
        task = id_parts[2]
        start_frame = int(id_parts[5][1:])  # 去掉 'F'
        end_frame = int(id_parts[6][1:])
        video_path = os.path.join(self.video_root, participant, record, f"{task}.mp4")
        return video_path, start_frame, end_frame, label

    def _read_clip(self, video_path, start_frame, end_frame):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        step = self.frames_per_clip
        clips = []

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

    def load_from_split(self, split_txt, val_split_ratio=0.2):
        with open(split_txt, 'r') as f:
            lines = f.readlines()

        split_index = int(len(lines) * (1 - val_split_ratio))
        train_lines = lines[:split_index]
        val_lines = lines[split_index:]

        train_data = self._load_lines(train_lines)
        val_data = self._load_lines(val_lines)

        return train_data, val_data

    def load_test_set(self, split_txt):
        with open(split_txt, 'r') as f:
            lines = f.readlines()
        return self._load_lines(lines)

    def _load_lines(self, lines):
        data, labels = [], []
        for line in tqdm(lines):
            try:
                video_path, start, end, label = self._parse_line(line)
                if not os.path.exists(video_path):
                    print(f"⚠️ File not found: {video_path}")
                    continue
                clips = self._read_clip(video_path, start, end)
                if not clips:
                    print(f"⚠️ No valid clip from {video_path} [{start}-{end}]")
                for clip in clips:
                    data.append(clip)
                    labels.append(label)
            except Exception as e:
                print(f"⚠️ Failed to process line: {line.strip()}, error: {e}")
        print(f"[INFO] Collected {len(data)} clips from {len(lines)} samples.")
        print(f"[✅] Loaded {len(data)} clips from {len(lines)} lines")
        return data, labels
