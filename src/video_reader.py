import os
import cv2
import torch
import pandas as pd
from torchvision import transforms
from glob import glob
from tqdm import tqdm


class VideoReader:
    def __init__(self, video_dir, label_csv, class_index_csv, frames_per_clip=16, model_type='2d'):
        self.video_dir = video_dir
        self.label_csv = pd.read_csv(label_csv)
        self.frames_per_clip = frames_per_clip
        self.model_type = model_type
        self.class_map = self._load_class_index(class_index_csv)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor()
        ])

    def _load_class_index(self, csv_path):
        df = pd.read_csv(csv_path)
        class_map = {}
        for _, row in df.iterrows():
            key = f"{row['verb']}_{row['noun']}"
            class_map[key] = int(row['class_index'])
        return class_map

    def _get_video_path(self, video_id):
        pattern = os.path.join(self.video_dir, "**", f"{video_id}.mp4")
        matches = glob(pattern, recursive=True)
        return matches[0] if matches else None

    def read_action_clip(self, video_path, start_frame, end_frame):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = end_frame - start_frame
        if duration < self.frames_per_clip:
            return None  # 跳过过短片段

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
                frame_tensor = self.transform(frame)
                frames.append(frame_tensor)
            if len(frames) == step:
                clip_tensor = torch.stack(frames)  # [T, C, H, W]
                clips.append(clip_tensor)
        cap.release()
        return clips

    def load_dataset(self):
        data = []
        labels = []
        for _, row in tqdm(self.label_csv.iterrows(), total=len(self.label_csv)):
            video_id = row["video_id"]
            start = int(row["start_frame"])
            end = int(row["end_frame"])
            key = f"{row['verb']}_{row['noun']}"
            label = self.class_map.get(key)

            if label is None:
                continue
            video_path = self._get_video_path(video_id)
            if not video_path:
                continue

            clips = self.read_action_clip(video_path, start, end)
            if clips:
                for clip in clips:
                    if self.model_type == '3d':
                        data.append(clip.permute(1, 0, 2, 3))  # [C, T, H, W]
                    else:
                        for frame in clip:  # 用作2D输入
                            data.append(frame)
                    labels.extend([label] * (len(clips) if self.model_type == '3d' else len(clip)))
        return data, labels, self.class_map
