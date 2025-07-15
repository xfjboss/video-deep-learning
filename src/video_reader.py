import os
import cv2
import torch
import pandas as pd
from torchvision import transforms
from glob import glob
from tqdm import tqdm


class VideoReader:
    def __init__(self, video_dir, label_csv, class_index_csv, max_frames_per_action=30):
        self.video_dir = video_dir
        self.label_csv = pd.read_csv(label_csv)
        self.class_map = self._load_class_index(class_index_csv)
        self.max_frames_per_action = max_frames_per_action

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
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
        # 搜索所有子文件夹中的 video_id.mp4
        pattern = os.path.join(self.video_dir, "**", f"{video_id}.mp4")
        matches = glob(pattern, recursive=True)
        if matches:
            return matches[0]
        else:
            return None

    def read_action_segment(self, video_path, start_frame, end_frame):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 限制读取帧数
        step = max(1, (end_frame - start_frame) // self.max_frames_per_action)

        frames = []
        for i in range(start_frame, min(end_frame, total_frames), step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = self.transform(frame)
            frames.append(frame_tensor)

        cap.release()
        return frames

    def load_dataset(self):
        data = []
        labels = []
        missing = 0

        print("📥 Loading labeled action segments...")
        for idx, row in tqdm(self.label_csv.iterrows(), total=len(self.label_csv)):
            video_id = row["video_id"]
            start = int(row["start_frame"])
            end = int(row["end_frame"])
            key = f"{row['verb']}_{row['noun']}"
            label = self.class_map.get(key)

            if label is None:
                continue

            video_path = self._get_video_path(video_id)
            if video_path is None or not os.path.exists(video_path):
                missing += 1
                continue

            frames = self.read_action_segment(video_path, start, end)
            for f in frames:
                data.append(f)
                labels.append(label)

        print(f"✅ Loaded {len(data)} frames, skipped {missing} missing videos.")
        return data, labels, self.class_map