import os
import cv2
import torch
from torchvision import transforms
from glob import glob

class VideoReader:
    def __init__(self, video_dir, label_map=None):
        self.video_dir = video_dir
        self.label_map = label_map or {}
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def read_video(self, video_path, max_frames=300):
        cap = cv2.VideoCapture(video_path)
        frames = []
        count = 0
        while cap.isOpened() and count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = self.transform(frame)
            frames.append(frame_tensor)
            count += 1
        cap.release()
        return torch.stack(frames)

    def load_dataset(self):
        """
        假设每个子文件夹是一个动作类别
        data/
            cutting/
            stirring/
            ...
        """
        data = []
        labels = []
        class_folders = glob(os.path.join(self.video_dir, "*"))
        for class_folder in class_folders:
            class_name = os.path.basename(class_folder)
            label = self.label_map.get(class_name, len(self.label_map))
            self.label_map[class_name] = label
            for video_file in glob(os.path.join(class_folder, "*.avi")):
                frames = self.read_video(video_file)
                for frame in frames:
                    data.append(frame)
                    labels.append(label)
        return data, labels, self.label_map
