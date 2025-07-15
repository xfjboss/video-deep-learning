# src/video_center.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from video_reader import VideoReader
from video_model import VideoModel


def main():
    # 配置参数
    model_type = '3d'  # '2d' or '3d'
    batch_size = 8 if model_type == '3d' else 32
    clip_len = 16 if model_type == '3d' else 1

    reader = VideoReader(
        video_dir="",
        label_csv="",
        class_index_csv="",
        frames_per_clip=clip_len,
        model_type=model_type
    )

    frames, labels, label_map = reader.load_dataset()
    print(f"📦 Loaded {len(frames)} samples.")

    X = torch.stack(frames)
    y = torch.tensor(labels)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoModel(num_classes=len(label_map), model_type=model_type).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)

    model.train()
    for epoch in range(3):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    # 评估
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            preds = torch.argmax(model(batch_x), dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_true.extend(batch_y.tolist())

    acc = accuracy_score(all_true, all_preds)
    print(f"✅ Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(all_true, all_preds))


if __name__ == "__main__":
    main()
