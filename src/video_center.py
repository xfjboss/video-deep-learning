# src/video_center.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix
from video_reader import VideoReader
from video_model import VideoModel


def run(model_type='3d', split='1', epochs=5):
    video_root = r'C:\xf\xf\research\video_task\data\raw_video'
    train_txt = f'train{split}.txt'
    test_txt = f'test{split}.txt'

    reader = VideoReader(video_root, frames_per_clip=16, model_type=model_type)
    train_data, val_data = reader.load_from_split(train_txt)
    test_data, test_labels = reader.load_test_set(test_txt)

    def to_loader(data, labels, batch_size):
        X = torch.stack(data)
        y = torch.tensor(labels)
        return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

    batch_size = 8 if model_type == '3d' else 32
    train_loader = to_loader(*train_data, batch_size)
    val_loader = to_loader(*val_data, batch_size)
    test_loader = to_loader(test_data, test_labels, batch_size)

    model = VideoModel(num_classes=max(test_labels) + 1, model_type=model_type).cuda()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        print(f"[Epoch {epoch+1}] loss: {loss.item():.4f}")

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.cuda()
                out = model(x).argmax(dim=1).cpu()
                preds.extend(out.tolist())
                targets.extend(y.tolist())
        acc = accuracy_score(targets, preds)
        print(f"🧪 Val Accuracy: {acc:.4f}")

    # 最终在测试集评估
    preds, targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.cuda()
            out = model(x).argmax(dim=1).cpu()
            preds.extend(out.tolist())
            targets.extend(y.tolist())

    print(f"🎯 Test Accuracy: {accuracy_score(targets, preds):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(targets, preds))


if __name__ == '__main__':
    run(model_type='2d', split='1', epochs=10)
    run(model_type='3d', split='1', epochs=10)
