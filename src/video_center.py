# src/video_center.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from video_reader import VideoDataset, VideoReader
from video_model import VideoModel

BC = 8
EP = 50
LR = 1e-4
def run(model_type='3d', epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    video_root = 'C:/xf/xf/research/video_task/data/Raw_Videos'
    train_txt = 'C:/xf/xf/research/video_task/data/Action_Annotations/train_split1.txt'
    test_txt = 'C:/xf/xf/research/video_task/data/Action_Annotations/test_split1.txt'
    reader = VideoReader(video_root, frames_per_clip=16, model_type=model_type)

    train_data, val_data, test_data = reader.load_sampled_data(
        train_txt, test_txt,
        train_n=1000, val_n=200, test_n=200
)

    def to_loader(data, labels, batch_size=BC, device=None):
        dataset = VideoDataset(data, labels, device=device)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    batch_size = BC if model_type == '3d' else 8
    train_loader = to_loader(*train_data, batch_size)
    val_loader = to_loader(*val_data, batch_size)
    test_loader = to_loader(*test_data, batch_size)

    # Step 2: Build model
    num_classes = max([*train_data[1], *val_data[1], *test_data[1]]) + 1
    model = VideoModel(num_classes=num_classes, model_type=model_type).to(device)

    optimizer = optim.Adam(model.classifier.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Step 3: Train
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] loss: {total_loss/len(train_loader):.4f}")

        model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
            acc = correct / total
            print(f"🧪 Val Accuracy: {acc:.4f}")

    # Step 4: Test
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        print(f"🎯 Test Accuracy: {correct/total:.4f}")


if __name__ == '__main__':
    run(model_type='2d', epochs=EP)