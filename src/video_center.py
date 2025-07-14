import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

from video_reader import VideoReader
from video_model import VideoModel

def main():
    # 配置路径
    data_path = "C:/xf/xf/research/video_task/data"

    # Step 1: 加载数据
    reader = VideoReader(data_path)
    frames, labels, label_map = reader.load_dataset()

    print(f"共加载帧数：{len(frames)}, 类别数：{len(label_map)}")

    # Step 2: 转为 DataLoader
    X = torch.stack(frames)
    y = torch.tensor(labels)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Step 3: 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoModel(num_classes=len(label_map)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)  # 只训练线性层

    # Step 4: 训练模型
    model.train()
    for epoch in range(3):  # Day1 简单训练几个 epoch
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} finished, Loss: {loss.item():.4f}")

    # Step 5: 模型评估
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_true.extend(batch_y.tolist())

    acc = accuracy_score(all_true, all_preds)
    cm = confusion_matrix(all_true, all_preds)
    print(f"\n✅ Accuracy: {acc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

if __name__ == "__main__":
    main()
