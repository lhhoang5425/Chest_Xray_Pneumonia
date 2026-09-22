import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. Định nghĩa kiến trúc mạng
class ChestXRayBinaryNet(nn.Module):
    def __init__(self):
        super(ChestXRayBinaryNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- BAT DAU HUAN LUYEN MODEL TREN {str(device).upper()} ---")

    # Đường dẫn thư mục gốc dự án (cha của src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    cand_train = [
        os.path.join(project_root, "chest_xray", "chest_xray", "train"),
        os.path.join(project_root, "chest_xray", "train")
    ]
    train_dir = cand_train[0] if os.path.exists(cand_train[0]) else cand_train[1]

    cand_val = [
        os.path.join(project_root, "chest_xray", "chest_xray", "val"),
        os.path.join(project_root, "chest_xray", "val")
    ]
    val_dir = cand_val[0] if os.path.exists(cand_val[0]) else cand_val[1]

    print("Thu muc train:", train_dir)

    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform_val)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_counts = [train_dataset.targets.count(i) for i in range(len(train_dataset.classes))]
    normal_cnt = class_counts[0]
    pneumonia_cnt = class_counts[1]
    pos_weight = torch.tensor([normal_cnt / pneumonia_cnt]).to(device)

    print(f"So luong anh: NORMAL={normal_cnt}, PNEUMONIA={pneumonia_cnt}")
    print(f"Trong so can bang pos_weight: {pos_weight.item():.3f}")

    model = ChestXRayBinaryNet().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    epochs = 5
    saved_models_dir = os.path.join(project_root, "saved_models")
    os.makedirs(saved_models_dir, exist_ok=True)
    save_path = os.path.join(saved_models_dir, "best_chest_xray_model.pth")

    best_acc = 0.0

    for ep in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, lbls in train_loader:
            imgs = imgs.to(device)
            lbls = lbls.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outs = model(imgs)
            loss = criterion(outs, lbls)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(outs) >= 0.5).float()
            train_correct += (preds == lbls).sum().item()
            train_total += lbls.size(0)

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                lbls = lbls.float().unsqueeze(1).to(device)
                outs = model(imgs)
                preds = (torch.sigmoid(outs) >= 0.5).float()
                val_correct += (preds == lbls).sum().item()
                val_total += lbls.size(0)

        tr_acc = train_correct / train_total
        vl_acc = val_correct / max(val_total, 1)
        print(f"Epoch {ep:02d}/{epochs:02d} - Train Loss: {train_loss/train_total:.4f} | Train Acc: {tr_acc*100:.2f}% | Val Acc: {vl_acc*100:.2f}%")

        # Lưu checkpoint
        torch.save(model.state_dict(), save_path)
        print(f"--> Da cap nhat trong so model vao: {save_path}")

    print("=== HOAN TAT HUAN LUYEN VA LUU TRONG SO MODEL! ===")

if __name__ == '__main__':
    main()
