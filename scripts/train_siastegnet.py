import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

from siastegnet_model import SiaStegNet

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "training_data")
WEIGHTS_DIR = os.path.join(ROOT_DIR, "weights")
SAVE_PATH = os.path.join(WEIGHTS_DIR, "siastegnet_best.pth")

class StegoDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        self.cover_dir = os.path.join(root_dir, "cover")
        self.stego_dir = os.path.join(root_dir, "stego")
        self.mode = mode
        
        self.covers = sorted(os.listdir(self.cover_dir))
        self.stegos = sorted(os.listdir(self.stego_dir))
        
        min_len = min(len(self.covers), len(self.stegos))
        self.covers = self.covers[:min_len]
        self.stegos = self.stegos[:min_len]
        self.data = []
        
        for f in self.covers:
            self.data.append((os.path.join(self.cover_dir, f), 0))
        for f in self.stegos:
            self.data.append((os.path.join(self.stego_dir, f), 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert('L')
            
            if self.mode == 'train':
                rot = random.choice([0, 90, 180, 270])
                if rot != 0:
                    image = image.rotate(rot)
                if random.random() > 0.5:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
            
            # 放大數值到 0-255
            image_tensor = transforms.ToTensor()(image) * 255.0
            return image_tensor, label
        except Exception as e:
            return torch.randn(1, 256, 256), label

def train():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🚀 [SiaStegNet] 訓練開始...")
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ 找不到 {DATA_DIR}")
        return

    full_dataset = StegoDataset(DATA_DIR, mode='train')
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
    
    model = SiaStegNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    epochs = 20
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  💾 保存最佳 SiaStegNet (Acc: {val_acc:.2f}%)")

if __name__ == "__main__":
    train()