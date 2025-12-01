import torch
import torch.nn as nn

class SiaStegNet(nn.Module):
    def __init__(self):
        super(SiaStegNet, self).__init__()
        # 簡化版 SiaStegNet 結構，強調特徵提取
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(10, 10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 128
            
            nn.Conv2d(10, 20, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(20, 20, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 64
            
            nn.Conv2d(20, 40, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(40, 40, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 32
            
            nn.Conv2d(40, 80, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(80, 80, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1) 
        )
        
        self.fc = nn.Sequential(
            nn.Linear(80, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        feat = self.conv_block(x)
        feat = feat.view(feat.size(0), -1)
        out = self.fc(feat)
        return out