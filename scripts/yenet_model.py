import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# === SRM 濾波器層 (YeNet 核心) ===
class SRM_conv2d(nn.Module):
    def __init__(self, stride=1, padding=0):
        super(SRM_conv2d, self).__init__()
        self.in_channels = 1
        self.out_channels = 30
        self.kernel_size = (5, 5)
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        self.stride = (stride, stride)

        # 定義 3種基礎的高通濾波器 (High-pass filters)
        # 這些濾波器能去除圖片內容，只留下雜訊殘差
        q = [4.0, 12.0, 2.0]
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]
        
        filter1 = np.asarray(filter1, dtype=float) / q[0]
        filter2 = np.asarray(filter2, dtype=float) / q[1]
        filter3 = np.asarray(filter3, dtype=float) / q[2]
        
        # 堆疊成 30 個通道 (這裡簡化處理，重複堆疊以填滿 30)
        filters = []
        for _ in range(10): filters.append(filter1)
        for _ in range(10): filters.append(filter2)
        for _ in range(10): filters.append(filter3)
        
        filters = np.array(filters) # Shape: (30, 5, 5)
        filters = np.expand_dims(filters, 1) # Shape: (30, 1, 5, 5)
        
        # 初始化卷積層權重，並設為可訓練
        self.weight = nn.Parameter(torch.tensor(filters, dtype=torch.float32), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(30), requires_grad=True)

    def forward(self, x):
        return F.conv2d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding)

class TLU(nn.Module):
    def __init__(self, threshold):
        super(TLU, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        # 截斷線性單元：限制數值在 [-T, T] 之間
        # 如果輸入是 0-1，這裡就不起作用；所以輸入必須是 0-255
        return torch.clamp(x, -self.threshold, self.threshold)

class YeNet(nn.Module):
    def __init__(self):
        super(YeNet, self).__init__()
        # 第一層改用 SRM 初始化
        self.preprocessing = SRM_conv2d(stride=1, padding=2)
        self.tlu = TLU(3.0)
        
        self.group1 = nn.Sequential(
            nn.Conv2d(30, 30, 3, padding=1),
            TLU(3.0),
            nn.Conv2d(30, 30, 3, padding=1),
            TLU(3.0),
            nn.AvgPool2d(2, 2)
        )
        
        self.group2 = nn.Sequential(
            nn.Conv2d(30, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        
        self.group3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        
        self.group4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        # x shape: [Batch, 1, 256, 256]
        x = self.preprocessing(x)
        x = self.tlu(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x