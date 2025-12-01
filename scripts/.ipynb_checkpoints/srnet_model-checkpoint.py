import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerType1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(LayerType1, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class LayerType2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(LayerType2, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Identity()
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return out

class LayerType3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(LayerType3, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.avg_pool = nn.AvgPool2d(3, 2, padding=1)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=2),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.avg_pool(self.bn2(self.conv2(out)))
        out += residual
        return out

class LayerType4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(LayerType4, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.global_pool(out)
        return out.view(out.size(0), -1)

class SRNet(nn.Module):
    def __init__(self):
        super(SRNet, self).__init__()
        self.layer1 = LayerType1(1, 64)
        self.layer2 = LayerType1(64, 16)
        self.layer3 = LayerType2(16, 16)
        self.layer4 = LayerType2(16, 16)
        self.layer5 = LayerType2(16, 16)
        self.layer6 = LayerType2(16, 16)
        self.layer7 = LayerType2(16, 16)
        self.layer8 = LayerType3(16, 16)
        self.layer9 = LayerType3(16, 64)
        self.layer10 = LayerType3(64, 128)
        self.layer11 = LayerType3(128, 256)
        self.layer12 = LayerType4(256, 512)
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.fc(x)
        return x