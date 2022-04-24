import torch.nn as nn
from typing import Optional

def downsample(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, bias=False),
        nn.BatchNorm2d(out_channels))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_stride: Optional[int] = None):
        super(BasicBlock, self).__init__()
        self.d_stride = 1
        self.downsample = None
        if d_stride is not None:
            self.d_stride = d_stride
            self.downsample = downsample(in_channels, out_channels, d_stride)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=self.d_stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        out = self.relu(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels, width, out_channels, d_stride: Optional[int] = None):
        super(Bottleneck, self).__init__()
        self.d_stride = 1
        self.downsample = None
        if d_stride is not None:
            self.d_stride = d_stride
            self.downsample = downsample(in_channels, out_channels, d_stride)
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=(1, 1), stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=(3, 3), stride=self.d_stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=(1, 1), stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)        

    def forward(self, x):
        identity = x
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        out = self.relu(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        )
        self.conv3 = nn.Sequential(
            BasicBlock(64, 128, 2),
            BasicBlock(128, 128)
        )
        self.conv4 = nn.Sequential(
            BasicBlock(128, 256, 2),
            BasicBlock(256, 256)
        )
        self.conv5 = nn.Sequential(
            BasicBlock(256, 512, 2),
            BasicBlock(512, 512)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, 5)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.avgpool(out)
        out = self.fc(out.reshape(out.shape[0], -1))
        return out


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.input_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.stage_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            Bottleneck(64, 64, 256, 1),
            Bottleneck(256, 64, 256),
            Bottleneck(256, 64, 256)
        )
        self.stage_2 = nn.Sequential(
            Bottleneck(256, 128, 512, 2),
            Bottleneck(512, 128, 512),
            Bottleneck(512, 128, 512),
            Bottleneck(512, 128, 512)
        )
        self.stage_3 = nn.Sequential(
            Bottleneck(512, 256, 1024, 2),
            Bottleneck(1024, 256, 1024),
            Bottleneck(1024, 256, 1024),
            Bottleneck(1024, 256, 1024),
            Bottleneck(1024, 256, 1024),
            Bottleneck(1024, 256, 1024)
        )
        self.stage_4 = nn.Sequential(
            Bottleneck(1024, 512, 2048, 2),
            Bottleneck(2048, 512, 2048),
            Bottleneck(2048, 512, 2048)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(2048, 5)

    def forward(self, x):
        out = self.input_stem(x)
        out = self.stage_1(out)
        out = self.stage_2(out)
        out = self.stage_3(out)
        out = self.stage_4(out)
        out = self.avgpool(out)
        out = self.fc(out.reshape(out.shape[0], -1))
        return out