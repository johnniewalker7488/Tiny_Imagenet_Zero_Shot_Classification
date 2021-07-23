import numpy as np
import torch
import torchvision
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super(CNNBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn2(self.conv(self.bn1(x))))


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.ReLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid())

    def forward(self, x):
        return x * self.se(x)


class SEResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 reduction=4,
                 survival_prob=0.8):
        super(SEResBlock, self).__init__()
        self.stride = stride
        self.survival_prob = survival_prob
        reduced_dim = int(in_channels / reduction)
        self.conv1 = CNNBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.se = SqueezeExcitation(in_channels, in_channels)
        self.conv2 = CNNBlock(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        self.downscale = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

    def stochastic_depth(self, x):
        if not self.training:
            return x

        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = self.se(x)
        x = self.conv2(x)
        inputs = self.downscale(inputs)
        stoch_x = self.stochastic_depth(x)
        return x + inputs


class SEResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SEResNet, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.conv1 = CNNBlock(in_channels, 64, kernel_size=7, stride=1, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res1_1 = SEResBlock(64, 64, 1)  # 32x32
        self.res1_2 = SEResBlock(64, 64, 1)

        self.res2_1 = SEResBlock(64, 64, 2)  # 16x16
        self.res2_2 = SEResBlock(64, 256, 1)

        self.res3_1 = SEResBlock(256, 256, 2)  # 8x8
        self.res3_2 = SEResBlock(256, 512, 1)

        self.res4_1 = SEResBlock(512, 512, 1)  # 8x8
        self.res4_2 = SEResBlock(512, 1024, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Linear(1024, 300)
        self.bn = nn.BatchNorm1d(300)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(300, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.res1_1(x)
        x = self.res1_2(x)
        x = self.res2_1(x)
        x = self.res2_2(x)
        x = self.res3_1(x)
        x = self.res3_2(x)
        x = self.res4_1(x)
        x = self.res4_2(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        emb = self.embedding(x)
        drop = self.relu(emb)
        out = self.classifier(self.bn(drop))
        return out, emb