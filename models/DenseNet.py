import numpy as np
import torch
import torchvision
import torch.nn as nn


class DenseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.bn(self.relu(self.bn(self.conv(x))))
    

class DenseConv1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseConv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv.weight)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.bn(self.relu(self.bn(self.conv(x))))
        

        
class DenseResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseResBlock, self).__init__()
        self.conv1 = DenseConv(in_channels, in_channels*2)
        self.conv2 = DenseConv(in_channels*2, in_channels*4)
        self.conv3 = DenseConv(in_channels*4, in_channels*8)
        self.conv4 = DenseConv(in_channels*8, in_channels*16)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.downscale = nn.Conv2d(in_channels, in_channels*16, kernel_size=3, stride=2, padding=1)
        self.upscale = nn.Conv2d(in_channels*16, out_channels, kernel_size=1, stride=1, padding=0)
        
        
    def forward(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        inputs = self.downscale(inputs)
        skip = x + inputs
        out = self.upscale(skip)
        return out

    
class DenseNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DenseNet, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.conv = DenseConv(in_channels, 32)
        self.res1 = DenseResBlock(32, 64)
        self.res2 = DenseResBlock(64, 64)
        self.res3 = DenseResBlock(64, 64)
        
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.embedding = nn.Linear(1024, 300)
        self.bn = nn.BatchNorm1d(300)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(300, num_classes)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        emb = self.embedding(x)
        drop = self.relu(emb)
        out = self.classifier(self.bn(drop))
        return out, emb