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
    
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        self.conv1 = CNNBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = CNNBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.downscale = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
                
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        x = self.downscale(x)
        return out + x
    
    
class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = CNNBlock(3, 64, kernel_size=7, stride=1, padding=3) # 64x64
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)   # 32x32
        
        self.res1_1 = ResBlock(64, 64, 1)                  # 32x32
        self.res1_2 = ResBlock(64, 64, 1)
                
        self.res2_1 = ResBlock(64, 64, 2)                # 16x16
        self.res2_2 = ResBlock(64, 256, 1)
        
        self.res3_1 = ResBlock(256, 256, 2)               # 8x8
        self.res3_2 = ResBlock(256, 512, 1)
        
        self.res4_1 = ResBlock(512, 512, 1)              # 8x8
        self.res4_2 = ResBlock(512, 1024, 1)
        self.drop2d = nn.Dropout2d(0.3)
        
        
        self.avgpool = nn.AvgPool2d(8)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024, 300, bias=False)
        
        self.bn1 = nn.BatchNorm1d(300)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(300, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool1(out)
        
        out = self.res1_1(out)
        out = self.res1_2(out)
        
        out = self.res2_1(out)
        out = self.res2_2(out)
        
        out = self.res3_1(out)
        out = self.res3_2(out)
        
        out = self.res4_1(out)
        out = self.res4_2(out)
#         out = self.drop2d(out)
        
        out = self.avgpool(out)
        out = self.flatten(out)
        emb = self.linear1(out)
        bn1 = self.bn1(emb)
        act = self.activation(bn1)
        out = self.dropout(act)
        out = self.linear2(out)
        return out, emb