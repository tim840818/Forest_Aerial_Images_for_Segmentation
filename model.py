import numpy as np
import cv2
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.double_conv(x)

class OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutputLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        return x

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=INPUT_CHANNEL, out_channels=OUTPUT_CHANNEL,
        dropout=0.1,
        n_filters=[32, 64, 128, 256] # [16, 32, 64, 128]
    ):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        channels = in_channels
        for n_filter in n_filters[:-1]:
            self.encoder.append(DoubleConv(channels, n_filter, dropout))
            channels = n_filter
        self.bottleneck = DoubleConv(channels, n_filters[-1], dropout)
        channels = n_filters[-1]
        for n_filter in reversed(n_filters[:-1]):
            self.decoder.append(nn.ConvTranspose2d(channels, n_filter, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(channels, n_filter, dropout))
            channels = n_filter
        # self.out = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.out = OutputLayer(channels, out_channels)

    def forward(self, x):
        skip_connections = []
        for encoder in self.encoder:
            x = encoder(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.bottleneck(x)
        for i in range(0, len(self.decoder), 2): # upsaling (con2dtrans) -> concat -> doubleconv
            x_upscale = self.decoder[i](x)
            concat = torch.cat((skip_connections.pop(), x_upscale), dim=1) # concat along channel axis
            x = self.decoder[i+1](concat)
        x = self.out(x)
        return x