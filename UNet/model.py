import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv1d

class DoubleConv(nn.Module):
    # input => convolution => BatchNorm => ReLU => convolution => BatchNorm => ReLU

    def __init__(self, in_c, out_c):
        super().__init__()
        self.double_conv == nn.Sequential(
            nn.Conv2d(in_c, out_c, padding=1, kernel_size=3),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, padding=1, kernel_size=3),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

        def forward(self, x):
            return self.double_conv(x)

class Down(nn.Module):
    # input => maxpool => double_conv
    def __init__(self, in_c, out_c):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_c, out_c)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    # input => upscale => double_conv
    
    def __init__(self, in_c, out_c):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_c, in_c//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_c, out_c)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        #input of type C x H x W
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX//2, diffX-diffX//2,
                        diffY//2, diffY-diffY//2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # Decoder
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.outconv = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outconv(x)

        return logits

