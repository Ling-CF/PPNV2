import torch.nn as nn
from FIRTorch import firwin
import torch
import torch.nn.functional as F
import sys

def design_filter(x, cutoff, pad_mode='replicate', numtabs=5, stride=1):
    channels = x.size(1)
    f = firwin(numtabs**2, cutoff, window='hamming')
    f = f.reshape(f.size(0), 1, numtabs, numtabs).float()
    if f.size(0) == 1:
        f = f.repeat(channels, 1, 1, 1)
    padding = numtabs // 2
    x = F.pad(x, (padding,)*4, mode=pad_mode, value=0)
    out = F.conv2d(x, weight=f, groups=channels, stride=stride)
    return out


class ScaleShift(nn.Module):
    def __init__(self, channels):
        super(ScaleShift, self).__init__()
        self.scale_shift = nn.Conv2d(channels, 2 * channels, kernel_size=(3, 3), padding=1)
        self.scale_factor = nn.Parameter(torch.ones(1, requires_grad=True))
        self.DS = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x1, x2):
        scale_x2, shift_x2 = torch.split(self.scale_shift(x2), x2.size(1), dim=1)
        scale_x2, shift_x2 = torch.sigmoid(scale_x2)* self.scale_factor, torch.tanh(shift_x2)
        x1 = x1 * scale_x2  + shift_x2
        x1 = self.DS(x1)
        return x1

class Concat(nn.Module):
    def __init__(self, channels):
        super(Concat, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels=2*channels, out_channels=channels, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x1, x2):
        out = self.Conv(torch.cat([x1, x2], dim=1))
        return out

class Add(nn.Module):
    def __init__(self, channels):
        super(Add, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x1, x2):
        out = self.Conv(x1 + x2)
        return out

def MergeStyle(mode, channels):
    assert mode in ['modulate', 'concat', 'add']
    if mode == 'modulate':
        return ScaleShift(channels)
    elif mode == 'concat':
        return Concat(channels)
    elif mode == 'add':
        return Add(channels)
    else:
        print('Unknow mode. Only "modulate", "concat" and "add" are available')
        sys.exit()

class UpPropagate(nn.Module):
    def __init__(self, in_channel, out_channel, pad_mode):
        super(UpPropagate, self).__init__()
        self.FE = nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=1, padding_mode=pad_mode)
        self.DS = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), padding=1, stride=1)
        self.cutoff = nn.Parameter(torch.ones(out_channel, 1, requires_grad=True, dtype=torch.float) * 2.0)
        self.relu = nn.LeakyReLU(inplace=True)
        self.relu2 = nn.LeakyReLU()


    def forward(self, x):
        x = self.FE(x)
        x = self.relu(x)

        cutoff = self.relu2(self.cutoff)
        cutoff = torch.sigmoid(cutoff)
        x = design_filter(x, cutoff=cutoff, pad_mode='replicate', numtabs=5, stride=2) # stride

        x = self.DS(x)
        x = self.relu(x)
        return x

class ZeroInterpolate(nn.Module):
    def __init__(self, channels):
        super(ZeroInterpolate, self).__init__()

        self.weight = torch.tensor(
            [[0.0347, 0.0498, 0.0502, 0.0599, 0.0502, 0.0498, 0.0347],
             [0.0498, 0., 0.0804, 0., 0.0804, 0., 0.0498],
             [0.0502, 0.0804, 0.1122, 0.1797, 0.1122, 0.0804, 0.0502],
             [0.0599, 0., 0.1797, 1., 0.1797, 0., 0.0599],
             [0.0502, 0.0804, 0.1122, 0.1797, 0.1122, 0.0804, 0.0502],
             [0.0498, 0., 0.0804, 0., 0.0804, 0., 0.0498],
             [0.0347, 0.0498, 0.0502, 0.0599, 0.0502, 0.0498, 0.0347]],
            requires_grad=True,
            dtype=torch.float
        )
        self.weight = self.weight.reshape(1, 1, 7, 7).repeat(channels, 1, 1, 1)
        self.weight = nn.Parameter(self.weight)
        self.channels = channels

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        zeros = torch.zeros_like(x)
        zeros[:, :, ::2, ::2] = 1.
        x = x * zeros

        x = F.pad(x, pad=[3, 3, 3, 3], mode='reflect')
        x = F.conv2d(x, self.weight, groups=self.channels)

        x = self.relu(x)
        return x



class DownPropagate(nn.Module):
    def __init__(self, in_channel, out_channel, pad_mode):
        super(DownPropagate, self).__init__()
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample = ZeroInterpolate(channels=in_channel)
        self.DS = nn.Conv2d(in_channel, in_channel, kernel_size=(3, 3), padding=1, padding_mode=pad_mode)
        self.CST = nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=1, padding_mode=pad_mode)
        self.relu = nn.LeakyReLU(inplace=True)

        self.cutoff = nn.Parameter(torch.ones(in_channel, 1, requires_grad=True, dtype=torch.float) * 2.0)

        self.relu2 = nn.LeakyReLU()

    def forward(self, x):
        x = self.DS(x)
        x = self.relu(x)
        x = self.upsample(x)

        cutoff = self.relu2(self.cutoff)
        cutoff = torch.sigmoid(cutoff)
        x = design_filter(x, cutoff=cutoff, pad_mode='replicate', numtabs=5)

        x = self.CST(x)
        x = self.relu(x)

        return x