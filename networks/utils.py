import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
import math

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first = False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        # x size: batch, steps, channels, length
        batch, steps, channels, length = x.size()
        x_reshape = x.contiguous().view(batch * steps, channels, length)
        out_ = self.module(x_reshape)
        out = out_.view(batch, steps, -1)
        if self.batch_first is False:
            out = out.permute(1, 0, 2)
        return out
    
class STCBlock(nn.Module):
    def __init__(self, in_ch, in_ts):
        super(STCBlock, self).__init__()
        self.temporal_conv = TemporalConv(in_ch, in_ts)
        self.spatial_conv = SpatialConv(in_ch, in_ts)

    def forward(self, x):
        out1 = self.temporal_conv(x)
        out2 = self.spatial_conv(x)
        out = torch.cat((out1, out2), dim=1)
        return out
    
class TemporalConv(nn.Module):
    def __init__(self, in_ch, in_ts):
        super(TemporalConv, self).__init__()
        self.block1 = ResidualBlock(in_ch, 16, 1)
        self.block2 = ResidualBlock(16, 32, 2)
        self.block3 = ResidualBlock(32, 64, 2)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ts * 16, in_ts * 8),
            nn.ReLU(),
            nn.Linear(in_ts * 8, 64)
        )
    def forward(self, x):
        x_ = x.permute(0,2,1)
        x_ = self.block1(x_)
        x_ = self.block2(x_)
        x_ = self.block3(x_)
        # print(x_.shape)
        out = self.fc(x_)
        return out
    
class SpatialConv(nn.Module):
    def __init__(self, in_ch, in_ts):
        super(SpatialConv, self).__init__()
        self.block1 = ResidualBlock(in_ts, 16, 1)
        self.block2 = ResidualBlock(16, 32, 2)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear((in_ch + 2) * 16, (in_ch + 2) * 8),
            nn.ReLU(),
            nn.Linear((in_ch + 2) * 8, 32)
        )
    def forward(self, x):
        x = torch.concat((x, x[:, :, :2]), dim=2)
        x = self.block1(x)
        x = self.block2(x)
        out = self.fc(x)
        return out
    
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first = False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        # x size: batch, steps, channels, length
        batch, steps, channels, length = x.size()
        x_reshape = x.contiguous().view(batch * steps, channels, length)
        out_ = self.module(x_reshape)
        out = out_.view(batch, steps, -1)
        if self.batch_first is False:
            out = out.permute(1, 0, 2)
        return out
    
def ConvBlock3x3(in_ch, out_ch, stride):
    return nn.Conv1d(in_channels = in_ch, out_channels = out_ch, kernel_size = 3, stride = stride, padding = 1, padding_mode = 'zeros')
    
def ConvBlock1x1(in_ch, out_ch, stride):
    return nn.Conv1d(in_channels = in_ch, out_channels = out_ch, kernel_size = 1, stride = stride, padding = 0)
   
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock3x3(in_ch, out_ch, stride)
        self.bat1 = nn.BatchNorm1d(out_ch)
        self.drop1 = nn.Dropout(0.3)
        self.actv1 = nn.ReLU()
        self.conv2 = ConvBlock3x3(out_ch, out_ch, 1)
        self.bat2 = nn.BatchNorm1d(out_ch)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                ConvBlock1x1(in_ch, out_ch, stride),
                nn.BatchNorm1d(out_ch)
            )
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bat1(x)
        x = self.actv1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.bat2(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x = x + identity
        return x