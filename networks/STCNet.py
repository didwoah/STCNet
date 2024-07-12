import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
import math

NL = 200
SL = 25

CFG = {
    'nina1': (10, 20, 96, 52, 27), 
    'nina2_sampled': (12, 20, 96, 49, 40),
    'nina4_sampled': (12, 20, 96, 52, 10),
    }

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
    
class ConvLayer(nn.Module):
    def __init__(self, in_ch, in_ts, kaiming_init = False):
        super(ConvLayer, self).__init__()
        self.temporal_conv = TemporalConv(in_ch, in_ts)
        self.spatial_conv = SpatialConv(in_ch, in_ts)

    def forward(self, x):
        out1 = self.temporal_conv(x)
        out2 = self.spatial_conv(x)
        out = torch.cat((out1, out2), dim=1)
        return out
    
class TemporalConv(nn.Module):
    def __init__(self, in_ch, in_ts, kaiming_init = False):
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
        out = self.fc(x_)
        return out
    
class SpatialConv(nn.Module):
    def __init__(self, in_ch, in_ts, kaiming_init = False):
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

class Encoder(nn.Module):
    def __init__(self, in_ch, in_ts, out_channels, transformer, kaiming_init = False):
        super(Encoder, self).__init__()
        self.stconv = TimeDistributed(ConvLayer(in_ch, in_ts, kaiming_init = kaiming_init), batch_first = True)
        self.transformer = transformer
        if transformer:
            self.longterm_encoder = TransformerLayer(out_channels)
        else:
            self.longterm_encoder = nn.LSTM(96, NL, num_layers = 2, batch_first = True, dropout = 0.2, bidirectional = True)
        self.flat = nn.Flatten()
    def forward(self, x):
        cout = self.stconv(x)
        if self.transformer:
            out = self.longterm_encoder(cout)
            out = self.flat(out)
        else:
            out, (_, _) = self.longterm_encoder(cout)
            out = self.flat(out)
        return out 

class TransformerLayer(nn.Module):
    def __init__(self, in_features, dim_feedforward=1024, nhead=8, layers=6, dropout=0.3, activation='gelu', max_len=25):
        super().__init__()
        self.positional_encoding = PositionalEncoding(in_features, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_features,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(layers)])

    def forward(self, x):
        x = self.positional_encoding(x)
        for layer in self.transformer_layers:
            x = layer(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=25):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x

class STCNetCE(nn.Module):
    def __init__(self, data = 'nina1', transformer = False, kaiming_init = False):
        super(STCNetCE, self).__init__()
        in_ch, in_ts, out_channels, cdn, _ = CFG[data]
        self.encoder = Encoder(in_ch, in_ts, out_channels, transformer, kaiming_init)
        fc_in = SL * 2 * NL
        if transformer:
            fc_in = 96*25
        self.fc = nn.Sequential(
            nn.Linear(fc_in, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, cdn)
        )
    def forward(self, x):
        out = self.encoder(x)
        out = self.fc(out)
        return out
    
class STCNetSAC(nn.Module):
    def __init__(self, data = 'nina1', transformer = False, head = 'mlp', label_feat_dim = 128, subject_feat_dim = 128, kaiming_init = False):
        super(STCNetSAC, self).__init__()
        in_ch, in_ts, out_channels, _, _ = CFG[data]
        self.encoder = Encoder(in_ch, in_ts, out_channels, transformer, kaiming_init)
        dim_in = SL * 2 * NL
        if transformer:
            dim_in = 96*25
        if head == 'linear':
            self.label_head = nn.Linear(dim_in, label_feat_dim)
            self.subject_head = nn.Linear(dim_in, subject_feat_dim)
        elif head == 'mlp':
            self.label_head = nn.Sequential(
                nn.Linear(dim_in, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, label_feat_dim)
            )
            self.subject_head = nn.Sequential(
                nn.Linear(dim_in, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, subject_feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        
    def forward(self, x):
        out = F.normalize(self.encoder(x), dim=1)
        label_feat = F.normalize(self.label_head(out), dim=1)
        subject_feat = F.normalize(self.subject_head(out), dim=1)
        return label_feat, subject_feat  

if __name__ == '__main__':
    model = STCNetCE(data = 'nina1')
    print(model)