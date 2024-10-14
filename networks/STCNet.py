import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
import math
from networks.utils import STCBlock, TimeDistributed

NL = 200
SL = 25

CFG = {
    'nina1': (10, 20, 96, 52, 27), 
    'nina2': (12, 20, 96, 49, 40),
    'nina4': (12, 20, 96, 52, 10),
    }

class Encoder(nn.Module):
    def __init__(self, in_ch, in_ts, out_channels, transformer):
        super(Encoder, self).__init__()
        self.stconv = TimeDistributed(STCBlock(in_ch, in_ts), batch_first = True)
        self.transformer = transformer
        if transformer:
            self.longterm_encoder = TransformerLayer(out_channels)
        else:
            self.longterm_encoder = nn.LSTM(out_channels, NL, num_layers = 2, batch_first = True, dropout = 0.2, bidirectional = True)
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
    def __init__(self, data = 'nina1', transformer = False):
        super(STCNetCE, self).__init__()
        in_ch, in_ts, out_channels, cdn, _ = CFG[data]
        self.encoder = Encoder(in_ch, in_ts, out_channels, transformer)
        fc_in = SL * 2 * NL
        if transformer:
            fc_in = out_channels*25
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
    def __init__(self, dataset = 'nina1', transformer = False, head = 'mlp', label_feat_dim = 128, subject_feat_dim = 128):
        super(STCNetSAC, self).__init__()
        in_ch, in_ts, out_channels, _, _ = CFG[dataset]
        self.encoder = Encoder(in_ch, in_ts, out_channels, transformer)
        dim_in = SL * 2 * NL
        if transformer:
            dim_in = out_channels*25
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
    model = STCNetSAC(data = 'nina1')
    print(model)