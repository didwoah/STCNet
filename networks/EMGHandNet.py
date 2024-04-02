import torch.nn as nn
import torch.nn.functional as F

NL = 200
SL = 25

CFG = {
    'nina1': (10, 64, 52, 27), 
    'nina2': (12, 832, 49, 40),
    'nina2_sampled': (12, 64, 49, 40),  
    'nina4': (12, 832, 52, 10),
    'nina4_sampled': (12, 64, 52, 10),
    }

class ConvLayer(nn.Module):
    def __init__(self, in_channels, kaiming_init = False):
        super(ConvLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = in_channels, out_channels = 64, kernel_size = 9, stride = 2, padding = 4, padding_mode = 'zeros')
        if kaiming_init:
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.bat1 = nn.BatchNorm1d(64, eps=1e-6, momentum = 0.95)
        self.maxpool1 = nn.MaxPool1d(kernel_size = 8, stride = 2)
        self.actv1 = nn.Tanh()
        self.block1 = nn.Sequential(self.conv1, self.bat1, self.maxpool1, self.actv1)

        self.conv2 = nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 5, stride = 2, padding = 2, padding_mode = 'zeros')
        if kaiming_init:
            nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        self.bat2 = nn.BatchNorm1d(64, eps=1e-6, momentum = 0.95)
        self.actv2 = nn.Tanh()
        self.dropout2 = nn.Dropout(0.2093)
        self.block2 = nn.Sequential(self.conv2, self.bat2, self.actv2, self.dropout2)

        self.conv3 = nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 5, stride = 2, padding = 2, padding_mode = 'zeros')
        if kaiming_init:
            nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        self.bat3 = nn.BatchNorm1d(64, eps=1e-6, momentum = 0.95)
        self.actv3 = nn.Tanh()
        self.dropout3 = nn.Dropout(0.2093)
        self.block3 = nn.Sequential(self.conv3, self.bat3, self.actv3, self.dropout3)

        self.conv4 = nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 2, padding = 1, padding_mode = 'zeros')
        if kaiming_init:
            nn.init.kaiming_normal_(self.conv4.weight, mode='fan_out', nonlinearity='relu')
        self.bat4 = nn.BatchNorm1d(64, eps=1e-6, momentum = 0.95)
        self.actv4 = nn.Tanh()
        self.dropout4 = nn.Dropout(0.2093)
        self.block4 = nn.Sequential(self.conv4, self.bat4, self.actv4, self.dropout4)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):

        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        out = self.flatten(self.relu(out4))

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
    def __init__(self, in_channels, out_channels, kaiming_init = False):
        super(Encoder, self).__init__()
        self.conv = TimeDistributed(ConvLayer(in_channels, kaiming_init = kaiming_init), batch_first = True)
        self.lstm = nn.LSTM(out_channels, NL, num_layers = 2, batch_first = True, dropout = 0.3, bidirectional = True)
        self.flat = nn.Flatten()
    def forward(self, x):
        cout = self.conv(x)
        lout, (_, _) = self.lstm(cout)
        out = self.flat(lout)
        return out
    
class EMGHandNetCE(nn.Module):
    def __init__(self, data = 'nina1', kaiming_init = False):
        in_channels, out_channels, cdn, _ = CFG[data]
        super(EMGHandNetCE, self).__init__()
        self.encoder = Encoder(in_channels, out_channels, kaiming_init)
        self.fc = nn.Sequential(
            nn.Linear(SL * 2 * NL, 512),
            nn.Tanh(),
            nn.BatchNorm1d(512, eps = 1e-5, momentum = 0.9),
            nn.Linear(512, cdn)
        )
    def forward(self, x):
        out = self.encoder(x)
        out = self.fc(out)
        return out
    
class EMGHandNetMultiCE(nn.Module):
    def __init__(self, data = 'nina1', kaiming_init = False):
        in_channels, out_channels, labels, subjects = CFG[data]
        super(EMGHandNetMultiCE, self).__init__()
        self.encoder = Encoder(in_channels, out_channels, kaiming_init)
        self.fc_label = nn.Sequential(
            nn.Linear(SL * 2 * NL, 512),
            nn.Tanh(),
            nn.BatchNorm1d(512, eps = 1e-5, momentum = 0.9),
            nn.Linear(512, labels)
        )
        self.fc_subject = nn.Sequential(
            nn.Linear(SL * 2 * NL, 512),
            nn.Tanh(),
            nn.BatchNorm1d(512, eps = 1e-5, momentum = 0.9),
            nn.Linear(512, subjects)
        )
    def forward(self, x):
        out = self.encoder(x)
        label_out = self.fc_label(out)
        subject_out = self.fc_subject(out)
        return label_out, subject_out
    
class EMGHandNetSupCon(nn.Module):
    def __init__(self, data = 'nina1', head = 'mlp', feat_dim = 128, kaiming_init = False):
        super(EMGHandNetSupCon, self).__init__()
        in_channels, out_channels, _, _ = CFG[data]
        self.encoder = Encoder(in_channels, out_channels, kaiming_init)

        dim_in = SL * 2 * NL

        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        
    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat
    
class EMGHandNetMultiSupCon(nn.Module):
    def __init__(self, data = 'nina1', head = 'mlp', label_feat_dim = 128, subject_feat_dim = 128, kaiming_init = False):
        super(EMGHandNetMultiSupCon, self).__init__()
        in_channels, out_channels, _, _ = CFG[data]
        self.encoder = Encoder(in_channels, out_channels, kaiming_init)
        dim_in = SL * 2 * NL

        if head == 'linear':
            self.label_head = nn.Linear(dim_in, label_feat_dim)
            self.subject_head = nn.Linear(dim_in, subject_feat_dim)
        elif head == 'mlp':
            self.label_head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, label_feat_dim)
            )
            self.subject_head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, subject_feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        
    def forward(self, x):
        
        out = self.encoder(x)
        label_feat = F.normalize(self.label_head(out), dim=1)
        subject_feat = F.normalize(self.subject_head(out), dim=1)

        return label_feat, subject_feat
    
if __name__ == '__main__':
    model = EMGHandNetCE(data = 'nina1')
    print(model)