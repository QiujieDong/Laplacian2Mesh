import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F


class BaseBlock(nn.Module):
    def __init__(self, in_channle, med_channle, outchannle):
        super(BaseBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channle, med_channle, kernel_size=3, padding=1),
            nn.BatchNorm1d(med_channle),
            nn.ELU(),
            nn.Conv1d(med_channle, outchannle, kernel_size=3, padding=1),
            nn.BatchNorm1d(outchannle),
        )

    def forward(self, input):
        res = self.block(input)
        return res


class SE_ResBlock(nn.Module):
    def __init__(self, in_channle, med_channle, outchannle, reduce=16):
        super(SE_ResBlock, self).__init__()
        self.resblock = BaseBlock(in_channle, med_channle, outchannle)
        self.se = nn.Sequential(nn.Linear(outchannle, outchannle // reduce),
                                nn.ELU(),
                                nn.Linear(outchannle // reduce, outchannle),
                                nn.Sigmoid())
        self.up = nn.Sequential(
            nn.Conv1d(in_channle, outchannle, kernel_size=1, padding=0),
            nn.BatchNorm1d(outchannle)
        )

    def forward(self, input):
        res = self.resblock(input)

        if isinstance(res.size()[2], int):
            y = nn.AvgPool1d(res.size()[2])(res)
        else:
            y = nn.AvgPool1d(res.size()[2].item())(res)

        y = y.squeeze(2)
        y = self.se(y).unsqueeze(2)
        y = res * y.expand_as(res)
        out = y + self.up(input)
        out = F.elu(out)
        return out


class Net(nn.Module):
    def __init__(self, num_class, last_input_dim):
        super(Net, self).__init__()
        self.num_class = num_class
        self.last_input_dim = last_input_dim

        self.level_0_input_encoder = SE_ResBlock(39, 64, 128)

        self.level_1_input_encoder = SE_ResBlock(39, 64, 128)
        self.level_1_encoder = SE_ResBlock(256, 512, 1024)

        self.level_2_input_encoder = SE_ResBlock(39, 64, 128)
        self.level_2_encoder = SE_ResBlock(1152, 1536, 2048)

        if self.num_class == 40:
            self.conv = nn.Sequential(
                nn.Conv1d(3200, 2048, kernel_size=1, padding=0),
                nn.BatchNorm1d(2048),
                nn.ELU(),

                nn.Conv1d(2048, 1024, kernel_size=1, padding=0),
                nn.BatchNorm1d(1024),
                nn.ELU(),
            )
            self.fc = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ELU(),
                nn.Dropout(p=0.5),
                nn.Linear(512, 128),
                nn.ELU(),
                nn.Dropout(p=0.5),
            )
            avg_out_dim = int((self.last_input_dim - 3) / 3 + 1)
            final_input_dim = 128 * avg_out_dim
        else:
            self.fc = nn.Sequential(
                nn.Linear(3200, 4096),
                nn.Linear(4096, 4096),
            )
            avg_out_dim = int((self.last_input_dim - 3) / 3 + 1)
            final_input_dim = 4096 * avg_out_dim

        self.avg = nn.AvgPool1d(3)

        self.final = nn.Linear(final_input_dim, self.num_class)

    def forward(self, level_0, level_1, level_2, c_1, c_2, c_3):
        level_0_input = level_0.permute(0, 2, 1)
        encode_level_0 = self.level_0_input_encoder(level_0_input)

        encode_level_0 = encode_level_0.permute(0, 2, 1)
        Transpose_level_01 = c_1 @ encode_level_0
        Transpose_level_02 = c_3 @ encode_level_0

        level_1_data = level_1.permute(0, 2, 1)
        level_1_data = self.level_1_input_encoder(level_1_data).permute(0, 2, 1)
        level_1_input = torch.cat((level_1_data, Transpose_level_01), dim=2).permute(0, 2, 1)
        encode_level_1 = self.level_1_encoder(level_1_input)
        encode_level_1 = encode_level_1.permute(0, 2, 1)
        Transpose_level_12 = c_2 @ encode_level_1

        level_2_data = level_2.permute(0, 2, 1)
        level_2_data = self.level_2_input_encoder(level_2_data).permute(0, 2, 1)
        level_2_input = torch.cat((level_2_data, Transpose_level_12), dim=2).permute(0, 2, 1)

        encode_level_2 = self.level_2_encoder(level_2_input)
        encode_level_2 = encode_level_2.permute(0, 2, 1)  # (B, K_min, 2048)

        feat = torch.cat((Transpose_level_02, Transpose_level_12, encode_level_2), dim=-1)

        if self.num_class == 40:
            feat = self.conv(feat.permute(0, 2, 1)).permute(0, 2, 1)
        feat = self.fc(feat).permute(0, 2, 1)
        feat = self.avg(feat).permute(0, 2, 1)
        feat = feat.view(feat.shape[0], -1)

        res = self.final(feat)

        return res


def init_weights(net, init_type='kaiming', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm1d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
