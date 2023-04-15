import os.path

import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
    def __init__(self, num_class):
        self.num_class = num_class
        super(Net, self).__init__()

        self.level_0_input_encoder = SE_ResBlock(39, 64, 128)

        self.level_1_input_encoder = SE_ResBlock(39, 64, 128)
        self.level_1_encoder = SE_ResBlock(256, 512, 1024)

        self.level_2_input_encoder = SE_ResBlock(39, 64, 128)
        self.level_2_encoder = SE_ResBlock(1152, 1536, 2048)

        self.level_3_encode = SE_ResBlock(2048, 3072, 4096)
        self.level_3_decode = SE_ResBlock(4096, 3072, 2048)

        self.level_2_decode = SE_ResBlock(4096, 2048, 1024)

        self.level_1_decoder = SE_ResBlock(2048, 1024, 512)

        self.level_0_decoder = SE_ResBlock(640, 256, 128)

        self.cat_conv_level_10 = nn.Conv1d(512, 128, kernel_size=1, padding=0)
        self.cat_conv_level_20 = nn.Conv1d(1024, 128, kernel_size=1, padding=0)

        self.decoder_0 = nn.Sequential(
            nn.Conv1d(384, 256, kernel_size=1, padding=0),
            nn.BatchNorm1d(256),
            nn.ELU(),
            nn.Dropout(p=0.5),

            nn.Conv1d(256, 128, kernel_size=1, padding=0),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(p=0.5),

            nn.Conv1d(128, 64, kernel_size=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ELU(),

            nn.Conv1d(64, 32, kernel_size=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ELU(),

        )

        self.decoder_1 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ELU(),

            nn.Linear(16, 8),
            nn.ELU(),

            nn.Linear(8, self.num_class),
        )

    def forward(self, level_0, level_1, level_2, c_1, c_2, c_3, final_mat):
        """
            input dimension (B:batch_size, K0:k_eig_list[0], F:input features dimension, N: the number of vertices):
                level_0: B*K0*F
                level_1: B*K1*F
                level_2: B*K2*F
                c_1: B*K1*K0
                c_2: B*K2*K1
                c_3: B*K0*K2
                final_mat: B*N*K0
        """

        level_0_input = level_0.permute(0, 2, 1)
        encode_level_0 = self.level_0_input_encoder(level_0_input)

        encode_level_0 = encode_level_0.permute(0, 2, 1)
        Transpose_level_01 = c_1 @ encode_level_0

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
        encode_level_2 = encode_level_2.permute(0, 2, 1)

        encode_level_3 = self.level_3_encode(encode_level_2.permute(0, 2, 1))
        decode_level_3 = self.level_3_decode(encode_level_3).permute(0, 2, 1)

        decode_level_2_input = torch.cat((encode_level_2, decode_level_3), dim=2).permute(0, 2, 1)
        decode_level_2 = self.level_2_decode(decode_level_2_input).permute(0, 2, 1)

        Transpose_level_21 = c_2.permute(0, 2, 1) @ decode_level_2

        decode_level_1_input = torch.cat((encode_level_1, Transpose_level_21), dim=2).permute(0, 2, 1)
        decode_level_1 = self.level_1_decoder(decode_level_1_input)
        decode_level_1 = decode_level_1.permute(0, 2, 1)
        Transpose_level_10 = c_1.permute(0, 2, 1) @ decode_level_1

        decode_level_input = torch.cat((encode_level_0, Transpose_level_10), dim=2).permute(0, 2, 1)
        decode_level_0 = self.level_0_decoder(decode_level_input)
        decode_level_0 = decode_level_0.permute(0, 2, 1)

        cat_level_10 = self.cat_conv_level_10(Transpose_level_10.permute(0, 2, 1)).permute(0, 2, 1)
        cat_level_20 = self.cat_conv_level_20((c_3 @ decode_level_2).permute(0, 2, 1)).permute(0, 2, 1)
        cat_level = torch.cat((decode_level_0, cat_level_10, cat_level_20), dim=2).permute(0, 2, 1)

        feat = self.decoder_0(cat_level).permute(0, 2, 1)
        feat = self.decoder_1(feat)

        res = final_mat @ feat

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
