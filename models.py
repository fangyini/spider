import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module
import torch.utils.data

import math
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules.utils import _single, _pair, _triple

from utils import Paras


parameters = Paras()
CELL_SIZE = parameters.cell_size
CYCLE = parameters.cycle


class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation, transposed, output_padding, groups, bias, weight=None):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(weight)

    def reset_parameters(self, weight):
        if weight == None:
            n = self.in_channels
            for k in self.kernel_size:
                n *= k
            stdv = 1. / math.sqrt(n)
            self.weight.data.uniform_(-stdv, stdv)
        else:
            self.weight.data = torch.FloatTensor(weight)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


def conv2d_same_padding(input, weight, bias=None, stride=1, padding='VALID', dilation=1, groups=1):
    if padding == 'SAME':
        padding = 0

        input_rows = input.size(2)
        filter_rows = weight.size(2)
        out_rows = (input_rows + stride[0] - 1) // stride[0]
        padding_rows = max(0, (out_rows - 1) * stride[0] +
                           (filter_rows - 1) * dilation[0] + 1 - input_rows)
        rows_odd = padding_rows % 2

        input_cols = input.size(3)
        filter_cols = weight.size(3)
        out_cols = (input_cols + stride[1] - 1) // stride[1]
        padding_cols = max(0, (out_cols - 1) * stride[1] +
                           (filter_cols - 1) * dilation[1] + 1 - input_cols)
        cols_odd = padding_cols % 2

        input = pad(input, [padding_cols // 2, padding_cols // 2 + int(cols_odd),
                            padding_rows // 2, padding_rows // 2 + int(rows_odd)])

    elif padding == 'VALID':
        padding = 0

    elif type(padding) != int:
        raise ValueError('Padding should be SAME, VALID or specific integer, but not {}.'.format(padding))

    return F.conv2d(input, weight, bias, stride, padding=padding,
                    dilation=dilation, groups=groups)


'''
Custom Conv2D:
PyTorch Conv 2D does not have 'SAME' padding option.
'''
class Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=[1, 1],
                 padding='VALID', dilation=[1, 1], groups=1, bias=True, weight=None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        #padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, weight)

    def forward(self, input, device='cpu'):
        return conv2d_same_padding(input, self.weight.to(device), self.bias.to(device), self.stride,
                                   self.padding, self.dilation, self.groups)


class MTRNet(nn.Module):
    def __init__(self, cell_size=1, observations=1, filter_size=4, feature_map=32, skip=True):
        super(MTRNet, self).__init__()  # input: (?, 10, 10, 4), ?, 4, 10, 10
        self.per = int(np.sqrt(cell_size))
        self.conv_ad = Conv2d(observations, feature_map, filter_size, stride=1, padding='SAME')
        self.conv3d = nn.Conv3d(observations, 16, kernel_size=(3, filter_size, filter_size), stride=1,
                                padding=0)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

        self.bn = nn.BatchNorm2d(feature_map)
        self.conv = Conv2d(feature_map, feature_map, filter_size, stride=1, padding='SAME')

        self.conv_f1 = nn.Sequential(
            Conv2d(feature_map, feature_map*2, filter_size, stride=1, padding='SAME'),
            self.activation_func('leaky_relu')
        )
        self.conv_f2 = nn.Sequential(
            Conv2d(feature_map*2, feature_map*3, filter_size, stride=1, padding='SAME'),
            self.activation_func('leaky_relu')
        )
        self.conv_f3 = nn.Sequential(
            Conv2d(feature_map*3, 1, filter_size, stride=1, padding='SAME'),
            self.activation_func('none')
        )
        self.flatten = nn.Flatten()

        self.zip1 = nn.Sequential(
            Conv2d(feature_map, feature_map, filter_size, stride=1, padding='SAME'),
            self.activation_func('leaky_relu')
        )

        self.zip2 = nn.Sequential(
            Conv2d(feature_map, feature_map, filter_size, stride=1, padding='SAME'),
            self.activation_func('leaky_relu')
        )

        self.zip3 = nn.Sequential(
            Conv2d(feature_map, feature_map, filter_size, stride=1, padding='SAME'),
            self.activation_func('leaky_relu')
        )

        self.zip4 = nn.Sequential(
            Conv2d(feature_map, feature_map, filter_size, stride=1, padding='SAME'),
            self.activation_func('leaky_relu')
        )

        self.zip5 = nn.Sequential(
            Conv2d(feature_map, feature_map, filter_size, stride=1, padding='SAME'),
            self.activation_func('leaky_relu')
        )
        self.skip = skip

    def activation_func(self, activation):
        return nn.ModuleDict([
            ['relu', nn.ReLU(inplace=True)],
            ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
            ['selu', nn.SELU(inplace=True)],
            ['none', nn.Identity()]
        ])[activation]

    def step(self, x, t=None):
        return self.forward(x)

    def change_para(self, device):
        self.to(device)

    def forward(self, x):
        batch = x.size()[0]
        x = x.reshape(batch, 1, 6, 100, 100)
        x = F.pad(x, pad=(1, 2, 1, 2))
        x = self.conv3d(x)
        x0 = F.leaky_relu(x)
        x0 = self.maxpool(x0)
        x0 = x0.reshape(batch, -1, 100, 100)

        b1 = self.zip1(x0)
        b2 = self.zip2(b1)
        b3 = self.zip3(b2)
        x = b1 + b3

        b4 = self.zip4(x)
        x = b2 + b4
        b5 = self.zip5(x)
        x = b5 + x0

        x = self.conv_f1(x)
        x = self.conv_f2(x)
        x = self.conv_f3(x)
        x = self.flatten(x)
        return x


class selection_prediction(nn.Module):
    def __init__(self, cell_size=1, observations=1, filter_size=4, feature_map=16):
        super(selection_prediction, self).__init__()
        self.per = int(np.sqrt(cell_size))
        self.conv_ad = Conv2d(observations, feature_map, filter_size, stride=1, padding='SAME')
        self.conv3d = nn.Conv3d(observations, 16, kernel_size=(3, filter_size, filter_size), stride=1,
                                padding=0)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

        self.bn = nn.BatchNorm2d(feature_map)
        self.conv = Conv2d(feature_map, feature_map, filter_size, stride=1, padding='SAME')

        self.conv_f1 = nn.Sequential(
            Conv2d(feature_map, feature_map * 2, filter_size, stride=1, padding='SAME'),
            self.activation_func('leaky_relu'),
        )
        self.conv_f2 = nn.Sequential(
            Conv2d(feature_map * 2, feature_map * 3, filter_size, stride=1, padding='SAME'),
            self.activation_func('leaky_relu'),
        )
        self.conv_f3 = nn.Sequential(
            Conv2d(feature_map * 3, 1, filter_size, stride=1, padding='SAME'),
            self.activation_func('leaky_relu')
        )
        self.flatten = nn.Flatten()

        self.zip1 = nn.Sequential(
            Conv2d(48, feature_map, filter_size, stride=1, padding='SAME'),
            self.activation_func('leaky_relu'),
            nn.BatchNorm2d(feature_map)
        )

        self.zip2 = nn.Sequential(
            Conv2d(feature_map, feature_map, filter_size, stride=1, padding='SAME'),
            self.activation_func('leaky_relu'),
            nn.BatchNorm2d(feature_map)
        )

        self.zip3 = nn.Sequential(
            Conv2d(feature_map, feature_map, filter_size, stride=1, padding='SAME'),
            self.activation_func('leaky_relu'),
            nn.BatchNorm2d(feature_map)
        )

        self.zip4 = nn.Sequential(
            Conv2d(feature_map, feature_map, filter_size, stride=1, padding='SAME'),
            self.activation_func('leaky_relu'),
            nn.BatchNorm2d(feature_map)
        )

        self.zip5 = nn.Sequential(
            Conv2d(feature_map, feature_map, filter_size, stride=1, padding='SAME'),
            self.activation_func('leaky_relu'),
            nn.BatchNorm2d(feature_map)
        )
        self.fc1 = nn.Linear(10015, 10000)
        self.fc2 = nn.Linear(10000, 10000)
        self.fc3 = nn.Linear(10000, 10000)
        self.sigmoid = nn.Sigmoid()

    def activation_func(self, activation):
        return nn.ModuleDict([
            ['relu', nn.ReLU(inplace=True)],
            ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
            ['selu', nn.SELU(inplace=True)],
            ['none', nn.Identity()]
        ])[activation]

    def step(self, x):
        t = x[:, -1]
        t = [t - a for a in range(5)]
        t = torch.stack(t).transpose(1, 0)
        minute = t * 10
        day_of_wk = ((minute / 1440).int() % 7) / 6 * 2 - 1
        hr_of_day = ((minute / 60).int() % 24) / 23 * 2 - 1
        min_of_hr = ((minute).int() % 60) / 50 * 2 - 1

        x = x[:, :50000]
        time_info = torch.cat((day_of_wk, hr_of_day, min_of_hr), dim=1)
        return self.forward(x, time_info.float())

    def forward(self, x, a):
        batch = x.size()[0]
        x = x.reshape(batch, 1, 5, 100, 100)
        x = F.pad(x, pad=(1, 2, 1, 2))
        x = self.conv3d(x)
        x0 = F.leaky_relu(x)
        x0 = x0.reshape(batch, -1, 100, 100)

        x0 = self.zip1(x0)
        b2 = self.zip2(x0)
        b3 = self.zip3(b2)
        x = x0 + b3

        b4 = self.zip4(x)
        x = b2 + b4
        b5 = self.zip5(x)
        x = b5 + x0

        x = self.conv_f1(x)
        x = self.conv_f2(x)
        x = self.conv_f3(x)
        x = self.flatten(x)
        x = torch.cat((x, a), dim=1)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        x = x.view(-1, 100, 100)
        return x


class zipNet(nn.Module):
    def __init__(self, cell_size=1, observations=1, filter_size=4, feature_map=32, skip=True):
        super(zipNet, self).__init__()  # input: (?, 10, 10, 4), ?, 4, 10, 10
        self.per = int(np.sqrt(cell_size))
        self.conv_ad = Conv2d(observations, feature_map, filter_size, stride=1, padding='SAME')
        self.conv3d = nn.Conv3d(observations, 16, kernel_size=(3, filter_size, filter_size), stride=1,
                                padding=0)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

        self.bn = nn.BatchNorm2d(feature_map)
        self.conv = Conv2d(feature_map, feature_map, filter_size, stride=1, padding='SAME')

        self.conv_f1 = nn.Sequential(
            Conv2d(feature_map, feature_map*2, filter_size, stride=1, padding='SAME'),
            self.activation_func('leaky_relu')
        )
        self.conv_f2 = nn.Sequential(
            Conv2d(feature_map*2, feature_map*3, filter_size, stride=1, padding='SAME'),
            self.activation_func('leaky_relu')
        )
        self.conv_f3 = nn.Sequential(
            Conv2d(feature_map*3, 1, filter_size, stride=1, padding='SAME'),
            self.activation_func('none')
        )
        self.flatten = nn.Flatten()

        self.zip1 = nn.Sequential(
            Conv2d(feature_map, feature_map, filter_size, stride=1, padding='SAME'),
            self.activation_func('leaky_relu')
        )

        self.zip2 = nn.Sequential(
            Conv2d(feature_map, feature_map, filter_size, stride=1, padding='SAME'),
            self.activation_func('leaky_relu')
        )

        self.zip3 = nn.Sequential(
            Conv2d(feature_map, feature_map, filter_size, stride=1, padding='SAME'),
            self.activation_func('leaky_relu')
        )

        self.zip4 = nn.Sequential(
            Conv2d(feature_map, feature_map, filter_size, stride=1, padding='SAME'),
            self.activation_func('leaky_relu')
        )

        self.zip5 = nn.Sequential(
            Conv2d(feature_map, feature_map, filter_size, stride=1, padding='SAME'),
            self.activation_func('leaky_relu')
        )
        self.skip = skip

    def activation_func(self, activation):
        return nn.ModuleDict([
            ['relu', nn.ReLU(inplace=True)],
            ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
            ['selu', nn.SELU(inplace=True)],
            ['none', nn.Identity()]
        ])[activation]

    def step(self, x, t=None):
        return self.forward(x)

    def change_para(self, device):
        self.to(device)

    def forward(self, x):
        batch = x.size()[0]
        x = x.reshape(batch, 1, 6, 100, 100)
        x = F.pad(x, pad=(1, 2, 1, 2))
        x = self.conv3d(x)
        x0 = F.leaky_relu(x)
        x0 = self.maxpool(x0)
        x0 = x0.reshape(batch, -1, 100, 100)

        b1 = self.zip1(x0)
        b2 = self.zip2(b1)
        b3 = self.zip3(b2)
        x = b1 + b3

        b4 = self.zip4(x)
        x = b2 + b4
        b5 = self.zip5(x)
        x = b5 + x0

        x = self.conv_f1(x)
        x = self.conv_f2(x)
        x = self.conv_f3(x)
        x = self.flatten(x)
        return x
