import torch
import torch.nn as nn
import torch.nn.functional as F
from models import Conv2d
import numpy as np
import time
from utils import Paras

device_def = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parameters = Paras()
CELL_SIZE = parameters.cell_size
CYCLE = parameters.cycle
preACTIONS = parameters.pre_actions
torch.manual_seed(parameters.seed)
np.random.seed(parameters.seed)
action_size = parameters.action_size


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class PolicyNetwork(nn.Module):
    def __init__(self, device=device_def):
        super(PolicyNetwork, self).__init__()
        observations = 1
        filter_size = 4
        feature_map = 32
        self.conv_ad = Conv2d(observations, feature_map, filter_size, stride=1, padding='SAME')

        self.conv1 = Conv2d(feature_map, feature_map, filter_size, stride=1, padding='SAME')
        self.conv12 = Conv2d(feature_map, feature_map, filter_size, stride=1, padding='SAME')
        self.conv3 = Conv2d(feature_map, 1, filter_size, stride=1, padding='SAME')
        self.lrelu = nn.ReLU()
        self.fc_test = nn.Linear(16 + (preACTIONS - 1) * 2 + 3, 128)
        self.fc_test2 = nn.Linear(128, 2)
        self.bm = nn.BatchNorm1d(128, affine=False)
        self.device = device

        self.bm1 = nn.BatchNorm2d(feature_map, affine=False)
        self.bm11 = nn.BatchNorm2d(feature_map, affine=False)
        self.bm12 = nn.BatchNorm2d(feature_map, affine=False)
        self.avgpool = nn.AvgPool2d(5, stride=5)
        self.init_weights()
        self.flatten = nn.Flatten()

    def activation_func(self, activation):
        return nn.ModuleDict([
            ['relu', nn.ReLU(inplace=True)],
            ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
            ['selu', nn.SELU(inplace=True)],
            ['none', nn.Identity()]
        ])[activation]

    def change_para(self, device):
        self.to(device)
    def init_weights(self):
        nn.init.xavier_uniform_(self.conv_ad.weight)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.kaiming_uniform_(self.fc_test.weight)

    def process(self, x, act, time=None, isPrint=False):
        act = torch.cat((act.reshape(1, -1), time), dim=1)
        y = self.forward(x.unsqueeze(0), act, isPrint)
        return y

    def forward(self, x, act, isPrint=False):
        x = self.conv_ad(x, self.device)
        x = self.bm1(x)
        x = self.lrelu(x)
        x = self.avgpool(x) * 25

        x = self.conv1(x, self.device)
        x = self.bm11(x)
        x = self.lrelu(x)
        x = self.avgpool(x) * 25

        x = self.conv3(x, self.device)
        x = self.flatten(x)
        x = self.lrelu(x)
        x = torch.cat((x, act), dim=1)
        x = self.fc_test(x)
        x = self.lrelu(x)
        x = self.fc_test2(x)
        return x


class ValueNetwork(nn.Module):
    def __init__(self, device=device_def):
        super(ValueNetwork, self).__init__()
        observations = 1
        filter_size = 4
        feature_map = 32
        self.conv_ad = Conv2d(observations, feature_map, filter_size, stride=1, padding='SAME')
        self.conv1 = Conv2d(feature_map, feature_map, filter_size, stride=1, padding='SAME')
        self.conv2 = Conv2d(feature_map, feature_map * 2, filter_size, stride=1, padding='SAME')
        self.conv3 = Conv2d(feature_map, 1, filter_size, stride=1, padding='SAME')
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.fc_c = nn.Linear(625+2, 256)
        self.bm = nn.BatchNorm1d(625, affine=False)
        self.fc_c1 = nn.Linear(256, 128)
        self.fc_c2 = nn.Linear(128, 1)

        self.fc1 = nn.Linear(625, 20)
        self.fc2 = nn.Linear(22, 1)

        self.device = device
        self.bm1 = nn.BatchNorm2d(feature_map, affine=False)
        self.bm11 = nn.BatchNorm2d(feature_map, affine=False)
        self.bm111 = nn.BatchNorm2d(feature_map, affine=False)
        self.bm_c = nn.BatchNorm1d(256, affine=False)
        self.bm_c1 = nn.BatchNorm1d(128, affine=False)
        self.layer_dim = 3
        self.hidden_dim = 256

        self.conv3d = nn.Conv3d(observations, 16, kernel_size=(3, filter_size, filter_size), stride=1,
                                padding=0)
        self.avgpool2d_3 = nn.AvgPool2d(3, 3)
        self.avgpool2d_2 = nn.AvgPool2d(2, 2)
        self.avgpool2d_4 = nn.AvgPool2d(4, 4)
        self.avgpool = nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv_ad.weight)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc_c.weight)
        nn.init.xavier_uniform_(self.fc_c1.weight)
        nn.init.xavier_uniform_(self.fc_c2.weight)

    def change_para(self, device):
        self.to(device)

    def activation_func(self, activation):
        return nn.ModuleDict([
            ['relu', nn.ReLU(inplace=True)],
            ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
            ['selu', nn.SELU(inplace=True)],
            ['none', nn.Identity()]
        ])[activation]

    def process(self, x, action):
        return self.forward(x, action)

    def forward(self, x, action):
        batch = x.size()[0]
        x = x.reshape(batch, 1, 6, 100, 100)
        x = F.pad(x, pad=(1, 2, 1, 2))
        x = self.conv3d(x)
        x = F.leaky_relu(x)
        x = self.avgpool(x) * 2
        x = x.reshape(batch, -1, 100, 100)

        x = self.conv1(x, self.device)
        x = self.bm111(x)
        x = self.lrelu(x)
        x = self.avgpool2d_4(x) * 16

        x = self.conv3(x, self.device)

        x = x.view(batch, -1)  # 1, 2500
        x = self.bm(x)
        x = self.lrelu(x)

        x = self.fc1(x)
        x = self.lrelu(x)

        x = torch.cat((x, action), dim=1)
        x = self.fc2(x)
        return x