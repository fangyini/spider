import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils import Paras

parameters = Paras()
CELL_SIZE = parameters.cell_size
CYCLE = parameters.cycle
softsign = parameters.softsign_scale
PER = parameters.per
preAction = parameters.pre_actions


class Trainer:
    '''
    Trainer provides the training function for value network and policy network.
    Currently we do not use training function of value network, because we use a trained MTRNet.
    '''
    def __init__(self, Policy, Value, device):
        self.p_model = Policy()
        self.v_model = Value()

        self.value_criterion = nn.MSELoss()
        self.p_optimizer = torch.optim.Adam(self.p_model.parameters(), lr=1e-4)
        self.v_optimizer = torch.optim.Adam(self.v_model.parameters(), lr=1e-5)
        self.device = device

    def load_ckp(self, p_model_path, v_model_path, p_optimizer_path, v_optimizer_path):
        if p_model_path is not None:
            self.p_model.load_state_dict(torch.load(p_model_path, map_location=torch.device('cpu')))
            print('loading p model path')
        if v_model_path is not None:
            self.v_model.load_state_dict(torch.load(v_model_path, map_location=torch.device('cpu')))
            print('loading v model path')
        if p_optimizer_path is not None:
            self.p_optimizer.load_state_dict(torch.load(p_optimizer_path, map_location=torch.device('cpu')))
        if v_optimizer_path is not None:
            self.v_optimizer.load_state_dict(torch.load(v_optimizer_path, map_location=torch.device('cpu')))

    def train(self, obs, act, time, returns, epoch=None):

        self.p_optimizer.zero_grad()
        p = self.p_model.process(obs[:, :50000].view(-1, 5, 100, 100),
                                 obs[:, 50000:].view(-1, 100, 100).unsqueeze(1),
                                 time)
        policy_loss = self.value_criterion(p, act)
        policy_loss.backward()
        self.p_optimizer.step()
        return 0, policy_loss.item()

    def train_policy(self, obs, act, time, epoch=None):
        self.p_optimizer.zero_grad()
        p = self.p_model.process(obs.view(-1, 100, 100), act[:, :-2], time)
        policy_loss = self.value_criterion(p, act[:, -2:])
        policy_loss.backward()
        self.p_optimizer.step()
        return policy_loss.item()

    def train_value(self, obs, gt, epoch=None):
        self.v_optimizer.zero_grad()
        value = self.v_model(obs)
        value_loss = self.value_criterion(value, gt.view(value.size()))
        value_loss.backward()
        self.v_optimizer.step()
        return value_loss.item()
