import numpy as np
import gym
from gym import spaces
from static_env import StaticEnv
import torch
from models import MTRNet
from random_process import OrnsteinUhlenbeckProcess
import math
import random
from utils import Paras
import sys
import time

parameters = Paras()
preACTIONS = parameters.pre_actions
torch.manual_seed(parameters.seed)
np.random.seed(parameters.seed)
random.seed(parameters.seed)


def samples(lst, k):
    n = len(lst)
    indices = []
    while len(indices) < k:
        index = random.randrange(n)
        if index not in indices:
            indices.append(index)
    return [lst[i] for i in indices]


class SSEnv(gym.Env, StaticEnv):
    '''
    RL Environment. This class uses gym package.
    This class store the sparse matrix, ground truth matrix and binary selection matrix.
    '''
    def __init__(self, ssnet=None):
        self.ssnet = ssnet
        self.k = 6

        self.cell_amount = 10000
        self.n_actions = self.cell_amount
        self.per_cell = 100
        self.threshold = 4
        self.budget = 0.8 * self.cell_amount
        self.actions_per_time = 100
        self.norm = 8044.071

        self.time = 0
        self.action_space = spaces.Discrete(self.cell_amount)
        self.DEPSILON = 1
        self.EPSILON = 1.0
        self.EPS_END = 0.1
        self.steps = 0
        self.random_process = OrnsteinUhlenbeckProcess(size=1,
                                                       theta=0.15, mu=0, sigma=0.2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.error = -1
        self.predict_node_value = False

    def change_device(self, new_gpu):
        '''
        Process on anther GPU. Used in parallel running.
        '''
        self.device = new_gpu
        self.preload()

    def sample(self):
        '''
        This function control the portion of random actions and the actions output by the policy network.
        Random actions provide exploration. This portion of random actions decreases during training,
        as the model converges.
        '''
        x = max(self.EPSILON, 0) * self.random_process.sample()
        self.EPSILON = self.EPSILON - self.EPSILON / self.DEPSILON
        return torch.from_numpy(x).to(self.device)

    def get_thre(self):
        a = self.EPS_END + (self.EPSILON - self.EPS_END) * math.exp(-1. * math.sqrt(self.epoch) / self.DEPSILON)
        return a

    def getState(self):
        full_matrix = self.get_full_matrix()
        collec_m = torch.mul(self.selec_m, full_matrix)
        return collec_m

    def move_to_next_t(self):
        new_array = torch.zeros((1, self.cell_amount)).to(self.device)
        self.selec_m = torch.cat((self.selec_m.reshape(self.k, self.cell_amount), new_array), dim=0)
        self.selec_m = self.selec_m[1:self.k + 1]
        index = int(self.first_time_actions * self.cell_amount)
        self.selec_m[self.k - 1, torch.randperm(self.cell_amount)[0:index]] = 1
        self.selec_m = self.selec_m.reshape(self.k, self.per_cell, self.per_cell)
        self.time = self.time + 1
        self.state = self.getState()
        self.pre_actions = torch.ones((preACTIONS - 1, 2)).float().to(self.device) * (-1)

    def get_time_info(self):
        '''
        This function returns the time feature used by models.
        The time feature includes the minute of hour, hour of day, and day of week.
        '''
        min = self.time * 10
        if not self.isTrain:
            min += (5760 * 10)
        day_of_wk = int(min / 1440) % 7
        hr_of_day = int(min / 60) % 24
        min_of_hr = int(min) % 60
        res = np.hstack([day_of_wk, hr_of_day, min_of_hr]).reshape(1, -1)
        return res

    def reset(self, isTrain=True):
        self.first_file = True
        self.isTrain = isTrain
        self.isFirst = True
        p = 0.6
        self.selec_m = np.random.choice([0, 1], size=(self.k, self.cell_amount), p=[1 - p, p])
        self.selec_m = torch.from_numpy(self.selec_m).float().to(self.device)

        self.selec_m[self.k - 1, :] = 0
        self.selec_m = self.selec_m.view(self.k, self.cell_amount)
        index = int(self.first_time_actions * self.cell_amount)
        self.selec_m[self.k - 1, torch.randperm(self.cell_amount)[0:index]] = 1
        self.selec_m = self.selec_m.view(self.k, self.per_cell, self.per_cell)  # size (6, 100, 100)
        self.total_cell = torch.sum(self.selec_m[self.k - 1])

        self.pre_actions = torch.ones((preACTIONS-1, 2)).float().to(self.device) * (-1)

        self.time = 0
        self.state = self.getState()
        return self.state, 0, False, None

    def preload(self):
        path = './data/mtrnet.pt'
        with torch.no_grad():
            print('loading network from ' + str(path))
            self.ssnet = self.network = MTRNet().to(self.device)
            self.network.load_state_dict(torch.load(path, map_location=torch.device(self.device)))
            self.network.eval()

    def render(self):
        return 0

    def step(self, action, move=True):
        done = False
        self.selec_m[self.k - 1][action] = 1
        full_matrix = self.get_full_matrix()
        self.state[self.k - 1][action] = full_matrix[self.k - 1][action]
        total_cell = torch.sum(self.selec_m[self.k - 1])

        if total_cell >= self.budget:
            print('Total cells more than budget')
            judge = True
        else:
            judge = self.is_done_state(self.state)

        if judge:
            done = True
            self.isFirst = True

        if done and move:
            self.move_to_next_t()
        return done, total_cell, self.error

    def get_full_matrix(self):
        if not self.first_file:
            return self.gt_matrix[self.time:(self.time + self.k), :]
        else:
            self.first_file = False
            if self.isTrain:
                print('loading training data')
            else:
                print('loading testing data')
            if self.isTrain:
                matrix = np.load('./data/milan_tra.npy')
            else:
                matrix = np.load('./data/milan_val.npy')
            matrix = np.log(1 + matrix) / 3.4086666

            matrix = np.transpose(matrix, (2, 0, 1))
            matrix = torch.from_numpy(matrix).to(self.device)
            gt = matrix[self.time:(self.time + self.k), :, :]
            self.gt_matrix = matrix
            return gt

    def inference_error(self, inferred_m, true_m):
        return torch.abs(inferred_m - true_m).mean()

    def set_t(self, t):
        self.time = t

    def reset_t(self):
        self.time = 0

    def next_state(self, actions):
        length = len(actions[0])
        previous_actions = self.pre_actions.clone()
        actions_x = torch.stack(actions[0]).view(1, -1)
        actions_y = torch.stack(actions[1]).view(1, -1)
        add_action = torch.cat((actions_x, actions_y), dim=0).view(-1, 2)
        previous_actions = torch.cat((previous_actions[length:], add_action), dim=0)

        new_state = self.state.clone()
        full_matrix = self.get_full_matrix()
        new_state[self.k - 1, actions[0], actions[1]] = full_matrix[self.k - 1, actions[0], actions[1]]
        s_selec = self.selec_m[self.k - 1].clone()
        s_selec[actions] = 1
        non_s = torch.where(s_selec == 0)
        return new_state, non_s, s_selec, previous_actions

    def is_done_state(self, state):
        full_matrix = self.get_full_matrix()
        y_gt = full_matrix[self.k - 1]
        last_f = state[state.shape[0] - 1, :, :]
        with torch.no_grad():
                yhat = self.ssnet(state.unsqueeze(0))
        yhat = torch.reshape(yhat, y_gt.shape)
        ind = torch.nonzero(last_f, as_tuple=True)
        yhat[ind] = last_f[ind]
        self.yhat = yhat
        yhat = torch.exp(3.4086666 * yhat) - 1
        y_gt = torch.exp(3.4086666 * y_gt) - 1
        self.error = torch.abs(yhat - y_gt).mean()
        return self.error < self.threshold

    def isDone(self, state):
        full_matrix = self.get_full_matrix()
        y_gt = full_matrix[-1]
        last_f = state[state.shape[0] - 1, :, :]
        with torch.no_grad():
            yhat = self.ssnet(state.unsqueeze(0))
        yhat = torch.reshape(yhat, y_gt.shape)
        ind = torch.nonzero(last_f, as_tuple=True)
        yhat[ind] = last_f[ind]
        yhat = torch.exp(3.4086666 * yhat) - 1
        y_gt = torch.exp(3.4086666 * y_gt) - 1
        error = torch.abs(yhat - y_gt).mean()
        return error < self.threshold

    def initial_state(self):
        self.reset()
        return self.getState()

    @staticmethod
    def get_obs_for_states(states):
        states = torch.stack(states)
        return states.cpu().numpy()
