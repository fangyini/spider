import torch
from utils import Paras
import random
import numpy as np
import time
import timeit

parameters = Paras()
SUBSET_SIZE = parameters.KNN
PER = parameters.per
CYCLE = 6
preACTIONS = parameters.pre_actions
skip_action = parameters.skip_action
action_size = parameters.action_size


def get_subset(p_network, train_env, gpuid, selection, time_info):
    non_s = torch.where(selection == 0)
    n_actions = SUBSET_SIZE

    perc = train_env.get_thre()
    n_random = int(n_actions * perc)
    with torch.no_grad():
        proto = p_network.process(train_env.state[CYCLE-1].unsqueeze(0),
                                  train_env.pre_actions.unsqueeze(0).flatten(start_dim=1, end_dim=2), time_info)
    action1 = proto[0][0]
    action2 = proto[0][1]
    dist = (non_s[0] - action1) ** 2 + (non_s[1] - action2) ** 2
    ind1 = torch.topk(dist, k=(n_actions - n_random), largest=False)[1].flatten()
    final_ind_tuple = [non_s[0][ind1], non_s[1][ind1]]
    final_ind = torch.stack(final_ind_tuple, dim=0).T

    non_s_mask = torch.zeros(PER, PER).to(gpuid)
    non_s_mask[non_s] = 1
    non_s_mask[final_ind_tuple] = 0
    unique = torch.where(non_s_mask == 1)
    indice = random.sample(range(unique[0].size()[0]), n_random)
    random_ind = [unique[0][indice], unique[1][indice]]
    random_ind = torch.stack(random_ind, dim=0).T
    return torch.cat((final_ind, random_ind), dim=0)


def execute_episode(p_network, train_env, gpuid):
    p_network.eval()
    #errors_data = []
    obs_data = []
    actions_data = []
    time_info = train_env.get_time_info()
    time_info = torch.from_numpy(time_info).float().to(gpuid)
    while True:
        selection = train_env.selec_m[CYCLE - 1]
        subset = get_subset(p_network, train_env, gpuid, selection, time_info)
        sbatch_state = train_env.state.unsqueeze(0).repeat(SUBSET_SIZE, 1, 1, 1)
        x = subset[:, 0]
        y = subset[:, 1]
        full_matrix = train_env.get_full_matrix()[CYCLE - 1]
        sbatch_state[torch.arange(SUBSET_SIZE), CYCLE - 1, x, y] = full_matrix[x, y]
        with torch.no_grad():
            yhat = train_env.network(sbatch_state)
        batch_full_matrix = full_matrix.unsqueeze(0).repeat(SUBSET_SIZE, 1, 1).view(yhat.size())
        errors = torch.abs(yhat - batch_full_matrix).mean(dim=1)
        _, best_ind = torch.topk(errors, skip_action, largest=False, sorted=True)
        best_actions = subset[best_ind]
        obs_data.append(train_env.state[CYCLE-1])
        actions_data.append(torch.cat((train_env.pre_actions, best_actions[0].view(-1, 2).float()), dim=0))
        done, _, _ = train_env.step([best_actions[:, 0], best_actions[:, 1]])
        if done:
            break
    obs = torch.stack(obs_data)
    act = torch.stack(actions_data).flatten(start_dim=1, end_dim=2)
    return obs, act, time_info.repeat(len(obs), 1)

