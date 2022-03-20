import torch
from utils import Paras
import random
import numpy as np
import time
import timeit
from utils import Paras

parameters = Paras()
CELL_SIZE = parameters.cell_size
CYCLE = parameters.cycle
KNN = parameters.KNN
softsign = parameters.softsign_scale
PER = parameters.per
action_size = parameters.action_size


def get_knn_action(real_action, non_selected, highestQ=True, KNN=100, state=None, model=None):
    if not highestQ:
        dist = (non_selected[0] - real_action[0]) ** 2 + (non_selected[1] - real_action[1]) ** 2
        ind = dist.argmin()
        indexs = [non_selected[0][ind], non_selected[1][ind]]
        return indexs
    else:
        dist = (non_selected[0] - real_action[0][0]) ** 2 + (non_selected[1] - real_action[0][1]) ** 2
        indexs = torch.topk(dist, k=KNN, largest=False)[1].flatten()
        indexs = [non_selected[0][indexs], non_selected[1][indexs]]
        state_input = state.repeat(len(indexs[0]), 1, 1, 1)
        with torch.no_grad():
            a = model.process(state_input, torch.stack(indexs).T.float())
        action = [indexs[0][a.argmax()], indexs[1][a.argmax()]]
        return action


def test(p_network, v_network, test_env, gpuid):
    snapshot = 1008
    print('Testing the agent...')
    p_network.eval()
    v_network.eval()
    cells = torch.zeros(snapshot)
    test_env.reset(True)
    t = 0
    ps = []

    def choose_action():
        state = test_env.state
        selec = test_env.selec_m[CYCLE - 1]
        time_info = test_env.get_time_info()
        time_info = torch.from_numpy(time_info).float().to(gpuid)

        with torch.no_grad():
            p = p_network.process(state[CYCLE - 1].unsqueeze(0),
                                  test_env.pre_actions.unsqueeze(0).flatten(start_dim=1, end_dim=2), time_info)
        if len(ps) < 20:
            ps.append(p)
        non_s = torch.where(selec == 0)
        action = get_knn_action(p[0], non_s, highestQ=False)
        del p, non_s
        return action

    while t < snapshot:
        # running_times = 0
        done, total_cell, _ = test_env.step(choose_action())
        t = test_env.time
        if done:
            time_info = test_env.get_time_info()
            time_info = torch.from_numpy(time_info).float().to(gpuid)
            if t < 5:
                print('predicted actions by policy network:')
                print(ps)
            ps = []
            t = test_env.time
            cells[t - 1] = total_cell

    mean_cell = cells.mean()
    print('validation Mean=' + str(mean_cell))
