import numpy as np
from ss_env import SSEnv
import torch
import time
import random
from utils import Paras
import sys
from models import selection_prediction
import torch.nn as nn

parameters = Paras()
CELL_SIZE = parameters.cell_size
CYCLE = parameters.cycle
KNN = parameters.KNN
softsign = parameters.softsign_scale
PER = parameters.per
random.seed(parameters.seed)
add_time = parameters.add_time_info

sig = nn.Sigmoid()
args = sys.argv
cell_size = CELL_SIZE
hidden_dim = 128
layer_dim = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_env = SSEnv()


def evaluate(train, start_t, duration, save_filename, model=None, isprint=False):
    test_env = SSEnv()
    print('start t=' + str(start_t))
    print('duration=' + str(duration))
    print('save_filename=' + str(save_filename))

    print('loading networks...')
    if model is None:
        with torch.no_grad():
            model = selection_prediction().to(device)
            model.load_state_dict(torch.load('models/selection_prediction.pt',
                                             map_location=torch.device('cpu')))
    model.eval()

    print('Testing the agent...')
    cells = torch.zeros(duration).to(device)
    errors = torch.zeros(duration).to(device)
    nmae = torch.zeros(duration).to(device)
    nrmse = torch.zeros(duration).to(device)
    selection_matrix = torch.zeros((duration, 100, 100)).to(device)
    times_array = torch.zeros(duration).to(device)

    test_env.mode = 'gt'
    print('mode=' + str(test_env.mode))
    test_env.reset(train)
    test_env.time = start_t
    test_env.change_device(device)
    t = test_env.time

    a = time.time()
    input2 = test_env.gt_matrix[:5, :, :]
    while t < (start_t + duration):
        with torch.no_grad():
            the_state = test_env.state[:5]
            input1 = torch.cat((the_state, input2))
            input2model = torch.cat((input1.view(1, -1),
                                     torch.tensor(t + 5760).float().view(1, 1).to(device)), dim=1)
            yhat = model.step(input2model).view(100, 100)
            yhat = sig(yhat)
            yhat = (yhat - yhat.min()) / (yhat.max() - yhat.min())
        nonz = torch.where(yhat >= 0.5)
        done, total_cell, _ = test_env.step([nonz[0], nonz[1]], move=False)

        input2 = input2[:-1, :, :]
        yhat = test_env.yhat.unsqueeze(0)
        input2 = torch.cat((input2, yhat), dim=0)

        if isprint:
            print('t=' + str(t) + ', cells=' + str(total_cell) + ', error=' + str(test_env.error))
        test_env.move_to_next_t()
        t += 1
        cells[t - start_t - 1] = total_cell
        errors[t - start_t - 1] = test_env.error
        selection_matrix[t - start_t - 1] = test_env.selec_m[4]
        times_array[t - start_t - 1] = time.time() - a
        a = time.time()
        yh = test_env.yhat_unnorm
        yg = test_env.gt_unnorm
        nmae[t - start_t - 1] = (yh - yg).abs().mean() / yg.abs().mean()
        nrmse[t - start_t - 1] = torch.sqrt(torch.pow(yh.abs() - yg.abs(), 2).mean()) / yg.abs().mean()

    mean_cell = cells.mean()
    print('validation Mean=' + str(mean_cell))
    print('MAE mean=' + str(errors.mean()))
    print('NMAE mean=' + str(nmae.mean()))
    print('NRMSE mean=' + str(nrmse.mean()))

if __name__ == '__main__':
    t1 = time.time()
    evaluate(False, 0, 2874, None, isprint=True)
    t2 = time.time()
    print('total time=' + str(t2 - t1))
