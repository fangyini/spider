import numpy as np
from trainer import Trainer
from replay_memory import ReplayMemory
from ss_env import SSEnv
from ss_policy import PolicyNetwork, ValueNetwork
import torch
import torch.multiprocessing as mp
import time
import sys
from utils import Paras
import random
from sampling import CellSelection
from models import MTRNet
from test import test
from gpu_info import device_no

parameters = Paras()
CELL_SIZE = parameters.cell_size
CYCLE = parameters.cycle
KNN = parameters.KNN
softsign = parameters.softsign_scale
PER = parameters.per
predict_q_value_mode = parameters.predict_q_value_mode
num_parallel = parameters.num_parallel
preACTIONS = parameters.pre_actions
print('num of parallel: ' + str(num_parallel))
r_seed = parameters.seed
torch.manual_seed(r_seed)
np.random.seed(r_seed)
print('Softsign scale='+str(softsign))
print('random seed='+str(r_seed))
skip_action = parameters.skip_action
print('skip action='+str(skip_action))
action_size = parameters.action_size


def train(trainer, t_sta, t_end, gpuid=None):
    p_network = trainer.p_model
    if gpuid is None:
        gpuid = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_network.change_para(gpuid)
    v_network.change_para(gpuid)
    training_network_time = 0
    i_time = 144  # * skip_action
    loop = 300
    mem = ReplayMemory(2000,
                       {"ob": torch.float32,
                        "act": torch.float32,
                        "time": torch.float32},
                       {"ob": [1*CELL_SIZE],
                        "act": [preACTIONS*2],
                        "time": [3]}, batch_size=128, gpuid=gpuid)

    knn = KNN
    print('knn=' + str(knn))
    isTrain = False
    isTest = False

    train_env = SSEnv(v_network)
    train_env.first_time_actions = 0.1
    train_env.change_device(gpuid)
    train_env.reset(isTrain)
    test_env = SSEnv(v_network)
    test_env.first_time_actions = train_env.first_time_actions
    test_env.change_device(gpuid)
    test_env.reset(isTest)
    test_env.predict_node_value = predict_q_value_mode
    value_losses = []
    policy_losses = []
    cells_array = []
    i = 0
    epoch = 20
    p_network.device = gpuid
    v_network.device = gpuid
    trainer.device =gpuid
    parameters.device = gpuid
    ext = str(gpuid)

    cell_selection = CellSelection(p_network, v_network, train_env, gpuid)

    #test(p_network, v_network, test_env, gpuid)
    states = []
    selec = []
    predicted = []
    cells = torch.zeros((t_end - t_sta))

    for epoch_i in range(epoch):
        print('Epoch='+str(epoch_i))
        train_env.reset(isTrain)
        train_env.time = t_sta
        t = train_env.time
        t1 = time.time()
        print('Start training...')
        while t < t_end:
            train_env.epoch = training_network_time
            obs, act, time_info = cell_selection.execute_episode()
            mem.add_all({"ob": obs, "act": act, "time": time_info})

            '''train_env.epoch = 999
            states.append(train_env.state[:5])
            cell_selection.execute_episode()
            selec.append(train_env.selec_m[4])
            predicted.append(train_env.yhat)'''

            if i % i_time == 0 and i >= i_time:
                p_network.train()
                v_network.train()
                for x in range(loop):
                    batch = mem.get_minibatch()
                    pl = trainer.train_policy(batch["ob"], batch["act"], batch["time"], x)
                    policy_losses.append(pl)
                print('policy loss=' + str(pl))
                print('*' * 50)
                torch.save(p_network.state_dict(), './models/agentP_'+str(ext)+'_' + str(epoch_i) + '_' + str(t) + '.pt')
                torch.save(trainer.p_optimizer.state_dict(),
                           './models/adam_p_' + str(ext) + '_' + str(epoch_i) + '_' + str(t) + '.pt')
                np.save('./models/policy_loss_' + str(ext) + '_' + str(epoch_i) + '_' + str(t) + '.npy', policy_losses)
                test(p_network, v_network, test_env, gpuid)
                print('snapshot=' + str(t) + ', epoch=' + str(epoch_i) + ', gpu id=' + str(gpuid) + ', is finished')
                sys.stdout.flush()
                training_network_time += 1
                del batch, pl
                t2 = time.time()
                print('time for last 20 episodes=' + str((t2 - t1) / 60) + ' min')
                t1 = time.time()
            i += 1
            t = train_env.time
        '''torch.save(states, 'models/training_with_selection/states_log_'+str(t_sta)+'_'+str(t_end)+'.pt')
        torch.save(selec, 'models/training_with_selection/selec_log_'+str(t_sta)+'_'+str(t_end)+'.pt')
        torch.save(predicted, 'models/training_with_selection/predi_log_' + str(t_sta) + '_' + str(t_end) + '.pt')'''

if __name__ == '__main__':
    torch.set_printoptions(precision=8)
    cell_size = CELL_SIZE
    hidden_dim = 128
    layer_dim = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Cell size='+str(cell_size))
    trainer = Trainer(lambda: PolicyNetwork().to(device),
                         lambda: MTRNet().to(device), device)
    trainer.load_ckp(None, 'data/mtrnet.pt', None, None)
    p_network = trainer.p_model
    v_network = trainer.v_model
    p_network.share_memory()
    v_network.share_memory()

    mp.set_start_method('spawn')
    day = 1
    mul = False
    try:
        num_processes = int(device_no())
        day_par = int(day / num_processes)
    except:
        num_processes = 0
        print('You got 0 GPU!')
    processes = []
    snapshot = 1008

    if mul:
        for rank in range(num_processes):
            dur = int(snapshot / num_processes)
            t_sta = dur * rank
            t_end = t_sta + dur
            print('Process:' + str(rank) + ', t_start=' + str(t_sta)+', t_end='+str(t_end))
            p = mp.Process(target=train, args=(trainer, t_sta, t_end, 'cuda:' + str(rank),))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        train(trainer, 0, int(day*snapshot))
