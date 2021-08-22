import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import random
import math
from conv2d import Conv2d
import time
from toolbox import DataProvider
import sys
from models import MTRNet

isCuda = torch.cuda.is_available()
CYCLE = 6


def make_train_step(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def train_steps(x, y, t):
        model.train()
        optimizer.zero_grad()
        yhat = model.step(x, t).float()
        yhat = torch.reshape(yhat, y.shape)
        loss = loss_fn(y, yhat)
        loss.backward()
        optimizer.step()
        return loss.item()
    return train_steps

def load_dataset():
    tra_set = np.load('./data/milan_tra.npy')
    tra_set = np.log(1 + tra_set)
    a_mean = tra_set.mean()
    tra_set = tra_set / a_mean

    val_set = np.load('data/milan_test.npy')
    val_set = np.log(1 + val_set)
    a_mean = val_set.mean()
    val_set = val_set / a_mean
    val_sparse_set = np.load('./data/milan_val_sparse.npy')
    print('training set:', tra_set.shape)
    print('validation set:', val_set.shape)
    return tra_set, val_set, val_sparse_set


def inference(path, rate):
    test_provider = DataProvider.MoverProvider(length=observations)

    test_set = np.load('data/milan_test.npy')
    test_set = np.log(1 + test_set)
    test_set = test_set / 3.4086666
    print('test set:', test_set.shape)

    print('Sampling rate: ' + str(rate))
    predicted_array = []
    test_kwag = {
        'inputs': test_set,
        'rate': rate,
        'special': False,
        'keepdims': True}
    model_inference = zipNet().float().to(device)
    model_inference.load_state_dict(torch.load(path, map_location=torch.device(device)))
    model_inference.eval()
    mae_errors = []
    t1 = time.time()
    for batch in test_provider.feed(**test_kwag):
        x_out, y_gt = batch
        last_f = x_out[0, x_out.shape[1] - 1, :, :].flatten()

        yhat = model_inference(torch.from_numpy(x_out).float().to(device))
        predicted = torch.reshape(yhat, y_gt.shape).cpu().detach().numpy()[0]
        predicted[np.nonzero(last_f)] = last_f[np.nonzero(last_f)]
        y_gt = np.exp(3.4086666 * y_gt) - 1
        predicted = np.exp(3.4086666 * predicted) - 1
        error_mae = np.mean(abs(y_gt - predicted)).mean()
        mae_errors.append(error_mae)
        predicted_array.append(predicted)

    t2 = time.time()
    print('spent time per snapshot='+str((t2-t1)/1440))
    mae = np.mean(mae_errors) # * max_value
    mae_std = np.std(mae_errors)
    print('MAE error: ' + str(mae))
    print('MAE std: ' + str(mae_std))
    np.save('./models/ssnet/predicted.npy', predicted_array)
    return mae


def train():
    for epoch in range(n_epochs):
        losses = []
        val_losses = []
        test_losses = []
        a1 = time.time()

        for batch in tra_provider.feed(**tra_kwag):
            X_train_a, y_train_a, time_info = batch
            X_train_a = torch.from_numpy(X_train_a).float().to(device)
            y_train_a = torch.from_numpy(y_train_a).float().to(device)
            time_info = torch.from_numpy(time_info).float().to(device)

            loss = train_step(X_train_a, y_train_a, time_info)
            losses.append(loss)
        a2 = time.time()
        print('Epoch: ' + str(epoch) + ', train loss: ' + str(np.mean(losses)) + ', time: ' + str(a2 - a1))
        if epoch % 20 == 0 and epoch > 20:
            torch.save(model.state_dict(), './models/ssnet/model_' + str(epoch) + '.pt')

        with torch.no_grad():
            for batch in val_provider.feed(**val_kwag):
                X_train_a, y_train_a, time_info = batch
                x_val = torch.from_numpy(X_train_a).float().to(device)
                y_val = torch.from_numpy(y_train_a).float().to(device)
                time_info = torch.from_numpy(time_info).float().to(device)
                model.eval()
                yhat = model.step(x_val, time_info)
                yhat = torch.reshape(yhat, y_val.shape)
                yhat = torch.exp(3.4086666 * yhat) - 1
                y_val = torch.exp(3.4086666 * y_val) - 1
                val_loss = loss_fn(y_val, yhat)
                val_losses.append(val_loss.item())
        print('Epoch: ' + str(epoch) + ', validation loss: ' + str(np.mean(val_losses)))
        print('-' * 50)


if __name__ == '__main__':
    #filename = '35'
    Train = True

    tra_set, val_set, val_sparse_set = load_dataset()

    tra_kwag = {
        'inputs': tra_set,
        'sparse_inputs': None,
        'isTrain': True}

    val_kwag = {
        'inputs': val_set,
        'sparse_inputs': val_sparse_set,
        'isTrain': False}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MTRNet().to(device)

    lr = 1e-3
    n_epochs = 500
    observations = 6

    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_step = make_train_step(model, loss_fn, optimizer)
    losses = []
    val_losses = []

    input_size = (100, 100, observations)
    output_size = (100, 100)
    batchsize = 128

    tra_provider = DataProvider.SuperResolutionProvider(stride=(1, 1), input_size=input_size,
                                                        output_size=output_size, batchsize=batchsize, shuffle=True)
    val_provider = DataProvider.SuperResolutionProvider(stride=(1, 1), input_size=input_size,
                                                        output_size=output_size, batchsize=batchsize, shuffle=True)
    train()



