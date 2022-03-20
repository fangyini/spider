import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import time
import sys
from models import zipNet
from models import selection_prediction2
from test import test_selection_prediction

isCuda = torch.cuda.is_available()
CYCLE = 6
CELL_SIZE = 10000
args = sys.argv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def processData():
    y = []
    size = 4032
    for i in range(0, size, 144):
        try:
            filename = './models/training_with_selection/selec_' + str(i) + '_' + str(i + 144) + '.pt'
            y_i = torch.stack(torch.load(filename, map_location=torch.device('cpu'))).to(device)
        except:
            filename = './models/training_with_selection/selec_log_' + str(i) + '_' + str(i + 144) + '.pt'
            y_i = torch.stack(torch.load(filename, map_location=torch.device('cpu'))).to(device)
        y.append(y_i)
    y = torch.stack(y).view(-1, 100, 100)
    a = y.sum(axis=0).flatten().cpu().numpy() / size
    m = np.mean(a)
    print('mean='+str(m))
    print('min+max/2='+str((np.max(a)+np.min(a))/2))
    print('median='+str(np.median(a)))
    print('mean-me/std='+str((m-np.median(a))/np.std(a)))

    hist, bin_edges = np.histogram(a, bins=np.arange(0, 1, 0.1))
    hist = hist / 10000
    mean = np.mean(hist)
    median = np.median(hist)
    std = np.std(hist)
    print('mean=' + str(mean))
    print('median=' + str(median))
    print('std=' + str(std))
    print('value=' + str((mean - median) / std))
    quit()

def genWhole():
    size = 4032
    network = zipNet().to(device)
    network.load_state_dict(torch.load('./data/ssnet_log.pt', map_location=torch.device(device)))
    network.eval()

    y = []
    for i in range(0, size, 144):
        try:
            filename = './models/training_with_selection/selec_' + str(i) + '_' + str(i + 144) + '.pt'
            y_i = torch.stack(torch.load(filename, map_location=torch.device('cpu'))).to(device)
        except:
            filename = './models/training_with_selection/selec_log_' + str(i) + '_' + str(i + 144) + '.pt'
            y_i = torch.stack(torch.load(filename, map_location=torch.device('cpu'))).to(device)
        y.append(y_i)
    y = torch.stack(y).view(-1, 100, 100)

    #a = y.sum(dim=(1, 2))
    #torch.save(a, 'models/training_with_selection/cells_of_selection_data.pt')

    x = []
    for i in range(0, size, 144):
        try:
            filename = './models/training_with_selection/states_' + str(i) + '_' + str(i + 144) + '.pt'
            x_i = torch.stack(torch.load(filename, map_location=torch.device('cpu'))).to(device)
            print('reading file: ' + str(i) + '_' + str(i + 144))
        except:
            filename = './models/training_with_selection/states_log_' + str(i) + '_' + str(i + 144) + '.pt'
            x_i = torch.stack(torch.load(filename, map_location=torch.device('cpu'))).to(device)
            print('reading file: ' + str(i) + '_' + str(i + 144))
        x_i *= 8044.071
        x_i = torch.log(1 + x_i) / 3.4086666
        x.append(x_i)
    x = torch.stack(x).view(size, 5, 100, 100)
    state_x = x

    pred = []
    for i in range(size-1):
        state = x[i]
        nexts = x[i+1][-1]
        state = torch.cat((state, nexts.unsqueeze(0)), dim=0)
        last_f = state[-1]
        with torch.no_grad():
            yhat = network(state.unsqueeze(0)).view(100, 100)
        ind = torch.nonzero(last_f, as_tuple=True)
        yhat[ind] = last_f[ind]
        pred.append(yhat)

    pred = torch.stack(pred)
    b = np.load('./data/milan_tra.npy')[:, :, :5].transpose(2, 0, 1)  # 5, 100, 100
    b = np.log(1 + b) / 3.4086666
    b = torch.from_numpy(b).to(device)

    b = torch.cat((b, pred), dim=0)
    x = [b[i:(i + 5)] for i in range(size)]
    x = torch.stack(x)

    final_x = torch.cat((state_x, x), dim=1)
    final_x = final_x.view(size, -1)
    x = torch.cat((final_x, torch.arange(size).view(size, 1).float().to(device)), dim=1)

    leave1out = x
    dataset_size = leave1out.size()[0]

    indices = list(range(dataset_size))
    split = int(np.floor(0.9 * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    # Creating PT data samplers and loaders:
    train_sampler = leave1out[train_indices]
    valid_sampler = leave1out[val_indices]

    print('training data size:')
    print(train_sampler.size())
    print('testing data size:')
    print(valid_sampler.size())

    torch.save(train_sampler, 'models/training_with_selection/train_x.pt')
    torch.save(valid_sampler, 'models/training_with_selection/val_x.pt')

    leave1out = y
    train_sampler = leave1out[train_indices]
    valid_sampler = leave1out[val_indices]

    print('training data size:')
    print(train_sampler.size())
    print('testing data size:')
    print(valid_sampler.size())

    torch.save(train_sampler, 'models/training_with_selection/train_y.pt')
    torch.save(valid_sampler, 'models/training_with_selection/val_y.pt')


def make_train_step(model, loss_fn, optimizer):
    def train_steps(x, y):
        model.train()
        optimizer.zero_grad()
        yhat = model.step(x)
        loss = loss_fn(yhat.view(-1, 10000), y.view(-1, 10000))
        loss.backward(retain_graph=True)
        optimizer.step()
        return loss.item(), yhat.detach()
    return train_steps


def train():
    mean_loss = []
    mean_loss_test = []
    for epoch in range(n_epochs):
        losses = []
        a1 = time.time()
        it = 0
        acc = []
        for x_batch, y_batch in train_loader:
            it += 1
            loss, yhat = train_step(x_batch, y_batch)
            losses.append(loss)
            yhat = sig(yhat)
            yhat = torch.where((yhat>=thre), 1, 0)

            accuracy = 1 - abs(yhat-y_batch).sum(dim=(1,2)).mean() / 10000
            acc.append(accuracy.item())
        a2 = time.time()
        mean_loss.append(np.mean(losses))
        print('Epoch: ' + str(epoch) + ', train loss: ' + str(np.mean(losses)) +
              ', acc: ' + str(np.mean(acc)) + ', time: ' + str(a2 - a1))
        loss_test = test(epoch, model)
        mean_loss_test.append(loss_test)
        if epoch % 1 == 0 and epoch > 5:
            test_selection_prediction(False, 0, 1008, 'fc' + str(epoch) + '.pt', 0.1, model)
            torch.save(model.state_dict(), './models/training_with_selection/model_'
                       + str(epoch) + '.pt')
            np.save('./models/training_with_selection/model_loss_'
                       + str(epoch) + '.npy', mean_loss)
            np.save('./models/training_with_selection/model_loss_test_'
                    + str(epoch) + '.npy', mean_loss_test)


def test(epoch, model):
    cells = []
    val_losses = []
    val_losses_be = []
    acc = []
    acc2 = []
    with torch.no_grad():
        model.eval()
        for x_val, y_val in val_loader:
            yhat = model.step(x_val)
            val_loss = loss_fn(yhat.view(-1, 10000), y_val.view(-1, 10000))
            val_losses_be.append(val_loss.item())
            yhat = sig(yhat)
            yhat = torch.where((yhat>=thre), 1, 0)
            #print(yhat.sum())

            accuracy = 1 - abs(yhat - y_val).sum().mean() / 10000
            acc.append(accuracy.item())
            acc2.append(abs(y_val.sum()-yhat.sum()).item())
    print('Epoch: ' + str(epoch) + ', validation loss: ' + str(np.mean(val_losses_be))
          + ', acc: ' + str(np.mean(acc)) + ', label1 acc: ' + str(np.mean(acc2)))
    print('-' * 50)
    return np.mean(val_losses_be)


def testJ(epoch, model, thre=0.5):
    #print('threshold='+str(thre))
    cells = []
    val_losses = []
    val_losses_be = []
    acc = []
    acc2 = []
    TP = torch.zeros((100, 100)).to(device)
    TN = torch.zeros((100, 100)).to(device)
    FP = torch.zeros((100, 100)).to(device)
    FN = torch.zeros((100, 100)).to(device)
    with torch.no_grad():
        model.eval()
        for x_val, y_val in val_loader:
            yhat = model.step(x_val)[0]
            y_val = y_val[0]
            val_loss = loss_fn(yhat.view(-1, 10000), y_val.view(-1, 10000))
            val_losses_be.append(val_loss.item())
            yhat = sig(yhat)

            nonz = torch.where(yhat > thre)
            yhat[nonz] = 1
            nonz = torch.where(yhat <= thre)
            yhat[nonz] = 0

            accuracy = 1 - abs(yhat - y_val).sum().mean() / 10000
            acc.append(accuracy.item())
            acc2.append(abs(y_val.sum()-yhat.sum()).item())

            TPp = torch.where(((yhat==1) & (y_val==1)), 1, 0)
            TNp = torch.where(((yhat==0) & (y_val==0)), 1, 0)
            FPp = torch.where(((yhat==1) & (y_val==0)), 1, 0)
            FNp = torch.where(((yhat==0) & (y_val==1)), 1, 0)
            TP += TPp
            TN += TNp
            FP += FPp
            FN += FNp
    sen = TP / (TP + FN)
    ind = torch.where(((TP!=0)|(FN!=0)))
    print(str(len(ind[0]))+' values are not 0 in sen')
    print('sen='+str(sen[ind].mean()))
    spe = TN / (FP + TN)
    ind2 = torch.where(((FP != 0) | (TN != 0)))
    print(str(len(ind2[0])) + ' values are not 0 in spe')
    print('spe=' + str(spe[ind2].mean()))
    print('J='+str(sen[ind].mean()+spe[ind2].mean()-1))
    pre = TP.sum() / (FP.sum() + TP.sum())
    rec = TP.sum() / (FN.sum() + TP.sum())
    micro = (2*pre*rec) / (pre+rec)
    pre = TP / (FP + TP)
    rec = TP / (FN + TP)
    ind = torch.where(((FP != 0) | (TP != 0)))
    ind2 = torch.where(((FN != 0) | (TP != 0)))
    macro = (2*pre[ind].mean()*rec[ind2].mean()) / (pre[ind].mean()+rec[ind2].mean())
    print('micro F1='+str(micro)+', macro F1='+str(macro))
    print('Epoch: ' + str(epoch) + ', validation loss: ' + str(np.mean(val_losses_be))
          + ', acc: ' + str(np.mean(acc)) + ', label1 acc: ' + str(np.mean(acc2)))
    print('-' * 50)
    return (sen+spe-1).view(1, 100, 100)


if __name__ == '__main__':
    #processData()
    #genWhole()
    #genData()
    #genDataTest()
    #model = selection_prediction(1, 32, (5, 5), 3, CELL_SIZE).to(device)
    #model.load_state_dict(torch.load('models/training_with_selection/model_69.pt',
    #                                 map_location=torch.device('cpu')))
    model = selection_prediction2().to(device)
    #model.load_state_dict(torch.load('models/training_with_selection/best_selection_prediction.pt',
    #                                map_location=torch.device('cpu')))
    thre = 0.3958 #torch.load('./thre.pt', map_location=torch.device('cpu')).to(device)

    lr = 1e-4
    n_epochs = 100

    losses = []
    val_losses = []
    batchsize = 128
    sig = nn.Sigmoid()

    x_train = torch.load('./models/training_with_selection/train_x.pt', map_location=torch.device('cpu')).to(device).float()
    y_train = torch.load('./models/training_with_selection/train_y.pt', map_location=torch.device('cpu')).to(device).float()
    ones = y_train.sum(dim=0)
    total = y_train.size()[0]
    weight = ((total - ones) / ones).view(10000)

    '''size = total
    y = y_train.cpu().numpy()
    a = y.sum(axis=0).flatten() / size
    print('min+max/2=' + str((np.max(a) + np.min(a)) / 2))

    hist, bin_edges = np.histogram(a, bins=np.arange(0, 1, 0.1))
    hist = hist / 10000
    mean = np.mean(hist)
    median = np.median(hist)
    std = np.std(hist)
    print('mean=' + str(mean))
    print('median=' + str(median))
    print('std=' + str(std))
    print('value=' + str((mean - median) / std))
    quit()'''

    loss_fn = nn.BCEWithLogitsLoss(weight=weight)  #TODO
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    train_step = make_train_step(model, loss_fn, optimizer)

    x_val = torch.load('./models/training_with_selection/val_x.pt', map_location=torch.device('cpu')).to(device).float()
    y_val = torch.load('./models/training_with_selection/val_y.pt', map_location=torch.device('cpu')).to(device).float()
    print('training x size:'+str(x_train.size()))
    print('training y size:' + str(y_train.size()))

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)


    train()
    #test(0, model)
    '''ind_m = torch.arange(0.01, 0.99, 0.01)
    m = testJ(0, model, 0.01)
    for i in range(1, 99, 1):
        i = float(i/100)
        #testJ(0, model, i)
        m = torch.cat((m, testJ(0, model, i)), dim=0)
    value, ind = m.max(dim=0)
    res = torch.zeros((100, 100))
    for x in range(100):
        for y in range(100):
            res[x][y] = ind_m[ind[x][y]]
    torch.save(res, './thre.pt')'''
    '''for i in range(1, 10):
        model.load_state_dict(torch.load('models/training_with_selection/model_'+str(i)+'.pt',
                                        map_location=torch.device('cpu')))
        test(i, model)'''