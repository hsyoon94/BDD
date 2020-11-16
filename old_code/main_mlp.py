import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as utils
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

import numpy as np
import time
from datetime import datetime
from model import MLP
import os


def array_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def array_rnd_sample(a, b):
    assert len(a) == len(b)
    idx = np.random.choice(np.arange(a.shape[0]), 600, replace=False)
    return a[idx], b[idx]


def train(epoch, model, train_data, optimizer, criterion, device):
    loss_total = 0
    states_tmp = train_data['states']
    actions_tmp = train_data['actions']

    states_tmp, actions_tmp = array_shuffle(states_tmp, actions_tmp)
    states, actions = array_rnd_sample(states_tmp, actions_tmp)

    for i in range(states.shape[0]):
        state = states[i]
        action = actions[i]

        state = Variable(torch.from_numpy(state)).to(device)
        action = Variable(torch.from_numpy(action)).to(device)

        optimizer.zero_grad()
        predicted_action = model.forward(state.float())
        loss = criterion(predicted_action, action.float())
        loss_total = loss_total + loss
        loss.backward()
        optimizer.step()

        if i % (states.shape[0] / 5) == 0:
            print("Epoch", epoch, " Iteration", i, " Loss", loss)

        # if epoch >= 400 and loss >= 100:
        #     print("err            ", loss)
        #     print("err state      ", state)
        #     print("err action gt  ", action)
        #     print("err action pred", predicted_action)

    loss_total = loss_total.cpu().detach().numpy()
    print("Epoch", epoch, " Iter loss", loss_total)
    return loss_total


def test(model, test_data):
    print("Testing starts...")
    test_loss = 0

    for i in range(test_data['states'].shape[0]):
        state = test_data['states'][i]
        action = test_data['actions'][i]

        state = Variable(torch.from_numpy(state)).to(device)
        action = Variable(torch.from_numpy(action)).to(device)

        with torch.no_grad():
            predicted_action = model.forward(state.float())
            test_loss = test_loss + criterion(predicted_action, action.float())

        if i % 10000 == 0:
            print("Test iteration", i)

    print("Testing ends...")
    test_loss = test_loss.cpu().detach().numpy()
    return test_loss


now = datetime.now()
now_date = str(now.year)[-2:] + str(now.month).zfill(2) + str(now.day).zfill(2)
now_time = str(now.hour).zfill(2) + str(now.minute).zfill(2)

os.mkdir('./trained_model/model_' + now_date + '_' + now_time + '/')

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

expert_demo_train = torch.load('./expert_demo/expert_demo_train.pt')
expert_demo_test = torch.load('./expert_demo/expert_demo_train.pt')

network_input_dim = expert_demo_train['states'].shape[1]
network_output_dim = expert_demo_train['actions'].shape[1]

epoch = 10000
model = MLP(network_input_dim, network_output_dim, device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

print(model)

loss_train_arr = np.array([])
loss_test_arr = np.array([])

for i in range(epoch):
    loss_train_tmp = train(i, model, expert_demo_train, optimizer, criterion, device)
    loss_train_arr = np.append(loss_train_arr, [loss_train_tmp])
    if i % 10 == 0:
        test_loss_tmp = test(model, expert_demo_test)
        loss_test_arr = np.append(loss_test_arr, [test_loss_tmp])
        print("Test loss:", test_loss_tmp)

        torch.save(model.state_dict(), './trained_model/model_' + now_date + '_' + now_time + '/epoch' + str(i) + '.pt')
        print("save complete with " + now_date + '_' + now_time + '.txt')


np.savetxt('./results/' + now_date + '_' + now_time + '_loss_train.txt', loss_train_arr)
np.savetxt('./results/' + now_date + '_' + now_time + '_loss_test.txt', loss_test_arr)