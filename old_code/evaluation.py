import torch
import numpy as np

from model import MLP
from pathlib import Path

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

            if i <= 600 and i % 100 == 0:
                print("Ground truth :", action)
                print("Prediction :", predicted_action)

    print("Testing ends...")


model_date = '200729_2029'
ep = 7000

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

expert_demo_train = torch.load('./expert_demo/expert_demo_train.pt')
expert_demo_test = torch.load('./expert_demo/expert_demo_train.pt')

network_input_dim = expert_demo_train['states'].shape[1]
network_output_dim = expert_demo_train['actions'].shape[1]

load_model = MLP(network_input_dim, network_output_dim, device)
load_model.load_state_dict(torch.load('./trained_model/model_' + model_date + '/epoch' + str(ep) + '.pt'))

test_loss_tmp = test(load_model, expert_demo_test)