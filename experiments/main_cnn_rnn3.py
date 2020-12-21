import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as utils
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms
from os import listdir
from os.path import isfile, join
import json

import math
from PIL import Image
import PIL.Image as pilimg
import numpy as np
import time
from datetime import datetime
from model import MLP, LPNET, LPNET_MLP, LPNET_R2P2, LPNET_V03, LPNET_V03_sep, LPNET_V04
import os

torch.set_num_threads(2)

def array_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def array_rnd_sample(a, b):
    assert len(a) == len(b)
    idx = np.random.choice(np.arange(a.shape[0]), 600, replace=False)
    return a[idx], b[idx]


def train(epoch, model, train_data_dir, train_data_name_list, optimizer, criterion, device):
    loss_total = 0

    data_size = 200
    # random number array for data index
    rnd_index = np.random.choice(len(train_data_name_list)-1, int(len(train_data_name_list)/10), replace=False)


    for i in range(rnd_index.shape[0]):

        try:
            with open(train_data_dir + '/' + train_data_name_list[rnd_index[i]]) as tmp_json:
                input_json = json.load(tmp_json)
        except ValueError:
            print("JSON value error with ", train_data_name_list[rnd_index[i]])
            continue
        except IOError:
            print("JSON IOerror with ", train_data_name_list[rnd_index[i]])
            continue

        input_tp = np.array([input_json['data']['tp_grid']])
        input_ev = np.array([input_json['data']['ev_grid']])
        input_evh = np.array([input_json['data']['evh_grid']])
        input_tv = np.array([input_json['data']['tv_grid']])
        input_tvh = np.array([input_json['data']['tvh_grid']])
        input_lane = np.array([input_json['data']['lane_grid']])
        input_cl = np.array([input_json['data']['cl_grid']])
        input_gp = np.array([input_json['data']['gp_grid']])
        input_bp = np.array(input_json['data']['behavior'])
        input_lane_coef = np.array(input_json['data']['lane_coef'])

        input = np.vstack([input_tp, input_ev, input_evh, input_tv, input_tvh, input_lane, input_cl, input_gp])
        # input = np.vstack([input_ev, input_gp])
        input = np.reshape(input, [1, input.shape[0], input.shape[1], input.shape[2]])

        input_tensor = torch.tensor(input).to(device)

        # Now the input is ready. In later, above process is not needed because IE will come from IE.

        optimizer.zero_grad()
        predicted_output = model.forward(input_tensor.float())

        predicted_output = predicted_output.double()

        # GT output / path
        output_raw = input_json['data']['future_trajectory_tr']

        output = []
        for output_tmp in output_raw:
            output.append(output_tmp[0])
            output.append(output_tmp[1])

        output = torch.tensor(output).to(device).double()


        if input_lane_coef.all() == 0.0:
            loss_weight_lane = 0
            loss_weight_mse = 1
        else:
            loss_weight_lane = 0
            loss_weight_mse = 1

        lane_loss = 0

        for j in range(int(predicted_output.shape[0] / 2)):
            lane_loss = lane_loss + abs(predicted_output[(j * 2) + 1] - input_lane_coef[0] * math.pow(predicted_output[j * 2], 3) + input_lane_coef[1] * math.pow(predicted_output[j * 2], 2) + input_lane_coef[2] * predicted_output[j * 2] + input_lane_coef[3])

        mse_loss = criterion(predicted_output, output)
        loss = loss_weight_mse * mse_loss + loss_weight_lane * lane_loss

        # print("mse_loss", mse_loss)
        # print("lane_loss", lane_loss)

        loss_total = loss_total + loss.clone()
        loss_total = loss_total.cpu().detach()
        loss.backward()
        optimizer.step()

        if i % (data_size/5) == 0:
            print("Training with", epoch, "epoch,", i, "steps")

    # loss_total = loss_total.cpu().detach().numpy()
    loss_total = loss_total.numpy() / (rnd_index.shape[0] * 5)
    print("Epoch", epoch, " Iter loss", loss_total)
    return loss_total


now = datetime.now()
now_date = str(now.year)[-2:] + str(now.month).zfill(2) + str(now.day).zfill(2)
now_time = str(now.hour).zfill(2) + str(now.minute).zfill(2)

if os.path.exists('/mnt/sda2/BDD/log/3/model_' + now_date + '_' + now_time + '/') is False:
    os.mkdir('/mnt/sda2/BDD/log/3/model_' + now_date + '_' + now_time + '/')

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

expert_demo_train_dir = '/mnt/sda2/BDD/data/3'

# With above expert_dmo_train_dir, extract data file name list and save to train_data_name_list
train_data_name_list = [f for f in listdir(expert_demo_train_dir) if isfile(join(expert_demo_train_dir, f))]

epoch = 1000000

"""
NETWORK STRUCTURE CONFIG
"""
FEATURE_DIM = 128
BP_DIM = 7
LPNET_OUTPUT = 30
LPNET_LSTM_INPUT = FEATURE_DIM + BP_DIM

with open(expert_demo_train_dir + '/' + train_data_name_list[0]) as tmp_json2:
    json_for_dynamic_input = json.load(tmp_json2)
    output_path = json_for_dynamic_input['data']['future_trajectory_tr']
    bp_tmp = json_for_dynamic_input['data']['behavior']
    LPNET_OUTPUT = len(output_path) * 2
    BP_DIM = len(bp_tmp)
    print("Local Path Length Set Completed with", LPNET_OUTPUT)
    print("BP Length Set Completed with", BP_DIM)

model = LPNET_V04(rnn_output_dim=100, path_length=LPNET_OUTPUT, device=device)

print(model)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

for i in range(epoch):

    loss_train_tmp = train(i, model, expert_demo_train_dir, train_data_name_list, optimizer, criterion, device)

    # myfile = open('./trained_model/model_' + now_date + '_' + now_time + '/loss.txt', 'a')
    myfile = open('/mnt/sda2/BDD/log/3/model_' + now_date + '_' + now_time + '/loss.txt', 'a')

    myfile.write(str(loss_train_tmp) + '\n')
    myfile.close()

    if i % 20 == 0:

        # torch.save(model.state_dict(), './trained_model/model_' + now_date + '_' + now_time + '/epoch' + str(i) + '_' + str(int(loss_train_tmp)) + '.pt')
        torch.save(model.state_dict(), '/mnt/sda2/BDD/log/3/model_' + now_date + '_' + now_time + '/epoch' + str(i) + '_' + str(int(loss_train_tmp)) + '.pt')
        print("save complete with " + now_date + '_' + now_time)


