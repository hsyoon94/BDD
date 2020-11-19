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
from model import MLP, LPNET, LPNET_MLP, LPNET_R2P2, LPNET_V03, LPNET_V03_sep, LPNET_V04, LPNET_MLP05
import os

torch.set_num_threads(2)

now = datetime.now()
now_date = str(now.year)[-2:] + str(now.month).zfill(2) + str(now.day).zfill(2)
now_time = str(now.hour).zfill(2) + str(now.minute).zfill(2)

# if os.path.exists('/mnt/sda2/BDD/log/0/model_' + now_date + '_' + now_time + '/') is False:
#     os.mkdir('/mnt/sda2/BDD/log/0/model_' + now_date + '_' + now_time + '/')

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

test_set_dir = '/mnt/sda2/BDD/data_test/'
trained_model_dir = '/mnt/sda2/BDD/log/'

lpnet0_dir = trained_model_dir + '0/model_201119_1452/epoch0_27.pt'
lpnet1_dir = trained_model_dir + '1/model_201119_1452/epoch0_2.pt'
lpnet2_dir = trained_model_dir + '2/model_201119_1452/epoch0_30.pt'
lpnet3_dir = trained_model_dir + '3/model_201119_1452/epoch0_8.pt'
lpnet4_dir = trained_model_dir + '4/model_201119_1452/epoch0_30.pt'
lpnet5_dir = trained_model_dir + '5/model_201119_1452/epoch0_18.pt'
lpnet6_dir = trained_model_dir + '6/model_201119_1453/epoch0_19.pt'

# With above expert_dmo_train_dir, extract data file name list and save to train_data_name_list
test_data_name_list = [f for f in listdir(test_set_dir) if isfile(join(test_set_dir, f))]

"""
NETWORK STRUCTURE CONFIG
"""
FEATURE_DIM = 128
BP_DIM = 7
LPNET_OUTPUT = 10
LPNET_LSTM_INPUT = FEATURE_DIM + BP_DIM

mse_loss_total = 0
mse_loss0 = 0
mse_loss1 = 0
mse_loss2 = 0
mse_loss3 = 0
mse_loss4 = 0
mse_loss5 = 0
mse_loss6 = 0

count_bp0 = 0
count_bp1 = 0
count_bp2 = 0
count_bp3 = 0
count_bp4 = 0
count_bp5 = 0
count_bp6 = 0

with open(test_set_dir + '/' + test_data_name_list[0]) as tmp_json2:
    json_for_dynamic_input = json.load(tmp_json2)
    output_path = json_for_dynamic_input['data']['future_trajectory_tr']
    bp_tmp = json_for_dynamic_input['data']['behavior']
    LPNET_OUTPUT = len(output_path) * 2
    BP_DIM = len(bp_tmp)
    print("Local Path Length Set Completed with", LPNET_OUTPUT)
    print("BP Length Set Completed with", BP_DIM)

lpnet0 = LPNET_MLP05(rnn_output_dim=100, path_length=LPNET_OUTPUT, device=device)
lpnet1 = LPNET_MLP05(rnn_output_dim=100, path_length=LPNET_OUTPUT, device=device)
lpnet2 = LPNET_MLP05(rnn_output_dim=100, path_length=LPNET_OUTPUT, device=device)
lpnet3 = LPNET_MLP05(rnn_output_dim=100, path_length=LPNET_OUTPUT, device=device)
lpnet4 = LPNET_MLP05(rnn_output_dim=100, path_length=LPNET_OUTPUT, device=device)
lpnet5 = LPNET_MLP05(rnn_output_dim=100, path_length=LPNET_OUTPUT, device=device)
lpnet6 = LPNET_MLP05(rnn_output_dim=100, path_length=LPNET_OUTPUT, device=device)

lpnet0.load_state_dict(torch.load(lpnet0_dir))
lpnet1.load_state_dict(torch.load(lpnet1_dir))
lpnet2.load_state_dict(torch.load(lpnet2_dir))
lpnet3.load_state_dict(torch.load(lpnet3_dir))
lpnet4.load_state_dict(torch.load(lpnet4_dir))
lpnet5.load_state_dict(torch.load(lpnet5_dir))
lpnet6.load_state_dict(torch.load(lpnet6_dir))

lpnet0.eval()
lpnet1.eval()
lpnet2.eval()
lpnet3.eval()
lpnet4.eval()
lpnet5.eval()
lpnet6.eval()

criterion = nn.MSELoss()

for i in range(len(test_data_name_list)):

    print("test data length", len(test_data_name_list))

    try:
        with open(test_set_dir + '/' + test_data_name_list[i]) as tmp_json:
            input_json = json.load(tmp_json)
    except ValueError:
        print("JSON value error with ", test_data_name_list[i])
        continue
    except IOError:
        print("JSON IOerror with ", test_data_name_list[i])
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

    input_other_car = input_tp + input_tv + input_tvh
    input_my_car = input_ev + input_evh
    input_static = input_lane + input_gp + input_cl

    input = np.vstack([input_static, input_other_car, input_my_car])
    input = np.reshape(input, [1, input.shape[0], input.shape[1], input.shape[2]])
    input_tensor = torch.tensor(input).to(device)

    output_raw = input_json['data']['future_trajectory_tr']
    output = []
    for output_tmp in output_raw:
        output.append(output_tmp[0])
        output.append(output_tmp[1])

    output = torch.tensor(output).to(device).double()

    if input_bp[0] == 1:
        predicted_output = lpnet0.forward(input_tensor.float())
        mse_loss0 = mse_loss0 + nn.MSELoss()(output, predicted_output[0])
        count_bp0 = count_bp0 + 1

    elif input_bp[1] == 1:
        predicted_output = lpnet1.forward(input_tensor.float())
        mse_loss1 = mse_loss1 + nn.MSELoss()(output, predicted_output[0])
        count_bp1 = count_bp1 + 1

    elif input_bp[2] == 1:
        predicted_output = lpnet2.forward(input_tensor.float())
        mse_loss2 = mse_loss2 + nn.MSELoss()(output, predicted_output[0])
        count_bp2 = count_bp2 + 1

    elif input_bp[3] == 1:
        predicted_output = lpnet3.forward(input_tensor.float())
        mse_loss3 = mse_loss3 + nn.MSELoss()(output, predicted_output[0])
        count_bp3 = count_bp3 + 1

    elif input_bp[4] == 1:
        predicted_output = lpnet4.forward(input_tensor.float())
        mse_loss4 = mse_loss4 + nn.MSELoss()(output, predicted_output[0])
        count_bp4 = count_bp4 + 1

    elif input_bp[5] == 1:
        predicted_output = lpnet5.forward(input_tensor.float())
        mse_loss5 = mse_loss5 + nn.MSELoss()(output, predicted_output[0])
        count_bp5 = count_bp5 + 1

    elif input_bp[6] == 1:
        predicted_output = lpnet6.forward(input_tensor.float())
        mse_loss6 = mse_loss6 + nn.MSELoss()(output, predicted_output[0])
        count_bp6 = count_bp6 + 1

    mse_loss_total = mse_loss0 + mse_loss1 + mse_loss2 + mse_loss3 + mse_loss4 + mse_loss5 + mse_loss6

mse_loss0_avg = mse_loss0 / count_bp0
mse_loss1_avg = mse_loss1 / count_bp1
mse_loss2_avg = mse_loss2 / count_bp2
mse_loss3_avg = mse_loss3 / count_bp3
mse_loss4_avg = mse_loss4 / count_bp4
mse_loss5_avg = mse_loss5 / count_bp5
mse_loss6_avg = mse_loss6 / count_bp6

mse_loss_total_avg = mse_loss_total / len(test_data_name_list)

print("AVERAGE MSE LOSS 0", mse_loss0_avg)
print("AVERAGE MSE LOSS 1", mse_loss1_avg)
print("AVERAGE MSE LOSS 2", mse_loss2_avg)
print("AVERAGE MSE LOSS 3", mse_loss3_avg)
print("AVERAGE MSE LOSS 4", mse_loss4_avg)
print("AVERAGE MSE LOSS 5", mse_loss5_avg)
print("AVERAGE MSE LOSS 6", mse_loss6_avg)
print("AVERAGE MSE LOSS TOTAL", mse_loss_total_avg)