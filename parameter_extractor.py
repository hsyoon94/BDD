import torch
import numpy as np
import os

from model import MLP
from pathlib import Path

model_param_dir = './model_param/'
model_param_name = '200811_1920'


is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

network_input_dim = 1000
network_output_dim = 10

load_model = MLP(network_input_dim, network_output_dim, device)
load_model.load_state_dict(torch.load('./trained_model/model_' + model_param_name + '/epoch1210.pt'))
os.mkdir(model_param_dir + model_param_name + '_1210/')

for k in load_model.state_dict():
    print(k)
    param = load_model.state_dict()[k].cpu().detach().numpy()

    print(param.shape)

    model_save_dir = model_param_dir + model_param_name + '_1210/' + str(k) + '.txt'

    np.savetxt(model_save_dir, param)