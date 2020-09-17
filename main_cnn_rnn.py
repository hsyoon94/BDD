import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as utils
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms


from PIL import Image
import PIL.Image as pilimg
import numpy as np
import time
from datetime import datetime
from model import MLP, LPNET
import os


def array_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def array_rnd_sample(a, b):
    assert len(a) == len(b)
    idx = np.random.choice(np.arange(a.shape[0]), 600, replace=False)
    return a[idx], b[idx]


def train(epoch, model, train_data_dir, optimizer, criterion, device):
    loss_total = 0

    data_size = 50
    # random number array for data index
    rnd_index = np.random.choice(214, data_size, replace=False)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for i in range(rnd_index.shape[0]):

        im = pilimg.open(train_data_dir + '/sem_img/img' + str(rnd_index[i]) + '.png')
        im = np.array(im)
        im = im[:, :, 0:3]
        input = Image.fromarray(im, 'RGB')
        input_tensor = preprocess(input)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        # Now the input is ready. In later, above process is not needed because IE will come from IE.

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        optimizer.zero_grad()
        BP_EXAMPLE = [0, 0, 0, 1, 0, 0, 0]
        predicted_output = model.forward(input_batch.float(), BP_EXAMPLE).double()

        # print("predicted_output", predicted_output)

        # GT output / path
        output = np.loadtxt(train_data_dir + '/path/path' + str(rnd_index[i]) + '.txt')
        output = torch.tensor(output).to(device)

        # groud truth output
        loss = criterion(predicted_output, output)

        loss_total = loss_total + loss
        loss.backward()
        optimizer.step()

        if i % (data_size/5) == 0:
            print("Training with", epoch, "epoch,", i, "steps")

    loss_total = loss_total.cpu().detach().numpy()
    print("Epoch", epoch, " Iter loss", loss_total)
    return loss_total

now = datetime.now()
now_date = str(now.year)[-2:] + str(now.month).zfill(2) + str(now.day).zfill(2)
now_time = str(now.hour).zfill(2) + str(now.minute).zfill(2)

if os.path.exists('./trained_model/model_' + now_date + '_' + now_time + '/') is False:
    os.mkdir('./trained_model/model_' + now_date + '_' + now_time + '/')

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

expert_demo_train_dir = './network_dataset'

epoch = 1000000

"""
NETWORK STRUCTURE CONFIG
"""
FEATURE_DIM = 128
BP_DIM = 7
IE_CHANNEL = 7
LPNET_OUTPUT = 10
# LPNET_OUTPUT = 30
LPNET_LSTM_INPUT = FEATURE_DIM + BP_DIM

model = LPNET(rnn_output_dim=2, bp_dim=BP_DIM, path_length=LPNET_OUTPUT, device=device)
print(model)

optimizer = optim.Adam(model.parameters(), lr=0.00001)
criterion = nn.MSELoss()

for i in range(epoch):

    loss_train_tmp = train(i, model, expert_demo_train_dir, optimizer, criterion, device)
    myfile = open('./trained_model/model_' + now_date + '_' + now_time + '/loss.txt', 'a')
    myfile.write(str(loss_train_tmp) + '\n')
    myfile.close()

    if i % 20 == 0:

        torch.save(model.state_dict(), './trained_model/model_' + now_date + '_' + now_time + '/epoch' + str(i) + '.pt')
        print("save complete with " + now_date + '_' + now_time)


