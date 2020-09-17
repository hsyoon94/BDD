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


def train(epoch, model, cnn_model, train_data_dir, optimizer, criterion, device):
    loss_total = 0

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    rnd_index = np.random.choice(32999, size=3000, replace=False)

    for i in range(rnd_index.shape[0]):

        im = pilimg.open(train_data_dir + '/sem_img/img' + str(rnd_index[i]) + '.png')
        im = np.array(im)
        im = im[:, :, 0:3]
        input = Image.fromarray(im, 'RGB')
        input_tensor = preprocess(input)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            cnn_model.to('cuda')

        with torch.no_grad():
            cnn_output = cnn_model(input_batch)

        cnn_output = np.squeeze(cnn_output)

        output = np.loadtxt(train_data_dir + '/path/path' + str(rnd_index[i]) + '.txt')
        output = np.array([output])

        output = torch.tensor(output).to(device)

        optimizer.zero_grad()
        predicted_output = model.forward(cnn_output.float()).double()

        loss = criterion(predicted_output, output) + lane_loss

        loss_total = loss_total + loss
        loss.backward()
        optimizer.step()

        if i % 300 == 0:
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

expert_demo_train_dir = '/media/hsyoon/HS_VILAB1/BDD/network_dataset'

epoch = 1000000

MLP_INPUT = 512
MLP_OUTPUT = 10

model = MLP(MLP_INPUT, MLP_OUTPUT, device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

resnet18_full = models.resnet18(pretrained=True)
resnet18_crop = torch.nn.Sequential(*(list(resnet18_full.children())[:-1]))

loss_train_arr = np.array([])
loss_test_arr = np.array([])

for i in range(epoch):
    loss_train_tmp = train(i, model, resnet18_crop, expert_demo_train_dir, optimizer, criterion, device)
    loss_train_arr = np.append(loss_train_arr, [loss_train_tmp])
    if i % 10 == 0:

        torch.save(model.state_dict(), './trained_model/model_' + now_date + '_' + now_time + '/epoch' + str(i) + '.pt')
        print("save complete with " + now_date + '_' + now_time + '.txt')


np.savetxt('./results/' + now_date + '_' + now_time + '_loss_train.txt', loss_train_arr)
