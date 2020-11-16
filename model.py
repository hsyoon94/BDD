import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import datasets, models, transforms

import numpy as np
import matplotlib.pyplot as plt

from torch.autograd import Variable
import torchvision.utils as utils
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import math

import numpy as np
import time
from datetime import datetime

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, output_dim)
        ).to(device)

    def forward(self, x):
        x = self.layers(x)
        return x


class LPNET_V04(nn.Module):
    def __init__(self, rnn_output_dim, path_length, device):
        super(LPNET_V04, self).__init__()

        # self.preprocess = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

        self.lane_coef = None
        self.device = device
        self.rnn_output_dim = rnn_output_dim  # 100
        self.dynamic_input_dim = path_length

        self.F0_DIM = 256
        self.F1_DIM = 128

        self.ReLU = nn.ReLU().to(self.device)
        self.IE_CHANNEL = 2

        self.CNN1 = nn.Conv2d(self.IE_CHANNEL, self.IE_CHANNEL, 5, stride=2).to(self.device)
        self.CNN2 = nn.Conv2d(self.IE_CHANNEL, self.IE_CHANNEL / 2, 3, stride=1).to(self.device)
        self.MaxPool = nn.MaxPool2d(3, 1).to(self.device)

        # # Unremark below with 8 channel input (IE)
        # self.resnet18[0] = nn.Conv2d(self.IE_CHANNEL, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(device)

        self.F0_NET = nn.Linear(6724, self.F1_DIM).to(self.device).to(self.device)

        self.GRU = nn.GRU(self.dynamic_input_dim, self.rnn_output_dim, 1).to(self.device)

        self.mlp1_for_rnn_output = nn.Linear(self.rnn_output_dim + self.F1_DIM,int(self.rnn_output_dim / 2)).to(self.device)
        self.mlp2_for_rnn_output = nn.Linear(int(self.rnn_output_dim / 2), 2).to(self.device)  # generate point

    def forward(self, information_embedder):

        dynamic_input = np.zeros(self.dynamic_input_dim)
        dynamic_input = torch.tensor(dynamic_input).to(self.device).float()

        # ResNet18
        # prediction = self.resnet18(information_embedder)
        prediction = self.CNN1(information_embedder)
        prediction = self.ReLU(prediction)

        prediction = self.MaxPool(prediction)

        prediction = self.CNN2(prediction)
        prediction = self.ReLU(prediction)
        prediction = self.MaxPool(prediction)

        # print(prediction.clone().cpu().detach().numpy().shape)
        # plt.imshow(prediction.clone().cpu().detach().numpy()[0, 0, :, :])
        # plt.show()

        # plt.imshow(transforms.ToPILImage()(prediction.clone().cpu()[0,0 , :, :]))
        # plt.show()

        self.F0_DIM = prediction.shape[0] * prediction.shape[1] * prediction.shape[2] * prediction.shape[3]
        prediction = prediction.view(-1, self.F0_DIM)  # reshape Variable

        # prediction = torch.reshape(prediction, (prediction.shape[1],))

        # F0 -> F1
        prediction = self.F0_NET(prediction)
        prediction = self.ReLU(prediction)
        prediction = torch.squeeze(prediction)

        # RNN
        with torch.autograd.set_detect_anomaly(True):
            for i in range(int(dynamic_input.shape[0] / 2)):
                dynamic_input_reshape = torch.reshape(dynamic_input, (1, 1, dynamic_input.shape[0])).clone()

                rnn_output, hidden = self.GRU(dynamic_input_reshape)

                rnn_output = torch.squeeze(rnn_output)

                concat = torch.cat((prediction, rnn_output), 0)

                rnn_output_encoded = self.mlp1_for_rnn_output(concat)
                tmp_local_acc = self.mlp2_for_rnn_output(rnn_output_encoded)

                # print("tmp_local_path", tmp_local_path)
                if i == 0:
                    dynamic_input[i * 2] = tmp_local_acc[0]
                    dynamic_input[(i * 2) + 1] = tmp_local_acc[1]

                elif i == 1:
                    dynamic_input[i * 2] = 2 * dynamic_input[(i-1) * 2] + tmp_local_acc[0]
                    dynamic_input[(i * 2) + 1] = 2 * dynamic_input[((i-1) * 2) + 1] + tmp_local_acc[1]

                else:
                    dynamic_input[i * 2] = 2 * dynamic_input[(i - 1) * 2]  - dynamic_input[(i - 2) * 2]+ tmp_local_acc[0]
                    dynamic_input[(i * 2) + 1] = 2 * dynamic_input[((i - 1) * 2) + 1] - dynamic_input[((i - 2) * 2) + 1] + tmp_local_acc[1]

        return dynamic_input



class LPNET_V03_sep(nn.Module):
    def __init__(self, rnn_output_dim, path_length, device):
        super(LPNET_V03_sep, self).__init__()

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.lane_coef = None
        self.device = device
        self.rnn_output_dim = rnn_output_dim  # 100
        self.dynamic_input_dim = path_length

        self.F0_DIM = 512
        self.F1_DIM = 256
        self.F2_DIM = 128
        self.ReLU = nn.ReLU().to(self.device)
        self.IE_CHANNEL = 8

        self.resnet18 = models.resnet18(pretrained=True).to(self.device)
        self.resnet18 = torch.nn.Sequential(*(list(self.resnet18.children())[:-1]))
        # Unremark below with 8 channel input (IE)
        self.resnet18[0] = nn.Conv2d(self.IE_CHANNEL, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(device)

        self.F0_NET = nn.Linear(self.F0_DIM, self.F1_DIM).to(self.device)
        self.F1_NET = nn.Linear(self.F1_DIM, self.F2_DIM).to(self.device)


        self.GRU = nn.GRU(self.dynamic_input_dim, self.rnn_output_dim, 1).to(self.device)

        self.mlp1_for_rnn_output = nn.Linear(self.rnn_output_dim + self.F2_DIM,int(self.rnn_output_dim / 2)).to(self.device)
        self.mlp2_for_rnn_output = nn.Linear(int(self.rnn_output_dim / 2), 2).to(self.device)  # generate point

    def forward(self, information_embedder):

        dynamic_input = np.zeros(self.dynamic_input_dim)
        dynamic_input = torch.tensor(dynamic_input).to(self.device).float()

        # ResNet18
        prediction = self.resnet18(information_embedder)
        prediction = torch.reshape(prediction, (prediction.shape[1],))

        # F0 -> F1
        prediction = self.F0_NET(prediction)
        prediction = self.ReLU(prediction)

        # F1 -> F2
        prediction = self.F1_NET(prediction)
        prediction = self.ReLU(prediction)

        lane_loss = 0

        # RNN
        with torch.autograd.set_detect_anomaly(True):
            for i in range(int(dynamic_input.shape[0] / 2)):
                dynamic_input_reshape = torch.reshape(dynamic_input, (1, 1, dynamic_input.shape[0])).clone()

                rnn_output, hidden = self.GRU(dynamic_input_reshape)

                rnn_output = torch.squeeze(rnn_output)

                concat = torch.cat((prediction, rnn_output), 0)

                rnn_output_encoded = self.mlp1_for_rnn_output(concat)
                tmp_local_acc = self.mlp2_for_rnn_output(rnn_output_encoded)

                # print("tmp_local_path", tmp_local_path)
                if i == 0:
                    dynamic_input[i * 2] = tmp_local_acc[0]
                    dynamic_input[(i * 2) + 1] = tmp_local_acc[1]

                elif i == 1:
                    dynamic_input[i * 2] = 2 * dynamic_input[(i-1) * 2] + tmp_local_acc[0]
                    dynamic_input[(i * 2) + 1] = 2 * dynamic_input[((i-1) * 2) + 1] + tmp_local_acc[1]

                else:
                    dynamic_input[i * 2] = 2 * dynamic_input[(i - 1) * 2]  - dynamic_input[(i - 2) * 2]+ tmp_local_acc[0]
                    dynamic_input[(i * 2) + 1] = 2 * dynamic_input[((i - 1) * 2) + 1] - dynamic_input[((i - 2) * 2) + 1] + tmp_local_acc[1]


        return dynamic_input



class LPNET_V03(nn.Module):
    def __init__(self, rnn_output_dim, bp_dim, path_length, device):
        super(LPNET_V03, self).__init__()

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.lane_coef = None
        self.device = device
        self.rnn_output_dim = rnn_output_dim  # 100
        self.dynamic_input_dim = path_length
        self.BP_DIM = bp_dim
        self.F0_DIM = 512
        self.F1_DIM = 256
        self.F2_DIM = 128
        self.ReLU = nn.ReLU().to(self.device)
        self.IE_CHANNEL = 8

        self.resnet18 = models.resnet18(pretrained=True).to(self.device)
        self.resnet18 = torch.nn.Sequential(*(list(self.resnet18.children())[:-1]))
        # Unremark below with 8 channel input (IE)
        self.resnet18[0] = nn.Conv2d(self.IE_CHANNEL, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                     bias=False).to(device)

        self.F0_NET = nn.Linear(self.F0_DIM + self.BP_DIM, self.F1_DIM).to(self.device)
        self.F1_NET = nn.Linear(self.F1_DIM + self.BP_DIM, self.F2_DIM).to(self.device)

        # self.GRU = nn.GRU(self.F2_DIM + self.BP_DIM + self.dynamic_input_dim, self.rnn_output_dim, 1).to(self.device)

        self.GRU = nn.GRU(self.dynamic_input_dim, self.rnn_output_dim, 1).to(self.device)

        self.mlp1_for_rnn_output = nn.Linear(self.rnn_output_dim + self.BP_DIM + self.F2_DIM,
                                             int(self.rnn_output_dim / 2)).to(
            self.device)
        self.mlp2_for_rnn_output = nn.Linear(int(self.rnn_output_dim / 2), 2).to(self.device)  # generate point

    def forward(self, information_embedder, BP):

        BP = torch.tensor(BP).to(self.device).float()
        dynamic_input = np.zeros(self.dynamic_input_dim)
        dynamic_input = torch.tensor(dynamic_input).to(self.device).float()

        # ResNet18
        prediction = self.resnet18(information_embedder)
        prediction = torch.reshape(prediction, (prediction.shape[1],))

        # F0 -> F1
        prediction = self.F0_NET(torch.cat((prediction, BP), 0))
        prediction = self.ReLU(prediction)

        # F1 -> F2
        prediction = self.F1_NET(torch.cat((prediction, BP), 0))
        prediction = self.ReLU(prediction)

        lane_loss = 0

        # RNN
        with torch.autograd.set_detect_anomaly(True):
            for i in range(int(dynamic_input.shape[0] / 2)):
                dynamic_input_reshape = torch.reshape(dynamic_input, (1, 1, dynamic_input.shape[0])).clone()

                rnn_output, hidden = self.GRU(dynamic_input_reshape)

                rnn_output = torch.squeeze(rnn_output)

                concat = torch.cat((prediction, BP, rnn_output), 0)

                rnn_output_encoded = self.mlp1_for_rnn_output(concat)
                tmp_local_acc = self.mlp2_for_rnn_output(rnn_output_encoded)

                # print("tmp_local_path", tmp_local_path)
                if i == 0:
                    dynamic_input[i * 2] = tmp_local_acc[0]
                    dynamic_input[(i * 2) + 1] = tmp_local_acc[1]

                elif i == 1:
                    dynamic_input[i * 2] = 2 * dynamic_input[(i-1) * 2] + tmp_local_acc[0]
                    dynamic_input[(i * 2) + 1] = 2 * dynamic_input[((i-1) * 2) + 1] + tmp_local_acc[1]

                else:
                    dynamic_input[i * 2] = 2 * dynamic_input[(i - 1) * 2]  - dynamic_input[(i - 2) * 2]+ tmp_local_acc[0]
                    dynamic_input[(i * 2) + 1] = 2 * dynamic_input[((i - 1) * 2) + 1] - dynamic_input[((i - 2) * 2) + 1] + tmp_local_acc[1]


        return dynamic_input



class LPNET_MLP05(nn.Module):
    def __init__(self, rnn_output_dim, path_length, device):
        super(LPNET_MLP05, self).__init__()

        self.device = device
        self.rnn_output_dim = rnn_output_dim  # 2
        self.dynamic_input_dim = path_length  # 10
        # self.BP_DIM = bp_dim
        self.F0_DIM = 256
        self.F1_DIM = 128
        self.F2_DIM = 64

        self.ReLU = nn.ReLU().to(self.device)
        self.IE_CHANNEL = 3

        self.CNN1 = nn.Conv2d(self.IE_CHANNEL, self.IE_CHANNEL, 5, stride=2).to(self.device)
        self.CNN2 = nn.Conv2d(self.IE_CHANNEL, self.IE_CHANNEL, 5, stride=2).to(self.device)
        self.CNN3 = nn.Conv2d(self.IE_CHANNEL, 1, 5, stride=2).to(self.device)
        self.MaxPool = nn.MaxPool2d(3, 1).to(self.device)

        self.F0_NET = nn.Linear(self.F0_DIM, self.F1_DIM).to(self.device)
        self.F1_NET = nn.Linear(self.F1_DIM, self.F2_DIM).to(self.device)
        self.F2_NET = nn.Linear(self.F2_DIM, self.dynamic_input_dim).to(self.device)


    def forward(self, information_embedder):

        # plt.imshow(information_embedder.clone().cpu().detach().numpy()[0, 0, :, :])
        # plt.show()

        plt.imshow(transforms.ToPILImage()(information_embedder.clone().cpu()[0,0 , :, :]))
        plt.show()

        prediction = self.CNN1(information_embedder)
        prediction = self.ReLU(prediction)
        prediction = self.MaxPool(prediction)

        prediction = self.CNN2(prediction)
        prediction = self.ReLU(prediction)
        prediction = self.MaxPool(prediction)

        prediction = self.CNN3(prediction)
        prediction = self.ReLU(prediction)
        prediction = self.MaxPool(prediction)

        self.F0_DIM = prediction.shape[0] * prediction.shape[1] * prediction.shape[2] * prediction.shape[3]
        prediction = prediction.view(-1, self.F0_DIM)  # reshape Variable

        # F0 -> F1
        prediction = self.F0_NET(prediction)
        prediction = self.ReLU(prediction)

        prediction = self.F1_NET(prediction)
        prediction = self.ReLU(prediction)

        prediction = self.F2_NET(prediction)
        prediction = self.ReLU(prediction)

        return prediction




class LPNET(nn.Module):
    def __init__(self, rnn_output_dim, bp_dim, path_length, device):
        super(LPNET, self).__init__()

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.device = device
        self.rnn_output_dim = rnn_output_dim  # 2
        self.dynamic_input_dim = path_length  # 30
        self.BP_DIM = bp_dim
        self.F0_DIM = 512
        self.F1_DIM = 256
        self.F2_DIM = 128
        self.ReLU = nn.ReLU().to(self.device)
        self.IE_CHANNEL = 8

        self.resnet18 = models.resnet18(pretrained=True).to(self.device)
        self.resnet18 = torch.nn.Sequential(*(list(self.resnet18.children())[:-1]))
        # Un-remark below with 8 channel input (IE)
        self.resnet18[0] = nn.Conv2d(self.IE_CHANNEL, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(device)

        self.F0_NET = nn.Linear(self.F0_DIM + self.BP_DIM, self.F1_DIM).to(self.device)
        self.F1_NET = nn.Linear(self.F1_DIM + self.BP_DIM, self.F2_DIM).to(self.device)
        # self.LSTM = nn.LSTM(self.F2_DIM + self.BP_DIM, self.rnn_output_dim, 2) .to(self.device)
        self.GRU = nn.GRU(self.F2_DIM + self.BP_DIM + self.dynamic_input_dim, self.rnn_output_dim, 1).to(self.device)

    def forward(self, information_embedder, BP):

        BP = torch.tensor(BP).to(self.device).float()
        dynamic_input = np.zeros(self.dynamic_input_dim)
        dynamic_input = torch.tensor(dynamic_input).to(self.device).float()

        # ResNet18
        prediction = self.resnet18(information_embedder)
        prediction = torch.reshape(prediction, (prediction.shape[1],))

        # F0 -> F1
        prediction = self.F0_NET(torch.cat((prediction, BP), 0))
        prediction = self.ReLU(prediction)

        # F1 -> F2
        prediction = self.F1_NET(torch.cat((prediction, BP), 0))
        prediction = self.ReLU(prediction)

        # RNN
        for i in range(int(dynamic_input.shape[0]/2)):
            concat = torch.cat((prediction, BP, dynamic_input), 0)
            concat = torch.reshape(concat, (1, 1, concat.shape[0]))

            tmp_local_path, hidden = self.GRU(concat)

            tmp_local_path = torch.squeeze(tmp_local_path)

            # print("tmp_local_path", tmp_local_path)

            dynamic_input[i * 2] = tmp_local_path[0]
            dynamic_input[(i * 2) + 1] = tmp_local_path[1]


        return dynamic_input


class LPNET_MLP(nn.Module):
    def __init__(self, rnn_output_dim, bp_dim, path_length, device):
        super(LPNET_MLP, self).__init__()

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.device = device
        self.rnn_output_dim = rnn_output_dim  # 2
        self.dynamic_input_dim = path_length  # 30
        self.BP_DIM = bp_dim
        self.F0_DIM = 512
        self.F1_DIM = 256
        self.F2_DIM = 128
        self.ReLU = nn.ReLU().to(self.device)
        self.IE_CHANNEL = 8
        self.MLP = nn.Sequential(
            nn.Linear(self.F2_DIM, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, path_length)
        ).to(device)

        self.resnet18 = models.resnet18(pretrained=True).to(self.device)
        self.resnet18 = torch.nn.Sequential(*(list(self.resnet18.children())[:-1]))
        # Unremark below with 8 channel input (IE)
        self.resnet18[0] = nn.Conv2d(self.IE_CHANNEL, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(device)

        self.F0_NET = nn.Linear(self.F0_DIM + self.BP_DIM, self.F1_DIM).to(self.device)
        self.F1_NET = nn.Linear(self.F1_DIM + self.BP_DIM, self.F2_DIM).to(self.device)
        # self.LSTM = nn.LSTM(self.F2_DIM + self.BP_DIM, self.rnn_output_dim, 2) .to(self.device)
        self.GRU = nn.GRU(self.F2_DIM + self.BP_DIM + self.dynamic_input_dim, self.rnn_output_dim, 1).to(self.device)

    def forward(self, information_embedder, BP):

        BP = torch.tensor(BP).to(self.device).float()
        dynamic_input = np.zeros(self.dynamic_input_dim)
        dynamic_input = torch.tensor(dynamic_input).to(self.device).float()

        # ResNet18
        prediction = self.resnet18(information_embedder)
        prediction = torch.reshape(prediction, (prediction.shape[1],))

        # F0 -> F1
        prediction = self.F0_NET(torch.cat((prediction, BP), 0))
        prediction = self.ReLU(prediction)

        # F1 -> F2
        prediction = self.F1_NET(torch.cat((prediction, BP), 0))
        prediction = self.ReLU(prediction)

        output = self.MLP(prediction)

        return output


class LPNET_R2P2(nn.Module):
    def __init__(self, rnn_output_dim, bp_dim, path_length, device):
        super(LPNET_R2P2, self).__init__()

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.device = device
        self.rnn_output_dim = rnn_output_dim  # 100
        self.dynamic_input_dim = path_length  # 30
        self.BP_DIM = bp_dim
        self.F0_DIM = 512
        self.F1_DIM = 256
        self.F2_DIM = 128
        self.ReLU = nn.ReLU().to(self.device)
        self.IE_CHANNEL = 8

        self.resnet18 = models.resnet18(pretrained=True).to(self.device)
        self.resnet18 = torch.nn.Sequential(*(list(self.resnet18.children())[:-1]))
        # Unremark below with 8 channel input (IE)
        self.resnet18[0] = nn.Conv2d(self.IE_CHANNEL, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(device)

        self.F0_NET = nn.Linear(self.F0_DIM + self.BP_DIM, self.F1_DIM).to(self.device)
        self.F1_NET = nn.Linear(self.F1_DIM + self.BP_DIM, self.F2_DIM).to(self.device)

        # self.GRU = nn.GRU(self.F2_DIM + self.BP_DIM + self.dynamic_input_dim, self.rnn_output_dim, 1).to(self.device)

        self.GRU = nn.GRU(self.dynamic_input_dim, self.rnn_output_dim, 1).to(self.device)

        self.mlp1_for_rnn_output = nn.Linear(self.rnn_output_dim + self.BP_DIM + self.F2_DIM, int(self.rnn_output_dim / 2)).to(
            self.device)
        self.mlp2_for_rnn_output = nn.Linear(int(self.rnn_output_dim / 2), 2).to(self.device)  # generate point

    def forward(self, information_embedder, BP):
        BP = torch.tensor(BP).to(self.device).float()
        dynamic_input = np.zeros(self.dynamic_input_dim)
        dynamic_input = torch.tensor(dynamic_input).to(self.device).float()

        # ResNet18
        prediction = self.resnet18(information_embedder)
        prediction = torch.reshape(prediction, (prediction.shape[1],))

        # F0 -> F1
        prediction = self.F0_NET(torch.cat((prediction, BP), 0))
        prediction = self.ReLU(prediction)

        # F1 -> F2
        prediction = self.F1_NET(torch.cat((prediction, BP), 0))
        prediction = self.ReLU(prediction)

        # RNN
        with torch.autograd.set_detect_anomaly(True):
            for i in range(int(dynamic_input.shape[0] / 2)):
                dynamic_input_reshape = torch.reshape(dynamic_input, (1, 1, dynamic_input.shape[0])).clone()

                rnn_output, hidden = self.GRU(dynamic_input_reshape)

                rnn_output = torch.squeeze(rnn_output)

                concat = torch.cat((prediction, BP, rnn_output), 0)

                rnn_output_encoded = self.mlp1_for_rnn_output(concat)
                tmp_local_path = self.mlp2_for_rnn_output(rnn_output_encoded)

                # print("tmp_local_path", tmp_local_path)

                dynamic_input[i * 2] = tmp_local_path[0]
                dynamic_input[(i * 2) + 1] = tmp_local_path[1]

                # print("dynamic_input", dynamic_input)

        return dynamic_input