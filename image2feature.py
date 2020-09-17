import numpy as np
import torch
import os

# ResNet 18 model
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)

# directory
dir = './expert_demo/', 'r'

for root, subdirs, files in os.walk(dir):
    for subdir in subdirs:
        for file in files:
            img = torch.load(root + subdir + '/' + file)
            feature = model(img)