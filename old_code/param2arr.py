import torch
import numpy as np
import os

param_file_dir = './model_param/' + '200723_2024' + '/'

for root, subdirs, files in os.walk(param_file_dir):
    for file in files:
        with open(os.path.join(param_file_dir, file)) as f:

            lines = f.readlines()

            for line in lines:
                if file == 'layers.0.biax.txt':



