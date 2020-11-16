import numpy as np
import torch
import os
from shutil import copyfile
import math

# ResNet 18 model

# directory
expert_demo_dir = '/media/hsyoon/HS_VILAB1/BDD/expert_demo/'
dataset_dir = '/media/hsyoon/HS_VILAB1/BDD/network_dataset/'
path_file_dir = ''
path_file = ''
local_path = np.array([])
PATH_INTERVAL = 5
PATH_FREQUENCY = 5
data_count = 0


for root, subdirs, files in os.walk(expert_demo_dir):
    for subdir in subdirs:
        for root_tmp, subdirs_tmp, files_tmp in os.walk(root + subdir):
            for file_tmp in files_tmp:



                print(root_tmp, file_tmp)
    #             if '.pt' in file_tmp:
    #                 path_file_dir = os.path.join(root_tmp, file_tmp)
    #                 path_file = torch.load(path_file_dir)
    #                 path_file = path_file['states'].cpu().numpy()
    #                 path_file = np.squeeze(path_file)
    #
    #                 print(path_file)
    #
    #         for file_tmp in files_tmp:
    #             if '.png' in file_tmp:
    #                 file_num = int(file_tmp.split('.')[0])
    #                 for i in range(PATH_INTERVAL):
    #
    #                     path_index = min(file_num + (i + 1) * PATH_FREQUENCY, path_file.shape[0] - 1)
    #
    #                     x1 = float(path_file[file_num][0])
    #                     y1 = float(path_file[file_num][1])
    #
    #                     x2 = float(path_file[path_index][0])
    #                     y2 = float(path_file[path_index][1])
    #
    #                     cos_theta = x1 / math.sqrt(math.pow(x1, 2) + math.pow(y1, 2))
    #                     sin_theta = y1 / math.sqrt(math.pow(x1, 2) + math.pow(y1, 2))
    #
    #                     local_x = (x2 - x1) * cos_theta + (y2 - y1) * sin_theta
    #                     local_y = (x1 - x2) * sin_theta + (y2 - y1) * cos_theta
    #
    #                     local_path = np.append(local_path, [local_x, local_y])
    #
    #                 # sem_img, local path ready!
    #                 # copy image to destination dir with data_count index
    #
    #                 # write path
    #                 try:
    #                     copyfile(os.path.join(root_tmp, file_tmp), os.path.join(dataset_dir, 'sem_img/', 'img' + str(data_count) + '.png'))
    #                 except:
    #                     pass
    #                 np.savetxt(os.path.join(dataset_dir, 'path/', 'path' + str(data_count) + '.txt'), local_path)
    #
    #                 # local_path initialize
    #                 local_path = np.array([])
    #                 data_count = data_count + 1
    #
    #         print('one file dataset save complete!')
    #     print('one subfile save complet!')
    # print('hello~')













