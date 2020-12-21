from os import listdir
from os.path import isfile, join
import json
import numpy as np
import time
from datetime import datetime
import os
import shutil

data_raw_dir  = '/mnt/sda2/BDD/data/0'
data_target_dir = '/mnt/sda2/BDD/data_removal/0'
PROBABILITY_THRESHOLD = 0.1

for i in range(7):
    if os.path.exists(data_raw_dir + '/' + str(i)) is False:
        os.mkdir(data_raw_dir + '/' + str(i))


data_name_list = [f for f in listdir(data_raw_dir) if isfile(join(data_raw_dir, f))]

out_count = 0
print(len(data_name_list))

for j in range(len(data_name_list)):

    try:
        with open(data_raw_dir + '/' + data_name_list[j]) as myfile:
            json_myfile = json.load(myfile)
    except ValueError as val_e:
        print("VALUE ERROR", val_e)
        continue
    except IOError as io_e:
        print("IO ERROR", io_e)
        continue

    probability = np.random.uniform(0, 1, 1)

    if probability <= PROBABILITY_THRESHOLD:
        shutil.move(data_raw_dir + '/' + data_name_list[j], data_target_dir + '/' + data_name_list[j])
        print(j,"th FILE with", len(data_name_list), "total length moved with probability", probability)
        out_count = out_count + 1


print(out_count, "DATA REMOVED")