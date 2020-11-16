import numpy as np

from os import listdir
from os.path import isfile, join
import json
import numpy as np
import time
from datetime import datetime
import os
import shutil

data_raw_dir  = '/mnt/sda2/BDD/data/0'

for i in range(7):
    if os.path.exists(data_raw_dir + '/' + str(i) + '_backup') is False:
        os.mkdir(data_raw_dir + '/' + str(i) + '_backup')

data_name_list = [f for f in listdir(data_raw_dir) if isfile(join(data_raw_dir, f))]

for i in range(len(data_name_list)):

    try:
        with open(data_raw_dir + '/' + data_name_list[j]) as myfile:
            json_myfile = json.load(myfile)
    except ValueError as val_e:
        print("VALUE ERROR", val_e)
        continue
    except IOError as io_e:
        print("IO ERROR", io_e)
        continue

    if i%2 == 0:
        shutil.move(data_raw_dir + '/' + data_name_list[j], data_raw_dir + '_backup/' + data_name_list[j])


