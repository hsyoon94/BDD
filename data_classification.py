from os import listdir
from os.path import isfile, join
import json
import numpy as np
import time
from datetime import datetime
import os
import shutil

data_raw_dir  = '/mnt/sda2/BDD/data'

for i in range(7):
    if os.path.exists(data_raw_dir + '/' + str(i)) is False:
        os.mkdir(data_raw_dir + '/' + str(i))


data_name_list = [f for f in listdir(data_raw_dir) if isfile(join(data_raw_dir, f))]

bp0 = 0
bp1 = 0
bp2 = 0
bp3 = 0
bp4 = 0
bp5 = 0
bp6 = 0

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

    if j%100 == 0:
        print(json_myfile['data']['behavior'])

    if json_myfile['data']['behavior'][0] == 1:
        shutil.move(data_raw_dir + '/' + data_name_list[j], data_raw_dir + '/0/' + data_name_list[j])
        bp0 = bp0 + 1
    elif json_myfile['data']['behavior'][1] == 1:
        shutil.move(data_raw_dir + '/' + data_name_list[j], data_raw_dir + '/1/' + data_name_list[j])
        bp1 = bp1 + 1
    elif json_myfile['data']['behavior'][2] == 1:
        shutil.move(data_raw_dir + '/' + data_name_list[j], data_raw_dir + '/2/' + data_name_list[j])
        bp2 = bp2 + 1
    elif json_myfile['data']['behavior'][3] == 1:
        shutil.move(data_raw_dir + '/' + data_name_list[j], data_raw_dir + '/3/' + data_name_list[j])
        bp3 = bp3 + 1
    elif json_myfile['data']['behavior'][4] == 1:
        shutil.move(data_raw_dir + '/' + data_name_list[j], data_raw_dir + '/4/' + data_name_list[j])
        bp4 = bp4 + 1
    elif json_myfile['data']['behavior'][5] == 1:
        shutil.move(data_raw_dir + '/' + data_name_list[j], data_raw_dir + '/5/' + data_name_list[j])
        bp5 = bp5 + 1
    elif json_myfile['data']['behavior'][6] == 1:
        shutil.move(data_raw_dir + '/' + data_name_list[j], data_raw_dir + '/6/' + data_name_list[j])
        bp6 = bp6 + 1



print("bp0", bp0)
print("bp1", bp1)
print("bp2", bp2)
print("bp3", bp3)
print("bp4", bp4)
print("bp5", bp5)
print("bp6", bp6)
