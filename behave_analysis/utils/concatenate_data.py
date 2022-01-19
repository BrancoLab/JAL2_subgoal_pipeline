
''' videos
create a txt file:
file 'path/to/file1.wav'
file 'path/to/file2.wav'
file 'path/to/file3.wav'

ffmpeg -f concat -safe 0 -i vids.txt -c copy cam3.avi
'''

''' bin data '''
from glob import glob
import os
import numpy as np
import dill as pickle

file_path = "D:\\Dropbox (UCL)\\DAQ\\upstairs_rig\\21APR01_9753_block pre evs"
file_path = "D:\\Dropbox (UCL)\\DAQ\\upstairs_rig\\21APR12_9756_allow then block"

# LASER BIN FILES
laser_files = glob(os.path.join(file_path, "laser*"))
laser_data1 = np.fromfile(laser_files[0])
laser_data2 = np.fromfile(laser_files[1])

laser_data3 = np.concatenate((laser_data1,laser_data2))

with open(os.path.join(file_path, 'laser_fire3'), "wb") as dill_file: pickle.dump(laser_data3, dill_file)
with open(os.path.join(file_path, 'laser_fire3'), "rb") as dill_file: laser_data4 = pickle.load(dill_file)

# ANALOG BIN FILES
analog_files = glob(os.path.join(file_path, "analog*"))
analog_data1 = np.fromfile(analog_files[0])
analog_data2 = np.fromfile(analog_files[1])

analog_data3 = np.concatenate((analog_data1,analog_data2))

with open(os.path.join(file_path, 'analog3'), "wb") as dill_file: pickle.dump(analog_data3, dill_file)
with open(os.path.join(file_path, 'analog3'), "rb") as dill_file: analog_data4 = pickle.load(dill_file)
