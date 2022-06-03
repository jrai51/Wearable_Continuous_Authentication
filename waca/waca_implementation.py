#############
## IMPORTS ##
#############

import pandas as pd
import os.path
import numpy as np
import matplotlib
from scipy import signal

###################
## PREPROCESSING ##
###################
## Set the participant we're analyzing

PARTICIPANT = "user1"
DATA_1 = PARTICIPANT+'_1.csv'
DATA_2 = PARTICIPANT+'_2.csv'
DATA_3 = PARTICIPANT+'_3.csv'

#loading data in
file_paths = ['user_data\\'+PARTICIPANT+'\\'+DATA_1, 'user_data\\'+PARTICIPANT+'\\'+DATA_2, 'user_data\\'+PARTICIPANT+'\\'+DATA_1 ]
f = file_paths[0]
gen = pd.read_csv(f, names=['Sensor', 'Time', 'X', 'Y', 'Z'], on_bad_lines='skip') #general data

#accel is 10, gyro is 4
GYRO_ID = 4
ACCEL_ID = 10

#Separate data by sensor id 
gyro_df = gen[gen.Sensor == GYRO_ID]
accel_df = gen[gen.Sensor == ACCEL_ID]


#sort data by time 
gyro_df.sort_values('Time', inplace=True)
accel_df.sort_values('Time', inplace=True)

#data is now separated and ordered by time, ready for M-point filer
print(gyro_df.head())
print(accel_df.head())
