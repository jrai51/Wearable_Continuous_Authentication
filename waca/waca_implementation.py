#############
## IMPORTS ##
#############

import pandas as pd
import os.path
import numpy as np
import matplotlib
import scipy
from scipy.fft import fft


#initializing variables
GYRO_ID = 4
ACCEL_ID = 10

gyro_df = None
accel_df = None
features_df = None

def load_data(user='user1', test='1'):
    ###################
    ## PREPROCESSING ##
    ###################
    ## Set the participant we're analyzing
    PARTICIPANT = user
    DATA_1 = PARTICIPANT+'_1.csv'
    DATA_2 = PARTICIPANT+'_2.csv'
    DATA_3 = PARTICIPANT+'_3.csv'

    #loading data in

    file_paths = ['waca\\user_data\\'+PARTICIPANT+'\\'+DATA_1, 'waca\\user_data\\'+PARTICIPANT+'\\'+DATA_2, 'waca\\user_data\\'+PARTICIPANT+'\\'+DATA_3]


    f = file_paths[test-1]
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

load_data('user1', 1)

def filter(M):
    ''' Applies M-point moving average filter'''

    ##################################
    # APPLYING MOVING AVERAGE FILTER #
    ##################################

    #Applying Moving Average Filter 

    #starts averaging at Mth point rather than 0th as opposed to vice versa in original paper 
    gyro_df['MA_Time'] = gyro_df['Time'].rolling(M).mean()
    gyro_df['MA_X'] = gyro_df['X'].rolling(M).mean()
    gyro_df['MA_Y'] = gyro_df['Y'].rolling(M).mean()
    gyro_df['MA_Z'] = gyro_df['Z'].rolling(M).mean()

    accel_df['MA_Time'] = accel_df['Time'].rolling(M).mean()
    accel_df['MA_X'] = accel_df['X'].rolling(M).mean()
    accel_df['MA_Y'] = accel_df['Y'].rolling(M).mean()
    accel_df['MA_Z'] = accel_df['Z'].rolling(M).mean()

    #Creating axis vectors 

    N = 1500 #number of samples for a profile feature 
    M_IDX = N + M -1#index of the Nth sample (accounts for NaNs of first M rows)


    #X axis 
    x_a = accel_df.loc[:, "MA_X"]
    x_a = list(x_a[M-1:M_IDX]) #this is done to avoid a keyerror in the loc function 

    x_g = gyro_df.loc[:, "MA_X"]
    x_g = list(x_g[M-1:M_IDX])  

    #y axis
    y_a = accel_df.loc[:, "MA_Y"]
    y_a = list(y_a[M-1:M_IDX])

    y_g = gyro_df.loc[:, "MA_Y"]
    y_g = list(y_g[M-1:M_IDX]) 

    #z axis
    z_a = accel_df.loc[:, "MA_Z"]
    z_a = list(z_a[M-1:M_IDX])

    z_g = accel_df.loc[:, "MA_Z"]
    z_g = list(z_g[M-1:M_IDX])

    features = {'x_a': x_a, 'y_a': y_a, 'z_a': z_a, 'x_g': x_g, 'y_g': y_g, 'z_g': z_g}
    features_df = pd.DataFrame.from_dict(features) #mostly for presentation purposes, will come in handy for feature extraction

    #the below may be necessary calculations (averaged time start/end)
    #start_time = gyro_df.MA_Time.loc[gyro_df.MA_Time.first_valid_index()]
    #end_time = gyro_df.MA_Time.iloc[-1]

    print(features_df)

    #preprocessing is now completed

filter(10)
