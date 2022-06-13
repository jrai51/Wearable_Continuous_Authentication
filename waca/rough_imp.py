#############
## IMPORTS ##
#############


import pandas as pd
import os.path
import numpy as np
import matplotlib
import scipy
from scipy.fft import fft
import pprint
pp = pprint.PrettyPrinter(indent=4)

def waca(user='user1', test='1'):
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

    ##################################
    # APPLYING MOVING AVERAGE FILTER #
    ##################################

    #Applying Moving Average Filter 


    M = 10 #M-point filter

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

    N = 1000 #number of samples for a profile feature 
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


    #preprocessing is now completed

    ###########################################
    ## FEATURE EXTRACTION and USER PROFILING ##
    ###########################################


    #Mean, Median, Variance, Average Absolute
    #Difference of Peaks, Range, Mode, Covariance,
    #Mewan Absolute Deviation (MAD), Inter-
    #quartile Range (IQR), correlation between axes
    #(xy, yz, xz), Skewness, Kurtosis


    def spectral_energy(X):
        ''' X is a list of FFT data '''
        sum = 0
        for item in X:
            sum += abs(item) **2 #is it supposed to be the square of each item or the square of the total sum?
        return sum / len(X)

    def shannon_entropy(label):
        vc = pd.Series(label).value_counts(normalize=True, sort=False)
        base = 2
        return -(vc * np.log(vc)/np.log(base)).sum()

    extracted_features = { 'mean': [], 'median': [], 'variance': [], 'AADP': [], 'range': [], 'mode':[], 
                    'covariance': [], 'mad': [], 'iqr': [], 'correlation': [], 'skewness': [], 'kurtosis': [],
                    'entropy': [], 's_nrg': []} #features in the domain of frequency

    for column in features_df:
        extracted_features['mean'].append(features_df[column].mean())
        extracted_features['median'].append(features_df[column].median())
        extracted_features['variance'].append(features_df.var()[column])
        extracted_features['range'].append(features_df[column].max() - features_df[column].min())
        extracted_features['mode'].append(features_df[column].mode().iat[0])
        extracted_features['iqr'].append( features_df[column].quantile(0.75) - features_df[column].quantile(0.25))
        extracted_features['skewness'].append(features_df[column].skew())
        extracted_features['kurtosis'].append(features_df[column].kurtosis())
        extracted_features['mad'].append(features_df[column].mad()) 
        #calculate the FFT for next calculations
        col_fft = fft(features_df[column].to_numpy())
        extracted_features['entropy'].append(shannon_entropy(column))  
        extracted_features['s_nrg'].append(spectral_energy(col_fft)) #what is up with the +0.00j??? 


        #need fixing :
        extracted_features['AADP'].append(0) ######################FIX
        

    labels = ['x_a', 'y_a', 'z_a', 'x_g', 'y_g', 'z_g']

    #could have done a nested loop here but whatever -- can change if necessary
    extracted_features['covariance'].append(features_df['x_a'].cov(features_df['y_a']))
    extracted_features['covariance'].append(features_df['x_a'].cov(features_df['z_a']))
    extracted_features['covariance'].append(features_df['y_a'].cov(features_df['z_a']))
    extracted_features['covariance'].append(features_df['x_g'].cov(features_df['y_g']))
    extracted_features['covariance'].append(features_df['x_g'].cov(features_df['z_g']))
    extracted_features['covariance'].append(features_df['y_g'].cov(features_df['z_g']))

    extracted_features['correlation'].append(features_df['x_a'].corr(features_df['y_a']))
    extracted_features['correlation'].append(features_df['x_a'].corr(features_df['z_a']))
    extracted_features['correlation'].append(features_df['y_a'].corr(features_df['z_a']))
    extracted_features['correlation'].append(features_df['x_g'].corr(features_df['y_g']))
    extracted_features['correlation'].append(features_df['x_g'].corr(features_df['z_g']))
    extracted_features['correlation'].append(features_df['y_g'].corr(features_df['z_g']))

    feature_set = pd.DataFrame.from_dict(extracted_features, orient='index', columns=labels)

    print(feature_set)

    user_id = PARTICIPANT
    t_start = gyro_df.Time.loc[gyro_df.Time.first_valid_index()]
    t_end = gyro_df.Time.iloc[-1]
    f_vec = feature_set.unstack().to_frame().sort_index(level=1).T
    f_vec.columns = f_vec.columns.map('_'.join)
    pp.pprint('F_VEC: '+str(f_vec.iloc[0].tolist()))
    normal_vec = normalize(np.array(f_vec.iloc[0].tolist()))

    user_profile = [user_id, t_start, t_end, normal_vec]
    #print(user_profile)
    
    return user_profile

def normalize(V):
    '''Returns linear normalization of a vector A'''
    normal_vec = []
    x_max = max(V)
    x_min = min(V)
    for x in V:
        x_new = (x - x_min) / (x_max - x_min)
        normal_vec.append(x_new)
    return normal_vec 


users = [] #For now, this array works as the 'database' to store user profiles to test against each other 


user_1 = waca('user1', 1)
user_2 = waca('user2', 1)
user_1_2 = waca('user1', 2)

users.append(user_1)
users.append(user_2)
users.append(user_1_2)

###################
# DECISION MODULE #
###################

threat_threshold = 0.05

print()

def minkow_dist(x, y, p=2):
    #takes two vectors stored as lists to return minkowski distance. 
    p = 2  #measurement for minkowski distance, euclidean distance when set to 2 
    distance_sum = 0   
    for i in range(0, len(x)):
        distance_sum += (x[i] - y[i])** p

    return distance_sum ** (1/p)


pp.pprint('LIST:'+str( user_1[3]))


dist = minkow_dist(user_1[3], user_1_2[3], 2)

print('DISTANCES:\n', dist)
#manhattan_dist = scipy.cityblock()

print()

if dist < threat_threshold:
    print('VERIFIED')
else:
    print('UNVERIFIED')