#############
## IMPORTS ##
#############

import pandas as pd
import os.path
import numpy as np
import matplotlib
import scipy
from scipy.fft import fft
from scipy import signal
import pprint 

pp = pprint.PrettyPrinter()

#pd.options.mode.chained_assignment = None  # default='warn'

class UserProfile():
    def __init__(self, userID, start_time, end_time, f_vec):
        self.userID = userID
        self.start_time = start_time
        self.end_time = end_time
        self.f_vec = f_vec
    
    def __str__(self):
        return self.userID

    def __repr__(self):
        return str(self.userID) +','+str(self.f_vec)

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
    gen = gen.dropna()

    #accel is 10, gyro is 4
    GYRO_ID = 4
    ACCEL_ID = 10

    #Separate data by sensor id 
    gyro_df = gen.loc[gen.Sensor == GYRO_ID]
    accel_df = gen.loc[gen.Sensor == ACCEL_ID]
    

    #sort data by time 
    gyro_df = gyro_df.sort_values('Time')
    accel_df = accel_df.sort_values('Time')
    

    #data is now separated and ordered by time, ready for M-point filer
    
    #print(accel_df.head())

    ##################################
    # APPLYING MOVING AVERAGE FILTER #
    ##################################

    #Applying Moving Average Filter 


    M = 9 #M-point filter

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

    z_g = gyro_df.loc[:, "MA_Z"]
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
        vc = label.value_counts(normalize=True, sort=False)
        base = 2
        return -(vc * np.log(vc)/np.log(base)).sum()

    def aadp(X):
        ''' Calculates average absolute difference of peaks of a list X'''

        #calculates the difference between each adjacent pair
        peaks = scipy.signal.find_peaks(X)
        peaks = list(peaks[0])
        try:
            sum = 0
            n = 0 #number of differences calculated 
            for i in range(1, len(peaks)):
                sum += abs(peaks[i]-peaks[i-1])
                n += 1

            return sum/ n
        except:
            return 0

    extracted_features = { 'mean': [], 'median': [], 'variance': [], 'AADP': [], 'range': [], 'mode':[], 
                    'covariance': [], 'mad': [], 'iqr': [], 'correlation': [], 'skewness': [], 'kurtosis': [],
                    'entropy': [], 's_nrg': []} #features in the domain of frequency

    for column in features_df:
        
        extracted_features['AADP'].append(aadp(features_df[column].tolist()))
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
        
        extracted_features['entropy'].append(shannon_entropy(features_df[column]))  
        extracted_features['s_nrg'].append(spectral_energy(col_fft)) #what is up with the +0.00j??? 
        #extracted_features['AADP'].append(0)
    

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
    #print(feature_set)

    user_id = PARTICIPANT
    t_start = gyro_df.Time.loc[gyro_df.Time.first_valid_index()]
    t_end = gyro_df.Time.iloc[-1]
    f_vec = feature_set.unstack().to_frame().sort_index(level=1).T
    f_vec.columns = f_vec.columns.map('_'.join)
    

    normal_vec = normalize(np.array(f_vec.iloc[0].tolist()))

    user_profile = UserProfile(user_id, t_start, t_end, normal_vec)
    
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

#waca('user7', 1)


users = {'test1': [], 'test2': [], 'test3': []} #For now, this array works as the 'database' to store user profiles to test against each other 

def get_users():
    directory = "C:\\Users\\Jagrit Rai\\Documents\\GitHub\\Wearable_Continuous_Authentication\\waca\\user_data"
    
    for root, subdirectories, files in os.walk(directory):
        for subdirectory in subdirectories:
            try:
                users['test1'].append(waca(subdirectory, 1))
                users['test2'].append(waca(subdirectory, 2))
                users['test3'].append(waca(subdirectory, 3))
            except:
                pass
  

get_users()

#database = pd.DataFrame.from_dict(users)

print(len(users['test1']))
print(len(users['test2']))
print(len(users['test3']))

###########
# MAKE IT INTO A DATAFRAME
##########


###################
# DECISION MODULE #
###################

class UserDist:
    def __init__(self, user_a, user_b, dist):
        self.user_a = user_a
        self.user_b = user_b
        self.dist = dist 
    
    def __repr__(self):
        strings = [self.user_a, self.user_b, str(self.dist)]
        return ','.join(strings)


threat_threshold = 0.05

def minkow_dist(x, y, p=2):
    #takes two vectors stored as lists to return minkowski distance. 
    p = 2  #measurement for minkowski distance, euclidean distance when set to 2 
    distance_sum = 0   
    for i in range(0, len(x)):
        distance_sum += (x[i] - y[i])** p

    return distance_sum ** (1/p)

user1 = users['test1'][1]
user2 = users['test2'][1]

dist = minkow_dist(user1.f_vec, user2.f_vec, 2)
manhattan_dist = scipy.spatial.distance.cityblock(user1.f_vec, user2.f_vec)

measure = dist

#populate dissimilarity matrix with all distances between users from test 1
matrix = []
for user in users['test1']:
    dists = []
    for next_user in users['test1']:
        tup = UserDist(user.userID, next_user.userID, round(minkow_dist(user.f_vec, next_user.f_vec), 3))
        dists.append(tup)
    matrix.append(dists)

matrix_df = pd.DataFrame(matrix)
print(matrix_df)



#legend
TRUE_POS = 0 #valid acceptance
FALSE_POS = 1 #false acceptance
TRUE_NEG = 2 #valid reject
FALSE_NEG  = 3 #false reject 


def assign_validity(item):
    new_item = UserDist(item.user_a, item.user_b, 0)
    if item.user_a == item.user_b and item.dist < threat_threshold:
        new_item.dist = TRUE_POS # valid acceptance
    elif item.user_a == item.user_b and item.dist > threat_threshold:
        new_item.dist = FALSE_NEG #false rejection
    elif item.user_a != item.user_b and item.dist < threat_threshold:
        new_item.dist = FALSE_POS #different users, valid distance
    elif item.user_a != item.user_b and item.dist > threat_threshold:
        new_item.dist = TRUE_NEG #difference users, invalid distance
    return new_item


#Matrix of all validly accepted users  

ar_mat = [] #acceptance rate matrix

for arr in matrix:
    valid = []
    for item in arr:
        new_item = assign_validity(item)
        valid.append(new_item)
    ar_mat.append(valid)

ar_mat_df = pd.DataFrame(ar_mat)
print(ar_mat_df)
        
            


print('DISTANCE:', measure )

if measure < threat_threshold:
    print('VERIFIED')
else:
    print('UNVERIFIED')

