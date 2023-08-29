import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import signal
from sklearn.metrics import accuracy_score
import random
import pyswarms as ps
import time

t = time.time()
#Loading Data and deleting unnessary columns
eeg_data = pd.read_csv("data.csv")
# Convert the 5 classes into binary class
def convert_binary_class(y):
    if y == 2 or y == 3 or y == 4 or y == 5 :
        return 0
    else :
        return 1

# Apply above function to convert into binary class
eeg_data['y'] = eeg_data['y'].apply(convert_binary_class)

# Remove unnecessory columns that is the name of the signal
del eeg_data['Signal']
#EEG Signal Data
eeg_X = eeg_data.drop('y',axis=1)

#EEG Signal Label
eeg_Y = eeg_data['y']

# Functions for Calculating Features
#feature extraction
#mean, variance, standard deviation, average power, mean absolute value, energy
#############################################################################################
# Mean
def mean(x):
    mean_value = []
    for a in range(11500):
        mean_value.append(np.mean(x[a]))
    return mean_value

#Energy
def energy(x):
    energy = []
    for a in range(11500):
        energy.append(np.sum(np.square(x[a])))
    return energy


#Standard Deviation
def std(x):
    std = []
    for a in range(11500):
        std.append(np.std(x[a]))
    return std

#Variance
def variance(x):
    var = []
    for a in range(11500):
        var.append(np.var(x[a]))
    return var

#Mean Abolute Value
def absolute_value(x):
    av = []
    for a in range(11500):
        av.append(np.mean(np.absolute(x[a])))
    return av

#Average Power
def average_power(x):
    av_power = []
    for a in range(11500):
        av_power.append(np.mean(np.square(x[a])))
    return av_power

#####################################################################


# Creating Dataframes for coefficients
cA = [i for i in range(17)]
cD1 = [i for i in range(17)]
cD2 = [i for i in range(28)]
cD3 = [i for i in range(49)]
cD4 = [i for i in range(92)]

df_cA = pd.DataFrame(columns=cA)
df_cD1 = pd.DataFrame(columns=cD1)
df_cD2 = pd.DataFrame(columns=cD2)
df_cD3 = pd.DataFrame(columns=cD3)
df_cD4 = pd.DataFrame(columns=cD4)

def c1(x):
    # cA
    a,b,c,d,e = pywt.wavedec(x, 'db4', level=4)
    return a

def c2(x):
    # cD1
    a,b,c,d,e = pywt.wavedec(x, 'db4', level=4)
    return b

def c3(x):
    # cD2
    a,b,c,d,e = pywt.wavedec(x, 'db4', level=4)
    return c

def c4(x):
    # cD3
    a,b,c,d,e = pywt.wavedec(x, 'db4', level=4)
    return d

def c5(x):
    # cD4
    a,b,c,d,e = pywt.wavedec(x, 'db4', level=4)
    return e

#Creating cA dataframe for each signal
for i in range(11500):
    row = eeg_X.loc[i]
    a = c1(row)
    df_cA.loc[i] = a

#Creating cD1 dataframe for each signal
for i in range(11500):
    row = eeg_X.loc[i]
    p = c2(row)
    df_cD1.loc[i] = p

#Creating cD2 dataframe for each signal
for i in range(11500):
    row = eeg_X.loc[i]
    q = c3(row)
    df_cD2.loc[i] = q

#Creating cD3 dataframe for each signal
for i in range(11500):
    row = eeg_X.loc[i]
    r = c4(row)
    df_cD3.loc[i] = r

#Creating cD4 dataframe for each signal
for i in range(11500):
    row = eeg_X.loc[i]
    s = c5(row)
    df_cD4.loc[i] = s

#Feature extraction from cA
mean_cA = mean(np.array(df_cA))
energy_cA = energy(np.array(df_cA))
std_cA = std(np.array(df_cA))
var_cA = variance(np.array(df_cA))
absvalue_cA = absolute_value(np.array(df_cA))
avgpower_cA = average_power(np.array(df_cA))


#Feature extraction from cD1
mean_cD1 = mean(np.array(df_cD1))
energy_cD1 = energy(np.array(df_cD1))
std_cD1 = std(np.array(df_cD1))
var_cD1 = variance(np.array(df_cD1))
absvalue_cD1 = absolute_value(np.array(df_cD1))
avgpower_cD1 = average_power(np.array(df_cD1))

#Feature extraction from cD2
mean_cD2 = mean(np.array(df_cD2))
energy_cD2 = energy(np.array(df_cD2))
std_cD2 = std(np.array(df_cD2))
var_cD2 = variance(np.array(df_cD2))
absvalue_cD2 = absolute_value(np.array(df_cD2))
avgpower_cD2 = average_power(np.array(df_cD2))

#Feature extraction from cD3
mean_cD3 = mean(np.array(df_cD3))
energy_cD3 = energy(np.array(df_cD3))
std_cD3 = std(np.array(df_cD3))
var_cD3 = variance(np.array(df_cD3))
absvalue_cD3 = absolute_value(np.array(df_cD3))
avgpower_cD3 = average_power(np.array(df_cD3))

#Feature extraction from cD4
mean_cD4 = mean(np.array(df_cD4))
energy_cD4 = energy(np.array(df_cD4))
std_cD4 = std(np.array(df_cD4))
var_cD4 = variance(np.array(df_cD4))
absvalue_cD4 = absolute_value(np.array(df_cD4))
avgpower_cD4 = average_power(np.array(df_cD4))


features = pd.DataFrame({'mean_cA': mean_cA,'energy_cA': energy_cA,'std_cA':std_cA,
                        'var_cA':var_cA,'absvalue_cA':absvalue_cA,'avgpower_cA':avgpower_cA,

                        'mean_cD1': mean_cD1,'energy_cD1': energy_cD1,'std_cD1':std_cD1,
                        'var_cD1':var_cD1,'absvalue_cD1':absvalue_cD1,'avgpower_cD1':avgpower_cD1,

                         'mean_cD2': mean_cD2,'energy_cD2': energy_cD2,'std_cD2':std_cD2,
                        'var_cD2':var_cD2,'absvalue_cD2':absvalue_cD2,'avgpower_cD2':avgpower_cD2,

                         'mean_cD3': mean_cD3,'energy_cD3': energy_cD3,'std_cD3':std_cD3,
                        'var_cD3':var_cD3,'absvalue_cD3':absvalue_cD3,'avgpower_cD3':avgpower_cD3,

                         'mean_cD4': mean_cD4,'energy_cD4': energy_cD4,'std_cD4':std_cD4,
                        'var_cD4':var_cD4,'absvalue_cD4':absvalue_cD4,'avgpower_cD4':avgpower_cD4
                        })

#Add the labels to the features dataframe
features = pd.concat([features, eeg_Y], axis=1, sort=False)

#creating csv file of features
features.to_csv("feat.csv", sep=',',index=False, encoding='utf-8')

#Displaying the first five rows of extracted features
extracted_features = pd.read_csv('features.csv')
print(extracted_features.head())

print("Time required for Feature Extraction : {}".format(time.time()-t))
