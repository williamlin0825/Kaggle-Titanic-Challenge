# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib
import numpy as np
import pandas as pd
from math import sqrt,exp,pi
from matplotlib import pyplot as plt

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# %% preprocessing
# ============================================================================
""" Fill Pclass Method 1
# replace Nan in Cabin row with 0
train['Cabin'].fillna(0, inplace=True)
test['Cabin'].fillna(0, inplace=True)
train['Pclass'].fillna(0, inplace=True)
test['Pclass'].fillna(0, inplace=True)
# change the data form of Cabin to 0 and 1
train.loc[train['Cabin'] == 0, 'Cabin'] = 0
train.loc[train['Cabin'] != 0, 'Cabin'] = 1
test.loc[test['Cabin'] == 0, 'Cabin'] = 0
test.loc[test['Cabin'] != 0, 'Cabin'] = 1

# find the mean Fare of different Pclass
fare_mean = train.groupby('Pclass')['Fare'].mean()
fare_mean = pd.DataFrame({'Pclass': fare_mean.index, 'Fare': fare_mean.values})
class1_fare_mean = fare_mean['Fare'][1]
class2_fare_mean = fare_mean['Fare'][2]
class3_fare_mean = fare_mean['Fare'][3]

# fill the Nan of Pclass based on Fare difference with mean of fare of different class
for i in range(len(train)):
    if train['Pclass'][i] == 0:
        fare_difference1 = abs(train['Fare'][i]-class1_fare_mean)
        fare_difference2 = abs(train['Fare'][i]-class2_fare_mean)
        fare_difference3 = abs(train['Fare'][i]-class3_fare_mean)
        if min(fare_difference1, fare_difference2, fare_difference3) == fare_difference1:
            train['Pclass'][i] = 1
        if min(fare_difference1, fare_difference2, fare_difference3) == fare_difference2:
            train['Pclass'][i] = 2
        if min(fare_difference1, fare_difference2, fare_difference3) == fare_difference3:
            train['Pclass'][i] = 3

for i in range(len(test)):
    if test['Pclass'][i] == 0:
        fare_difference1 = abs(test['Fare'][i]-class1_fare_mean)
        fare_difference2 = abs(test['Fare'][i]-class2_fare_mean)
        fare_difference3 = abs(test['Fare'][i]-class3_fare_mean)
        if min(fare_difference1, fare_difference2, fare_difference3) == fare_difference1:
            test['Pclass'][i] = 1
        if min(fare_difference1, fare_difference2, fare_difference3) == fare_difference2:
            test['Pclass'][i] = 2
        if min(fare_difference1, fare_difference2, fare_difference3) == fare_difference3:
            test['Pclass'][i] = 3
"""

"""Fill Pclass Method 2"""
# replace Nan in Cabin row with 0
train['Pclass'].fillna(0, inplace=True)
test['Pclass'].fillna(0, inplace=True)

for i in range(len(train)):
    if train['Pclass'][i] == 0:
        if 'T' in str(train['Cabin'][i]):
            train['Pclass'][i] = 1
        if 'A' in str(train['Cabin'][i]):
            train['Pclass'][i] = 1
        if 'B' in str(train['Cabin'][i]):
            train['Pclass'][i] = 1
        if 'C' in str(train['Cabin'][i]):
            train['Pclass'][i] = 1
        if 'D' in str(train['Cabin'][i]):
            train['Pclass'][i] = 1
        if 'E' in str(train['Cabin'][i]):
            train['Pclass'][i] = 1
        
        
"""
# %% predict multivariate
# ============================================================================
# split train for train and test
train_train = train.sample(frac = 0.7, axis = 0)
train_test = train.append(train_train)
train_test = train_test.drop_duplicates(list(train), keep = False)

train_train_survived = train_train[train_train["Survived"] == 1]
train_train_dead = train_train[train_train["Survived"] == 0]

def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent
"""