# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

# ============================================================================
#preprocessing
# ============================================================================
#%%
#replace Nan in Cabin row with 0
train['Cabin'].fillna(0, inplace=True)
test['Cabin'].fillna(0, inplace=True)
train['Pclass'].fillna(0, inplace=True)
test['Pclass'].fillna(0, inplace=True)
#change the data form of Cabin to 0 and 1
train.loc[train['Cabin'] ==0, 'Cabin'] = 0
train.loc[train['Cabin'] !=0, 'Cabin'] = 1
test.loc[test['Cabin'] ==0, 'Cabin'] = 0
test.loc[test['Cabin'] !=0, 'Cabin'] = 1

#find the mean Fare of different Pclass
fare_mean=train.groupby('Pclass')['Fare'].mean()
fare_mean=pd.DataFrame({'Pclass':fare_mean.index, 'Fare':fare_mean.values})
class1_fare_mean=fare_mean['Fare'][1]
class2_fare_mean=fare_mean['Fare'][2]
class3_fare_mean=fare_mean['Fare'][3]

#fill the Nan of Pclass based on Fare difference with mean of fare of different class
for i in range(len(train)):
    if train['Pclass'][i]==0:
        fare_difference1=abs(train['Fare'][i]-class1_fare_mean)
        fare_difference2=abs(train['Fare'][i]-class2_fare_mean)
        fare_difference3=abs(train['Fare'][i]-class3_fare_mean)
        if min(fare_difference1,fare_difference2,fare_difference3)==fare_difference1:
            train['Pclass'][i]=1
        if min(fare_difference1,fare_difference2,fare_difference3)==fare_difference2:
            train['Pclass'][i]=2
        if min(fare_difference1,fare_difference2,fare_difference3)==fare_difference3:
            train['Pclass'][i]=3

for i in range(len(test)):
    if test['Pclass'][i]==0:
        fare_difference1=abs(test['Fare'][i]-class1_fare_mean)
        fare_difference2=abs(test['Fare'][i]-class2_fare_mean)
        fare_difference3=abs(test['Fare'][i]-class3_fare_mean)
        if min(fare_difference1,fare_difference2,fare_difference3)==fare_difference1:
            test['Pclass'][i]=1
        if min(fare_difference1,fare_difference2,fare_difference3)==fare_difference2:
            test['Pclass'][i]=2
        if min(fare_difference1,fare_difference2,fare_difference3)==fare_difference3:
            test['Pclass'][i]=3
            
#fill the age of Nan people based on their title with the mean age of that title

