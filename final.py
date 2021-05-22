# -*- coding: utf-8 -*-
"""
Created on Sat May 15 09:46:06 2021

@author: user
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
#林志維的
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

#%%

#查看姓的數量分析
for title in ["Mr\.", "Sir\.", "Dr\.", "Major\.", "Master\."]:

    num = train[(train['Name'].str.contains(title))]["Name"].count()

    age_mean = round(train[(train['Name'].str.contains(title))]["Age"].mean(),1)

    age_median = train[(train['Name'].str.contains(title))]["Age"].median()

    num_survived = train[(train['Survived']==1) & (train['Name'].str.contains(title))]["Name"].count()

    num_died = train[(train['Survived']==0) & (train['Name'].str.contains(title))]["Name"].count()

    num_total = num_survived+num_died

    print("{} –> {} males, Age average is {}, median is {},  {} survived, {} died. {}% survived"

          .format(title, num, age_mean, age_median, num_survived, num_died, round(num_survived*100/num_total, 1)))

print("——————————————————————————–")

for title in ["Ms\.", "Miss\.", "Mrs\.", "Lady\."]:

    num = train[(train['Name'].str.contains(title))]["Name"].count()

    age_mean = round(train[(train['Name'].str.contains(title))]["Age"].mean(), 1)

    age_median = train[(train['Name'].str.contains(title))]["Age"].median()

    num_survived = train[(train['Survived']==1) & (train['Name'].str.contains(title))]["Name"].count()

    num_died = train[(train['Survived']==0) & (train['Name'].str.contains(title))]["Name"].count()

    num_total = num_survived+num_died

   

    print("{} –> {} females, Age average is {}, median is {},  {} survived, {} died. {}% survived"

          .format(title, num, age_mean, age_median, num_survived, num_died, round(num_survived*100/num_total, 1)))

#%%
#依照姓填入age
mask =  (train["Age"].isnull()) & ( (train['Name'].str.contains("Ms.")) | (train['Name'].str.contains("Miss.")) )

mask2 = ( (train['Name'].str.contains("Ms.")) | (train['Name'].str.contains("Miss.")) )

train.loc[mask,'Age'] = train.loc[mask,'Age'].fillna(train.loc[mask2,'Age'].median())


mask =  (train["Age"].isnull()) & ( (train['Name'].str.contains("Mr.")) | (train['Name'].str.contains("Sir.")) | (train['Name'].str.contains("Major")) )

mask2 =  ( (train['Name'].str.contains("Mr.")) | (train['Name'].str.contains("Sir.")) | (train['Name'].str.contains("Major")) )

train.loc[mask,'Age'] = train.loc[mask,'Age'].fillna(train.loc[mask2,'Age'].median())

mask =  (train["Age"].isnull()) & ( train['Name'].str.contains("Master.") )

train.loc[mask,'Age'] = train.loc[mask,'Age'].fillna(train[train['Name'].str.contains("Master.")]["Age"].median())

mask =  (train["Age"].isnull()) & ( train['Name'].str.contains("Dr.") )

train.loc[mask,'Age'] = train.loc[mask,'Age'].fillna(train[train['Name'].str.contains("Dr.")]["Age"].median())