# -*- coding: utf-8 -*-
"""
Created on Wed May 26 22:43:45 2021

@author: KuanHungWu
"""
"""
from IPython import get_ipython
get_ipython().magic('clear')
get_ipython().magic('reset -f')
"""

import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt,exp,pi
from sklearn.model_selection import train_test_split

#%% Data Preprocessing
data_original = pd.read_csv("train.csv")
data_filtered_Nan = data_original.dropna(axis = 0, how = "all", subset = ["Age"])
data_filtered_Nan["Relative Age"] = data_filtered_Nan["Age"] - data_filtered_Nan["Age"].mean() #年齡 - 有年齡值的人的平均年齡
data_filtered_Nan["Relative Age Squared"] = data_filtered_Nan["Relative Age"] ** 2 #(年齡 - 有年齡值的人的平均年齡) ** 2

X = data_filtered_Nan.drop(["Survived"], axis = 1)
y = data_filtered_Nan["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
data_train = pd.concat([X_train, y_train], axis = 1)

#%% Prediction
data_train_survived = data_train[data_train["Survived"] == 1]
data_train_dead = data_train[data_train["Survived"] == 0]

Age_mean_target_survived = data_train_survived["Relative Age Squared"].mean()
Age_std_target_survived = data_train_survived["Relative Age Squared"].std()
row_count_target_survived = data_train_survived.shape[0]
Age_mean_target_dead = data_train_dead["Relative Age Squared"].mean()
Age_std_target_dead = data_train_dead["Relative Age Squared"].std()
row_count_target_dead = data_train_dead.shape[0]

def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent

def high_or_low_1f(Age):
    row_count_total = row_count_target_survived + row_count_target_dead
    
    probabilities = dict()
    #P(C_survived)
    probabilities[1] = row_count_target_survived / float(row_count_total)
    #P(C_survived) * P(Age_survived | C_survived)
    probabilities[1] *= calculate_probability(Age, Age_mean_target_survived, Age_std_target_survived)
    
    #P(C_dead)
    probabilities[0] = row_count_target_dead / float(row_count_total)
    #P(C_dead) * P(Age_dead | C_dead)
    probabilities[0] *= calculate_probability(Age, Age_mean_target_dead, Age_std_target_dead)
    
    if probabilities[1] >= probabilities[0]:
        return 1.0
    else:
        return 0.0

target_guess = []
for i in range(len(X_test)):
    target_guess.append(high_or_low_1f(X_test.iloc[i].at["Relative Age Squared"]))
target_guess = pd.DataFrame(target_guess, columns = ["target guess 1f"])
y_test = pd.DataFrame(y_test, columns = ["Survived"])

accurate_count = 0
for j in range(y_test.shape[0]):
    if target_guess.iloc[j].at["target guess 1f"] == y_test.iloc[j].at["Survived"]:
        accurate_count = accurate_count + 1

accuracy_1f = accurate_count / y_test.shape[0]
print("Accuracy is : ", accuracy_1f * 100, "%")