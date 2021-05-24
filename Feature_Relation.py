# -*- coding: utf-8 -*-
"""
Created on Mon May 24 15:20:53 2021

@author: KuanHungWu
"""
import matplotlib
import numpy as np
import pandas as pd
from math import sqrt, exp, pi
from matplotlib import pyplot as plt

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# %% 找Pclass關係來預測
# ============================================================================
# 不同的親屬人數 Pclass比例
Pclass_Kinship_relation = []  # Final Relation Result
Kinship_count = []
Pclass1_number = []
Pclass2_number = []
Pclass3_number = []
for i in range(max(train["SibSp"] + train["Parch"]) + 1):
    Kinship_count.append(i)
    select = train[train["SibSp"] + train["Parch"] == i]

    Pclass1_number.append(select[select["Pclass"] == 1].shape[0])
    Pclass2_number.append(select[select["Pclass"] == 2].shape[0])
    Pclass3_number.append(select[select["Pclass"] == 3].shape[0])

Pclass_Kinship_relation = pd.DataFrame(Kinship_count)
Pclass_Kinship_relation["Pclass1 number"] = Pclass1_number
Pclass_Kinship_relation["Pclass2 number"] = Pclass2_number
Pclass_Kinship_relation["Pclass3 number"] = Pclass3_number

# 不同港口上船的Pclass比例
print("=================不同港口上船的Pclass比例=================")
train = pd.read_csv("train.csv")
print("from S")
from_S = train[train["Embarked"].str.startswith("S") == True]
print(from_S["Pclass"].value_counts())
print("from C")
from_C = train[train["Embarked"].str.startswith("C") == True]
print(from_C["Pclass"].value_counts())
print("from Q")
from_Q = train[train["Embarked"].str.startswith("Q") == True]
print(from_Q["Pclass"].value_counts())
