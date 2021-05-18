#!/usr/bin/env python
# coding: utf-8

# In[9]:


# import function & setting

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[21]:


# import data

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

display(train)
display(test)


# In[19]:


# feature info

"""
PassengerId   Survival   Pclass   Name   Sex   Age   SibSp          Parch          Ticket   Fare    Cabin   Embarked
乘客ID編號     是否倖存     船票等級  姓名    性別  年齡   船上旁系親屬數目  船上直系親屬數目 船票編號   船票價格 船艙號碼 登船的口岸
                         1 = 1st
                         2 = 2nd
                         3 = 3rd
"""


# In[46]:


# filter strange data & fix data

# Pclass
# Missing : 133

# Name
# Missing : 0

# Sex
# Missing : 0
# Perfect!

# Age
# Missing : 189

# SibSp
# Missing : 0
# Perfect!

# Parch
# Missing : 0
# Perfect!

# Ticket
# Missing : 0
# different presentation way
print("Ticket data contains")
print(train["Ticket"].value_counts())
print("=============================")


# Fare
# Missing : 1

# Cabin
# Missing : 690
print("Cabin data contains")
print(train["Cabin"].value_counts())

# Embarked
# Missing : 0
# Perfect


# In[56]:


# split train data

split_train = train.sample(frac = 0.7, axis = 0)
split_test = train.append(split_train)
split_test = split_test.drop_duplicates(list(train), keep = False)

print("train shape : ", end = "")
print(train.shape)
print("split_train shape : ", end = "")
print(split_train.shape)
print("split_test shape : ", end = "")
print(split_test.shape)


# In[ ]:


# fill null data

# 1. fill mean value
Pclass_mean
Age_mean
SibSp_mean
Parch_mean
Ticket_mean
Fare_mean
Cabin_mean
Embarked_mean


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


"""
plt.scatter(train["SibSp"], train["Pclass"], edgecolors = "blue", s = 20, alpha = 0.5)
plt.xlabel("")
plt.ylabel("Pclass")
plt.title("")
plt.show()
"""

