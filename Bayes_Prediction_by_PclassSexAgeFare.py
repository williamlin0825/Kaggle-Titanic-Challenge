# -*- coding: utf-8 -*-
"""
Created on Sat May 29 00:30:10 2021

@author: user
"""

# %% Set Environment & Import Function
# ============================================================================
# Hide Warnings
import warnings
warnings.filterwarnings("ignore")

# Automatically Clear Var. & Console
from IPython import get_ipython
get_ipython().magic("clear")
get_ipython().magic("reset -f")

import numpy as np
import pandas as pd
from math import sqrt, exp, pi
from matplotlib import pyplot as plt

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#我有放preprocessing 有新的再換中間 從這裡!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# %% Preprocessing
# ============================================================================
# Change Column "Name" to "Full Name"
train = train.rename(columns = {"Name" : "Full Name"})

# Add "Family = SibSp + Parch Column"
train.insert(8, "Family", train["SibSp"] + train["Parch"] + 1) # 在船上的家族人數(有包含自己)

# %%% Fare
# 在還沒填補Pclass值之前先算各Pclass的Fare平均，避免因Pclass填補誤差造成Fare平均的誤差
Fare_Pclass1_mean = train.dropna(axis = 0, how = "all", subset = ["Fare"]).dropna(axis = 0, how = "all", subset = ["Pclass"])[train["Pclass"] == 1]["Fare"].mean()
Fare_Pclass2_mean = train.dropna(axis = 0, how = "all", subset = ["Fare"]).dropna(axis = 0, how = "all", subset = ["Pclass"])[train["Pclass"] == 2]["Fare"].mean()
Fare_Pclass3_mean = train.dropna(axis = 0, how = "all", subset = ["Fare"]).dropna(axis = 0, how = "all", subset = ["Pclass"])[train["Pclass"] == 3]["Fare"].mean()

train["Fare"].fillna(-1, inplace = True)
for i in range(len(train)):
    if train["Fare"][i] == -1:
        if train["Pclass"][i] == 1:
            train["Fare"][i] = Fare_Pclass1_mean
        if train["Pclass"][i] == 2:
            train["Fare"][i] = Fare_Pclass2_mean
        if train["Pclass"][i] == 3:
            train["Fare"][i] = Fare_Pclass3_mean



# Fill Pclass Method 2
# Age Distribution of Pclass
plt.bar(train[train["Pclass"] == 1]["Age"].value_counts().index, train[train["Pclass"] == 1]["Age"].value_counts())
plt.title("Age Distribution of Pclass 1")
plt.xlabel("Age")
plt.ylabel("Number of people")
plt.xlim([0, 100])
plt.ylim([0, 25])
plt.show()
plt.bar(train[train["Pclass"] == 2]["Age"].value_counts().index, train[train["Pclass"] == 2]["Age"].value_counts())
plt.title("Age Distribution of Pclass 2")
plt.xlabel("Age")
plt.ylabel("Number of people")
plt.xlim([0, 100])
plt.ylim([0, 25])
plt.show()
plt.bar(train[train["Pclass"] == 3]["Age"].value_counts().index, train[train["Pclass"] == 3]["Age"].value_counts())
plt.title("Age Distribution of Pclass 3")
plt.xlabel("Age")
plt.ylabel("Number of people")
plt.xlim([0, 100])
plt.ylim([0, 25])
plt.show()

# Max Fare of Pclass 2 & 3
max_fare_pclass_2and3 = train[train["Pclass"] > 1].max(skipna = True)["Fare"]

# Replace Nan in Cabin Row With -1
train["Pclass"].fillna(-1, inplace = True)

# Fill Pclass
for i in range(len(train)):
    if train["Pclass"][i] == -1: # Refer to Cabin
        if "T" in str(train["Cabin"][i]):
            train["Pclass"][i] = 1
        if "A" in str(train["Cabin"][i]):
            train["Pclass"][i] = 1
        if "B" in str(train["Cabin"][i]):
            train["Pclass"][i] = 1
        if "C" in str(train["Cabin"][i]):
            train["Pclass"][i] = 1
        if "D" in str(train["Cabin"][i]):
            train["Pclass"][i] = 1
        if "E" in str(train["Cabin"][i]):
            train["Pclass"][i] = 1
    if train["Pclass"][i] == -1: # Refer to Fare
        if train["Pclass"][i] > max_fare_pclass_2and3:
            train["Pclass"][i] = 1
    if train["Pclass"][i] == -1: # Refer to Embarked
        if train["Embarked"][i] == "Q":
            train["Pclass"][i] = 3

# %%% Name
# Split Last Name, Title and First Name
train.insert(4, "Last Name", train["Full Name"].str.split(", ", expand = True).iloc[:, 0])
train.insert(5, "Title", train["Full Name"].str.split(", ", expand = True).iloc[:, 1].str.split(".", expand = True).iloc[:, 0])
train.insert(6, "First Name", train["Full Name"].str.split(", ", expand = True).iloc[:, 1].str.split(".", expand = True).iloc[:, 1].map(lambda x: str(x)[1:]))


# %%% Age


Age_total_mean = train.dropna(axis = 0, how = "all", subset = ["Age"])["Age"].mean()
Age_Mr_mean = train[train["Title"] == "Mr"].dropna(axis = 0, how = "all", subset = ["Age"])["Age"].mean()
Age_Miss_mean = train[train["Title"] == "Miss"].dropna(axis = 0, how = "all", subset = ["Age"])["Age"].mean()
Age_Mrs_mean = train[train["Title"] == "Mrs"].dropna(axis = 0, how = "all", subset = ["Age"])["Age"].mean()
Age_Master_mean = train[train["Title"] == "Master"].dropna(axis = 0, how = "all", subset = ["Age"])["Age"].mean()
Age_Dr_mean = train[train["Title"] == "Dr"].dropna(axis = 0, how = "all", subset = ["Age"])["Age"].mean()
Age_Rev_mean = train[train["Title"] == "Rev"].dropna(axis = 0, how = "all", subset = ["Age"])["Age"].mean()
Age_Col_mean = train[train["Title"] == "Col"].dropna(axis = 0, how = "all", subset = ["Age"])["Age"].mean()
Age_Major_mean = train[train["Title"] == "Major"].dropna(axis = 0, how = "all", subset = ["Age"])["Age"].mean()
Age_Jonkheer_mean = train[train["Title"] == "Jonkheer"].dropna(axis = 0, how = "all", subset = ["Age"])["Age"].mean()
Age_Mlle_mean = train[train["Title"] == "Mlle"].dropna(axis = 0, how = "all", subset = ["Age"])["Age"].mean()
Age_Mme_mean = train[train["Title"] == "Mme"].dropna(axis = 0, how = "all", subset = ["Age"])["Age"].mean()
Age_Capt_mean = train[train["Title"] == "Capt"].dropna(axis = 0, how = "all", subset = ["Age"])["Age"].mean()
Age_Sir_mean = train[train["Title"] == "Sir"].dropna(axis = 0, how = "all", subset = ["Age"])["Age"].mean()

# Fill Nan Data With Mean of Certain Catag.
train["Age"].fillna(-1, inplace = True)
for i in range(len(train)):
    if train["Age"][i] == -1:
        if train["Title"][i] == "Mr":
            train["Age"][i] = Age_Mr_mean
        if train["Title"][i] == "Miss":
            train["Age"][i] = Age_Miss_mean
        if train["Title"][i] == "Mrs":
            train["Age"][i] = Age_Mrs_mean
        if train["Title"][i] == "Master":
            train["Age"][i] = Age_Master_mean
        if train["Title"][i] == "Dr":
            train["Age"][i] = Age_Dr_mean
        if train["Title"][i] == "Rev":
            train["Age"][i] = Age_Rev_mean
        if train["Title"][i] == "Col":
            train["Age"][i] = Age_Col_mean
        if train["Title"][i] == "Major":
            train["Age"][i] = Age_Major_mean
        if train["Title"][i] == "Jonkheer":
            train["Age"][i] = Age_Jonkheer_mean
        if train["Title"][i] == "Mlle":
            train["Age"][i] = Age_Mlle_mean
        if train["Title"][i] == "Mme":
            train["Age"][i] = Age_Mme_mean
        if train["Title"][i] == "Ms":
            train["Age"][i] = Age_Miss_mean
        if train["Title"][i] == "Capt":
            train["Age"][i] = Age_Capt_mean
        if train["Title"][i] == "Sir":
            train["Age"][i] = Age_Sir_mean
        else:
            train["Age"][i] = Age_total_mean

#我有放preprocessing 有新的再換中間 到這裡!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!            
            
#%%
accuracy_total=0
train1=train
#因為下面split會越split越小 所以將train複製
for w in range(30):
    from sklearn.model_selection import train_test_split
    train, test= train_test_split(train1 ,test_size = 0.3)
    
    
    #存活數與死亡數
    Survived1=len(train[train['Survived']==1])
    Survived0=len(train[train['Survived']==0])
    
    #3個Pclass之存活數與死亡數
    Pclass1_Survived1=len(train[(train['Pclass']==1) & (train['Survived']==1)])
    Pclass1_Survived0=len(train[(train['Pclass']==1) & (train['Survived']==0)])
    
    Pclass2_Survived1=len(train[(train['Pclass']==2) & (train['Survived']==1)])
    Pclass2_Survived0=len(train[(train['Pclass']==2) & (train['Survived']==0)])
    
    Pclass3_Survived1=len(train[(train['Pclass']==3) & (train['Survived']==1)])
    Pclass3_Survived0=len(train[(train['Pclass']==3) & (train['Survived']==0)])
    
    
    #兩性之存活數與死亡數
    Sex_male_Survived1=len(train[(train['Sex']=='male')& (train['Survived']==1)])
    Sex_male_Survived0=len(train[(train['Sex']=='male')& (train['Survived']==0)])
    Sex_female_Survived1=len(train[(train['Sex']=='female')& (train['Survived']==1)])
    Sex_female_Survived0=len(train[(train['Sex']=='female')& (train['Survived']==0)])
    
    
    
    data_train_survived = train[train["Survived"] == 1]
    data_train_dead = train[train["Survived"] == 0]
    
    Age_mean_target_survived = data_train_survived["Age"].mean()
    Age_std_target_survived = data_train_survived["Age"].std()
    #row_count_target_survived = data_train_survived.shape[0]
    Age_mean_target_dead = data_train_dead["Age"].mean()
    Age_std_target_dead = data_train_dead["Age"].std()
    #row_count_target_dead = data_train_dead.shape[0]
      
    
    Fare_mean_target_survived = data_train_survived["Fare"].mean()
    Fare_std_target_survived = data_train_survived["Fare"].std()
    #row_count_target_survived = data_train_survived.shape[0]
    Fare_mean_target_dead = data_train_dead["Fare"].mean()
    Fare_std_target_dead = data_train_dead["Fare"].std()
    #row_count_target_dead = data_train_dead.shape[0]
    
    
    
    
    # Prediction
    
    #各feature之函數 輸出為P(X|Ci)
    def Pclass_PXCi(data):
        if data==1:
            return [Pclass1_Survived1/Survived1,Pclass1_Survived0/Survived0]
        elif data==2:
            return [Pclass2_Survived1/Survived1,Pclass2_Survived0/Survived0]
        elif data==3:
            return [Pclass3_Survived1/Survived1,Pclass3_Survived0/Survived0]
        else:
            return [1,1]
        
        
    def Sex_PXCi(data):
        if data=='male':
            return (Sex_male_Survived1/Survived1,Sex_male_Survived0/Survived0)
        elif data=='female':
            return (Sex_female_Survived1/Survived1,Sex_female_Survived0/Survived0)
        else:
            return (1,1)
    
      
    def calculate_probability(x, mean, stdev):
    	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
    	return (1 / (sqrt(2 * pi) * stdev)) * exponent    
        
    def Age_PXCi(Age):
        Age_s1=1-calculate_probability(Age, Age_mean_target_survived, Age_std_target_survived)
        Age_s0=1-calculate_probability(Age, Age_mean_target_dead, Age_std_target_dead)
        return(Age_s1,Age_s0)
        
    
    def Fare_PXCi(Fare):
        Fare_s1=calculate_probability(Fare, Fare_mean_target_survived, Fare_std_target_survived)
        Fare_s0=calculate_probability(Fare, Fare_mean_target_dead, Fare_std_target_dead)
        return(Fare_s1,Fare_s0)
        
    
    #拿來放feature之P(X|Ci)
    z1=np.ones([len(test),5])
    z0=np.ones([len(test),5])
    
    #套用函數並將P(X|Ci)存至z1,z2    
    for i in range(len(test)):
        [z1[i,0],z0[i,0]]=Pclass_PXCi(test.iloc[i].at["Pclass"])
        [z1[i,1],z0[i,1]]=Sex_PXCi(test.iloc[i].at["Sex"])
        [z1[i,2],z0[i,2]]=Age_PXCi(test.iloc[i].at["Age"])
        [z1[i,3],z0[i,3]]=Fare_PXCi(test.iloc[i].at["Age"])
    
    
    predict=np.zeros([len(test)])
    for i  in range(len(test)):
        #要乘P(X|Ci)與P(活著)或P(死掉)
        survived=Survived1
        dead=Survived0
        for j in range(z1.shape[1]):
            survived*=z1[i,j]
            dead*=z0[i,j]
        if survived>=dead:
            predict[i]=1
        elif survived<dead:
            predict[i]=0
        
    predict_d= pd.DataFrame(predict, columns = ["Survived"])
    
    accurate_count = 0
    for j in range(predict_d.shape[0]):
        if test.iloc[j].at["Survived"] == predict_d.iloc[j].at["Survived"]:
            accurate_count = accurate_count + 1
    
    accuracy_1f = accurate_count / predict_d.shape[0]
    #print("Accuracy is : ", accuracy_1f * 100, "%")   
    accuracy_total=accuracy_total+accuracy_1f
    
print(accuracy_total/(w+1)  )              
