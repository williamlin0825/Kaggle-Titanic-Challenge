# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 00:41:09 2021

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


# %% Preprocessing
# ============================================================================
# Change Column "Name" to "Full Name"
train = train.rename(columns = {"Name" : "Full Name"})
test = test.rename(columns = {"Name" : "Full Name"})

# Add "Family = SibSp + Parch Column"
train.insert(8, "Family", train["SibSp"] + train["Parch"] + 1)
test.insert(7, "Family", test["SibSp"] + test["Parch"] + 1)

# %%% Fare
# 在還沒填補Pclass值之前先算各Pclass的Fare平均，避免因Pclass填補誤差造成Fare平均的誤差
Fare_Pclass1_mean = train.dropna(axis = 0, how = "all", subset = ["Fare"]).dropna(axis = 0, how = "all", subset = ["Pclass"])[train["Pclass"] == 1]["Fare"].mean()
Fare_Pclass2_mean = train.dropna(axis = 0, how = "all", subset = ["Fare"]).dropna(axis = 0, how = "all", subset = ["Pclass"])[train["Pclass"] == 2]["Fare"].mean()
Fare_Pclass3_mean = train.dropna(axis = 0, how = "all", subset = ["Fare"]).dropna(axis = 0, how = "all", subset = ["Pclass"])[train["Pclass"] == 3]["Fare"].mean()

train["Fare"].fillna(-1, inplace = True)
for i in range(len(train)):
    if train["Fare"][i] == -1: # Refer to Ticket
        train_drop_nan = train.drop(train[train["Fare"] == -1].index)
        if train_drop_nan[train_drop_nan["Ticket"] == train["Ticket"].iloc[i]].shape[0] != 0:
            # Find Mode
            mode_counts = np.bincount(train_drop_nan[train_drop_nan["Ticket"] == train["Ticket"].iloc[i]]["Fare"])
            mode = np.argmax(mode_counts)
            # Fill Mode
            train["Fare"][i] = mode
    if train["Fare"][i] == -1: # Fill Each Class Fare Mean
        if train["Pclass"][i] == 1:
            train["Fare"][i] = Fare_Pclass1_mean
        if train["Pclass"][i] == 2:
            train["Fare"][i] = Fare_Pclass2_mean
        if train["Pclass"][i] == 3:
            train["Fare"][i] = Fare_Pclass3_mean
            
test["Fare"].fillna(-1, inplace = True)
for i in range(len(test)):
    if test["Fare"][i] == -1: # Refer to Ticket
        train_drop_nan = train.drop(train[train["Fare"] == -1].index)
        if train_drop_nan[train_drop_nan["Ticket"] == test["Ticket"].iloc[i]].shape[0] != 0:
            # Find Mode
            mode_counts = np.bincount(train_drop_nan[train_drop_nan["Ticket"] == test["Ticket"].iloc[i]]["Fare"])
            mode = np.argmax(mode_counts)
            # Fill Mode
            test["Fare"][i] = mode
    if test["Fare"][i] == -1:
        if test["Pclass"][i] == 1:
            test["Fare"][i] = Fare_Pclass1_mean
        if test["Pclass"][i] == 2:
            test["Fare"][i] = Fare_Pclass2_mean
        if test["Pclass"][i] == 3:
            test["Fare"][i] = Fare_Pclass3_mean

# %%% Pclass
# Fill Pclass Method 1
"""
# replace Nan in Cabin row with 0
train["Cabin"].fillna(0, inplace=True)
test["Cabin"].fillna(0, inplace=True)
train["Pclass"].fillna(0, inplace=True)
test["Pclass"].fillna(0, inplace=True)
# change the data form of Cabin to 0 and 1
train.loc[train["Cabin"] == 0, "Cabin"] = 0
train.loc[train["Cabin"] != 0, "Cabin"] = 1
test.loc[test["Cabin"] == 0, "Cabin"] = 0
test.loc[test["Cabin"] != 0, "Cabin"] = 1
# find the mean Fare of different Pclass
fare_mean = train.groupby("Pclass")["Fare"].mean()
fare_mean = pd.DataFrame({"Pclass": fare_mean.index, "Fare": fare_mean.values})
class1_fare_mean = fare_mean["Fare"][1]
class2_fare_mean = fare_mean["Fare"][2]
class3_fare_mean = fare_mean["Fare"][3]
# fill the Nan of Pclass based on Fare difference with mean of fare of different class
for i in range(len(train)):
    if train["Pclass"][i] == 0:
        fare_difference1 = abs(train["Fare"][i]-class1_fare_mean)
        fare_difference2 = abs(train["Fare"][i]-class2_fare_mean)
        fare_difference3 = abs(train["Fare"][i]-class3_fare_mean)
        if min(fare_difference1, fare_difference2, fare_difference3) == fare_difference1:
            train["Pclass"][i] = 1
        if min(fare_difference1, fare_difference2, fare_difference3) == fare_difference2:
            train["Pclass"][i] = 2
        if min(fare_difference1, fare_difference2, fare_difference3) == fare_difference3:
            train["Pclass"][i] = 3
for i in range(len(test)):
    if test["Pclass"][i] == 0:
        fare_difference1 = abs(test["Fare"][i]-class1_fare_mean)
        fare_difference2 = abs(test["Fare"][i]-class2_fare_mean)
        fare_difference3 = abs(test["Fare"][i]-class3_fare_mean)
        if min(fare_difference1, fare_difference2, fare_difference3) == fare_difference1:
            test["Pclass"][i] = 1
        if min(fare_difference1, fare_difference2, fare_difference3) == fare_difference2:
            test["Pclass"][i] = 2
        if min(fare_difference1, fare_difference2, fare_difference3) == fare_difference3:
            test["Pclass"][i] = 3
"""

# Fill Pclass Method 2
"""
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
"""

# Max Fare of Pclass 2 & 3
max_fare_pclass_2and3 = train[train["Pclass"] > 1].max(skipna = True)["Fare"]

# Replace Nan in Cabin Row With -1
train["Pclass"].fillna(-1, inplace = True)
test["Pclass"].fillna(-1, inplace = True)

# Fill Pclass
for i in range(len(train)):
    if train["Pclass"][i] == -1: # Refer to Ticket
        train_drop_nan = train.drop(train[train["Pclass"] == -1].index)
        if train_drop_nan[train_drop_nan["Ticket"] == train["Ticket"].iloc[i]].shape[0] != 0:
            # Find Mode
            mode_counts = np.bincount(train_drop_nan[train_drop_nan["Ticket"] == train["Ticket"].iloc[i]]["Pclass"])
            mode = np.argmax(mode_counts)
            # Fill Mode
            train["Pclass"][i] = mode
    if train["Pclass"][i] == -1: # Refer to Cabin
        if "T" in str(train["Cabin"][i]):
            train["Pclass"][i] = 1
        elif "A" in str(train["Cabin"][i]):
            train["Pclass"][i] = 1
        elif "B" in str(train["Cabin"][i]):
            train["Pclass"][i] = 1
        elif "C" in str(train["Cabin"][i]):
            train["Pclass"][i] = 1
        elif "D" in str(train["Cabin"][i]):
            train["Pclass"][i] = 1
        elif "E" in str(train["Cabin"][i]):
            train["Pclass"][i] = 1
    if train["Pclass"][i] == -1: # Refer to Fare
        if train["Pclass"][i] > max_fare_pclass_2and3:
            train["Pclass"][i] = 1
    if train["Pclass"][i] == -1: # Refer to Embarked
        if train["Embarked"][i] == "Q":
            train["Pclass"][i] = 3
            
for i in range(len(test)):
    if test["Pclass"][i] == -1: # Refer to Ticket
        train_drop_nan = train.drop(train[train["Pclass"] == -1].index)
        if train_drop_nan[train_drop_nan["Ticket"] == test["Ticket"].iloc[i]].shape[0] != 0:
            # Find Mode
            mode_counts = np.bincount(train_drop_nan[train_drop_nan["Ticket"] == test["Ticket"].iloc[i]]["Pclass"])
            mode = np.argmax(mode_counts)
            # Fill Mode
            test["Pclass"][i] = mode
    if test["Pclass"][i] == -1: # Refer to Cabin
        if "T" in str(test["Cabin"][i]):
            test["Pclass"][i] = 1
        elif "A" in str(test["Cabin"][i]):
            test["Pclass"][i] = 1
        elif "B" in str(test["Cabin"][i]):
            test["Pclass"][i] = 1
        elif "C" in str(test["Cabin"][i]):
            test["Pclass"][i] = 1
        elif "D" in str(test["Cabin"][i]):
            test["Pclass"][i] = 1
        elif "E" in str(test["Cabin"][i]):
            test["Pclass"][i] = 1
    if test["Pclass"][i] == -1: # Refer to Fare
        if test["Pclass"][i] > max_fare_pclass_2and3:
            test["Pclass"][i] = 1
    if test["Pclass"][i] == -1: # Refer to Embarked
        if test["Embarked"][i] == "Q":
            test["Pclass"][i] = 3

# %%% Name
# Split Last Name, Title and First Name
train.insert(4, "Last Name", train["Full Name"].str.split(", ", expand = True).iloc[:, 0])
train.insert(5, "Title", train["Full Name"].str.split(", ", expand = True).iloc[:, 1].str.split(".", expand = True).iloc[:, 0])
train.insert(6, "First Name", train["Full Name"].str.split(", ", expand = True).iloc[:, 1].str.split(".", expand = True).iloc[:, 1].map(lambda x: str(x)[1:]))

test.insert(3, "Last Name", test["Full Name"].str.split(", ", expand = True).iloc[:, 0])
test.insert(4, "Title", test["Full Name"].str.split(", ", expand = True).iloc[:, 1].str.split(".", expand = True).iloc[:, 0])
test.insert(5, "First Name", test["Full Name"].str.split(", ", expand = True).iloc[:, 1].str.split(".", expand = True).iloc[:, 1].map(lambda x: str(x)[1:]))

"""
print(train["Title"].value_counts()) # number of people of each title
"""

# %%% Age
"""
# Age Distribution of "Mr" title
plt.bar(train[train["Title"] == "Mr"]["Age"].value_counts().index, train[train["Title"] == "Mr"]["Age"].value_counts())
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of people")
plt.show()
"""

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
        elif train["Title"][i] == "Miss":
            train["Age"][i] = Age_Miss_mean
        elif train["Title"][i] == "Mrs":
            train["Age"][i] = Age_Mrs_mean
        elif train["Title"][i] == "Master":
            train["Age"][i] = Age_Master_mean
        elif train["Title"][i] == "Dr":
            train["Age"][i] = Age_Dr_mean
        elif train["Title"][i] == "Rev":
            train["Age"][i] = Age_Rev_mean
        elif train["Title"][i] == "Col":
            train["Age"][i] = Age_Col_mean
        elif train["Title"][i] == "Major":
            train["Age"][i] = Age_Major_mean
        elif train["Title"][i] == "Jonkheer":
            train["Age"][i] = Age_Jonkheer_mean
        elif train["Title"][i] == "Mlle":
            train["Age"][i] = Age_Mlle_mean
        elif train["Title"][i] == "Mme":
            train["Age"][i] = Age_Mme_mean
        elif train["Title"][i] == "Ms":
            train["Age"][i] = Age_Miss_mean
        elif train["Title"][i] == "Capt":
            train["Age"][i] = Age_Capt_mean
        elif train["Title"][i] == "Sir":
            train["Age"][i] = Age_Sir_mean
        else:
            train["Age"][i] = Age_total_mean

test["Age"].fillna(-1, inplace = True)
for i in range(len(test)):
    if test["Age"][i] == -1:
        if test["Title"][i] == "Mr":
            test["Age"][i] = Age_Mr_mean
        elif test["Title"][i] == "Miss":
            test["Age"][i] = Age_Miss_mean
        elif test["Title"][i] == "Mrs":
            test["Age"][i] = Age_Mrs_mean
        elif test["Title"][i] == "Master":
            test["Age"][i] = Age_Master_mean
        elif test["Title"][i] == "Dr":
            test["Age"][i] = Age_Dr_mean
        elif test["Title"][i] == "Rev":
            test["Age"][i] = Age_Rev_mean
        elif test["Title"][i] == "Col":
            test["Age"][i] = Age_Col_mean
        elif test["Title"][i] == "Major":
            test["Age"][i] = Age_Major_mean
        elif test["Title"][i] == "Jonkheer":
            test["Age"][i] = Age_Jonkheer_mean
        elif test["Title"][i] == "Mlle":
            test["Age"][i] = Age_Mlle_mean
        elif test["Title"][i] == "Mme":
            test["Age"][i] = Age_Mme_mean
        elif test["Title"][i] == "Ms":
            test["Age"][i] = Age_Miss_mean
        elif test["Title"][i] == "Capt":
            test["Age"][i] = Age_Capt_mean
        elif test["Title"][i] == "Sir":
            test["Age"][i] = Age_Sir_mean
        else:
            test["Age"][i] = Age_total_mean

# %%% Embarked
test["Embarked"].fillna(-1, inplace = True)
for i in range(len(test)):
    if test["Embarked"][i] == -1:
        test["Embarked"][i] = "S"

# %%% Age Group
# Age Distribution Plot
"""
plt.bar(train["Age"].value_counts().index, train["Age"].value_counts())
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of people")
plt.show()
"""

train["Age Group"] = 0
for i in range(len(train)): # ~12 : 1 ; 12~60 : 2 ; 60~ : 3
    if train["Age"].iloc[i] < 12:
        train["Age Group"].iloc[i] = 1
    elif train["Age"].iloc[i] >= 12 and train["Age"].iloc[i] < 60:
        train["Age Group"].iloc[i] = 2
    elif train["Age"].iloc[i] >= 60:
        train["Age Group"].iloc[i] = 3
        
test["Age Group"] = 0
for i in range(len(test)): # ~12 : 1 ; 12~60 : 2 ; 60~ : 3
    if test["Age"].iloc[i] < 12:
        test["Age Group"].iloc[i] = 1
    elif test["Age"].iloc[i] >= 12 and test["Age"].iloc[i] < 60:
        test["Age Group"].iloc[i] = 2
    elif test["Age"].iloc[i] >= 60:
        test["Age Group"].iloc[i] = 3

# %%% Relationship
# Number of People Who Share One Ticket Number Distribution
plt.bar(train["Ticket"].value_counts().value_counts().index, train["Ticket"].value_counts().value_counts())
plt.title("Number of People Who Share One Ticket Number Distribution")
plt.xlabel("Number of People Who Share One Ticket Number")
plt.ylabel("Number")
plt.show()

# 先用船票判斷是否一起再用年齡整理關係

# Family = 2
# Husband(211) and Wife(212)
# Meaning of the number (eg. 211) : "2" people, case "1", no."1"

# Dad and Child

# Mom and Child


# Family = 3
# Dad and Mom and Child

# %% To Do List
# ============================================================================
# 把所有填值都合併在一個for迴圈內
#%%
accuracy_total=0

train1=train

#因為下面split會越split越小 所以將train複製
for w in range(50):
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
    
    #3個Embarked之存活數與死亡數
    Embarked_C_Survived1=len(train[(train['Embarked']=='C') & (train['Survived']==1)])
    Embarked_C_Survived0=len(train[(train['Embarked']=='C') & (train['Survived']==0)])
    
    Embarked_S_Survived1=len(train[(train['Embarked']=='S') & (train['Survived']==1)])
    Embarked_S_Survived0=len(train[(train['Embarked']=='S') & (train['Survived']==0)])
    
    Embarked_Q_Survived1=len(train[(train['Embarked']=='Q') & (train['Survived']==1)])
    Embarked_Q_Survived0=len(train[(train['Embarked']=='Q') & (train['Survived']==0)])
    
    
    def family_i1(i):
        f1=(len(train[(train['Family']==i) & (train['Survived']==1)]))/Survived1
        #f0=(len(train[(train['Family']==i) & (train['Survived']==0)]))/Survived0
        return (f1)
    def family_i0(i):
        #f1=(len(train[(train['Family']==i) & (train['Survived']==1)]))/Survived1
        f0=(len(train[(train['Family']==i) & (train['Survived']==0)]))/Survived0
        return (f0)
    
    family_atleast6_Survived1=0
    family_atleast6_Survived0=0
    
    for i in range(6,12):
        family_atleast6_Survived1 += family_i1(i)
        family_atleast6_Survived0 += family_i0(i)
    
    
#%%    
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
        
    def Embarked_PXCi(data):
        if data=='C':
            return [Embarked_C_Survived1/Survived1,Embarked_C_Survived0/Survived0]
        elif data=='S':
            return [Embarked_S_Survived1/Survived1,Embarked_S_Survived0/Survived0]
        elif data=='Q':
            return [Embarked_Q_Survived1/Survived1,Embarked_Q_Survived0/Survived0]
        else:
            return [1,1]
        
        
    def Family_PXCi(data):
        if data<=5:
            return [family_i1(data),family_i0(data)]
        elif data>=6:
            return [family_atleast6_Survived1,family_atleast6_Survived0]
        else:
            return [1,1]
    
    
    
    #拿來放feature之P(X|Ci)
    z1=np.ones([len(test),8])
    z0=np.ones([len(test),8])
    
    #套用函數並將P(X|Ci)存至z1,z2    
    for i in range(len(test)):
        [z1[i,0],z0[i,0]]=Pclass_PXCi(test.iloc[i].at["Pclass"])
        [z1[i,1],z0[i,1]]=Sex_PXCi(test.iloc[i].at["Sex"])
        [z1[i,2],z0[i,2]]=Age_PXCi(test.iloc[i].at["Age"])
        #[z1[i,3],z0[i,3]]=Fare_PXCi(test.iloc[i].at["Fare"])
        [z1[i,4],z0[i,4]]=Embarked_PXCi(test.iloc[i].at["Embarked"])
        [z1[i,5],z0[i,5]]=Family_PXCi(test.iloc[i].at["Family"])
    
    
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
