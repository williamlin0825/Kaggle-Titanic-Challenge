# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
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
import random 

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


def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

train=set_Cabin_type(train)
test=set_Cabin_type(test)



#Factorize if 因子化 is disabled
"""
train['Sex'] = pd.factorize(train['Sex'])[0] 
test['Sex'] = pd.factorize(test['Sex'])[0] 
train['Embarked'] = pd.factorize(train['Embarked'])[0] 
test['Embarked'] = pd.factorize(test['Embarked'])[0]
print(train)
"""
#因子化

d_Sex=pd.get_dummies(train['Sex'],prefix='Sex')
d_Pclass=pd.get_dummies(train['Pclass'],prefix='Pclass')
d_Embarked=pd.get_dummies(train['Embarked'],prefix='Embarked')
d_Cabin=pd.get_dummies(train['Cabin'],prefix='Cabin')
train=pd.concat([train,d_Sex,d_Pclass,d_Embarked,d_Cabin],axis=1)
train_temp=train
train.drop(['Full Name','Last Name','Title','First Name','Sex','Embarked','Cabin','Ticket'],axis=1,inplace=True)


d_Sex1=pd.get_dummies(test['Sex'],prefix='Sex')
d_Pclass1=pd.get_dummies(test['Pclass'],prefix='Pclass')
d_Embarked1=pd.get_dummies(test['Embarked'],prefix='Embarked')
d_Cabin1=pd.get_dummies(test['Cabin'],prefix='Cabin')
test=pd.concat([test,d_Sex1,d_Pclass1,d_Embarked1,d_Cabin1],axis=1)
test_temp=test
test.drop(['Full Name','Last Name','Title','First Name','Sex','Embarked','Cabin','Ticket'],axis=1,inplace=True)



#standardize Age and Fare

train_normalize_max_age=((train['Age']-train['Age'].mean())/train['Age'].std()).max()
train_normalize_min_age=((train['Age']-train['Age'].mean())/train['Age'].std()).min()
test['Age'] = 2*(((test['Age']-train['Age'].mean())/train['Age'].std())-train_normalize_min_age)/(train_normalize_max_age-train_normalize_min_age)-1
train['Age'] = 2*(((train['Age']-train['Age'].mean())/train['Age'].std())-train_normalize_min_age)/(train_normalize_max_age-train_normalize_min_age)-1


train_normalize_max_fare=((train['Fare']-train['Fare'].mean())/train['Fare'].std()).max()
train_normalize_min_fare=((train['Fare']-train['Fare'].mean())/train['Fare'].std()).min()
test['Fare'] = 2*(((test['Fare']-train['Fare'].mean())/train['Fare'].std())-train_normalize_min_fare)/(train_normalize_max_fare-train_normalize_min_fare)-1
train['Fare'] = 2*(((train['Fare']-train['Fare'].mean())/train['Fare'].std())-train_normalize_min_fare)/(train_normalize_max_fare-train_normalize_min_fare)-1




#Split Training set value to train_of_train and train_of_test
train_of_train=train.sample(frac=0.7)
test_of_train=train.drop(train_of_train.index)
#print (train_of_train,'\n',test_of_train)

y_train_of_train=train_of_train.loc[:,['Survived']]
y_test_of_train=test_of_train.loc[:,['Survived']]       #get the survived array
y_train=train.loc[:,['Survived']]

passenger=test.loc[:,['PassengerId']]


test_of_train.drop(['PassengerId','Survived'], axis=1,inplace=True)
train_of_train.drop(['PassengerId','Survived'], axis=1,inplace=True)
train.drop(['PassengerId','Survived'], axis=1,inplace=True)             #drop the unused data of training
test.drop(['PassengerId'], axis=1,inplace=True)


#change to array to compute
y_train_of_train=y_train_of_train.to_numpy()
y_test_of_train=y_test_of_train.to_numpy()
y_train=y_train.to_numpy()

train_of_train=train_of_train.to_numpy()
test_of_train=test_of_train.to_numpy()

test=test.to_numpy()
train=train.to_numpy()

passenger=passenger.to_numpy()

#reshape y
y_train_of_train=np.squeeze(y_train_of_train)
y_test_of_train=np.squeeze(y_test_of_train)
y_train=np.squeeze(y_train)

# ============================================================================
# %% Additional function
# ============================================================================

def accuracy_calculate(preds,y):
    count=0
    for i in range(len(preds)):
        if y_train[i]==preds[i]:
            count+=1
    accuracy=count/len(preds)
    print("The accuracy of training set is ",accuracy*100," %")
    return accuracy

def save_data(passenger,result):
    result=np.reshape(result,(len(result),1))
    final_result=np.concatenate((passenger, result),axis=1)
    final_result=final_result.astype(int)
    dataframe=pd.DataFrame(final_result, columns=['PassengerId','Survived']) 
    print (dataframe)
    dataframe.to_csv('result.csv',index=False)
    


# ============================================================================
# %% Logistic Regression
# ============================================================================
def sigmoid(input):    
    output = 1 / (1 + np.exp(-input))
    return output

weight_history=[]
bias_history=[]

def optimize(x, y,learning_rate,iterations,parameters): 
    size = x.shape[0]
    weight = parameters["weight"] 
    bias = parameters["bias"]
    for i in range(iterations): 
        sigma = sigmoid(np.dot(x, weight) + bias)
        loss = (-1/size )*np.sum(y * np.log(sigma) + (1 - y) * np.log(1-sigma))
        dW = 1/size * np.dot(x.T, (sigma - y))
        db = 1/size * np.sum(sigma - y)
        weight -= learning_rate * dW
        weight_history.append(weight)
        bias -= learning_rate * db 
        bias_history.append(bias)
        if (i+1)%100==0:
            print('=== Iteration: %d ===' %(i+1))
            print('Training loss: %.4f' %loss)
    
    parameters["weight"] = weight
    parameters["bias"] = bias
    return parameters

init_parameters = {} 
init_parameters["weight"] = np.random.randn(train.shape[1])
init_parameters["bias"] = 0

def train_process(x, y, learning_rate,iterations):
    parameters_out = optimize(x, y, learning_rate, iterations ,init_parameters)
    return parameters_out

# ============================================================================
# %% Training and Testing
# ============================================================================
parameters_out = train_process(train, y_train, learning_rate = 0.1, iterations = 100000)
output_values=np.dot(train,parameters_out["weight"])+parameters_out["bias"]
prediction=np.zeros(len(output_values))

for i in range(len(output_values)):
    if sigmoid(output_values[i])>=1/2:
        prediction[i]=1
    else:
        prediction[i]=0
accuracy_calculate(prediction,y_train)

# ============================================================================
# %% Get the final result for testing set
# ============================================================================
final_output=np.dot(test,parameters_out["weight"])+parameters_out["bias"]
result=np.zeros(len(final_output))

for i in range(len(final_output)):
    if sigmoid(final_output[i])>=1/2:
        result[i]=1
    else:
        result[i]=0
        
# ============================================================================
# %% Save the result
# ============================================================================
save_data(passenger,result)



"""
temp=[]
from collections import Counter

def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:

    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if (depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # greedily select the best split according to information gain
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        
        # grow the children that result from the split
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        # parent loss
        parent_entropy = entropy(y)

        # generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # compute the weighted avg. of the loss for the children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # information gain is difference in loss before vs. after split
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    
def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    return X[idxs], y[idxs]

def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common


class RandomForest:
    
    def __init__(self, n_trees=10, min_samples_split=2,
                 max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split,
                max_depth=self.max_depth, n_feats=self.n_feats)
            X_samp, y_samp = bootstrap_sample(X, y)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        temp=tree_preds
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)
    
clf = RandomForest(n_trees=3, max_depth=10)

clf.fit(train, y_train)
preds = clf.predict(train)

accuracy=accuracy_calculate(preds,y_train)
# ============================================================================
# %% Get the final result for testing set
# ============================================================================
result=clf.predict(test)

# ============================================================================
# %% Save the result
# ============================================================================
save_data(passenger,result)

from sklearn import ensemble, preprocessing, metrics

forest = ensemble.RandomForestClassifier(n_estimators = 100)
forest_fit = forest.fit(train, y_train)
preds=forest.predict(train)
accuracy_calculate(preds,y_train)

# 預測
result= forest.predict(test)

save_data(passenger,result)

"""