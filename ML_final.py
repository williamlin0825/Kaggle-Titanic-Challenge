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
# Import library
from IPython import get_ipython
get_ipython().magic("clear")
get_ipython().magic("reset -f")
import numpy as np
import pandas as pd
from math import sqrt, exp, pi
from matplotlib import pyplot as plt
import random 

# Import Dataset
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

# 將cabin資料整理成有值與缺失值，有的填yes,沒有填no
def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

train=set_Cabin_type(train)
test=set_Cabin_type(test)

#因子化,選取藥用的資料

d_Sex=pd.get_dummies(train['Sex'],prefix='Sex')
d_Pclass=pd.get_dummies(train['Pclass'],prefix='Pclass')
d_Embarked=pd.get_dummies(train['Embarked'],prefix='Embarked')
d_Cabin=pd.get_dummies(train['Cabin'],prefix='Cabin')
train=pd.concat([train,d_Sex,d_Pclass],axis=1)
train_temp=train
train.drop(['Full Name','Last Name','Title','First Name','Embarked','Cabin','Ticket','Family','SibSp','Sex','Pclass','Fare'],axis=1,inplace=True)


d_Sex1=pd.get_dummies(test['Sex'],prefix='Sex')
d_Pclass1=pd.get_dummies(test['Pclass'],prefix='Pclass')
d_Embarked1=pd.get_dummies(test['Embarked'],prefix='Embarked')
d_Cabin1=pd.get_dummies(test['Cabin'],prefix='Cabin')
test=pd.concat([test,d_Sex1,d_Pclass1],axis=1)
test_temp=test
test.drop(['Full Name','Last Name','Title','First Name','Embarked','Cabin','Ticket','Family','SibSp','Sex','Pclass',"Fare",],axis=1,inplace=True)



#standardize Age and Fare

train_normalize_max_age=((train['Age']-train['Age'].mean())/train['Age'].std()).max()
train_normalize_min_age=((train['Age']-train['Age'].mean())/train['Age'].std()).min()
test['Age'] = 2*(((test['Age']-train['Age'].mean())/train['Age'].std())-train_normalize_min_age)/(train_normalize_max_age-train_normalize_min_age)-1
train['Age'] = 2*(((train['Age']-train['Age'].mean())/train['Age'].std())-train_normalize_min_age)/(train_normalize_max_age-train_normalize_min_age)-1

"""
train_normalize_max_fare=((train['Fare']-train['Fare'].mean())/train['Fare'].std()).max()
train_normalize_min_fare=((train['Fare']-train['Fare'].mean())/train['Fare'].std()).min()
test['Fare'] = 2*(((test['Fare']-train['Fare'].mean())/train['Fare'].std())-train_normalize_min_fare)/(train_normalize_max_fare-train_normalize_min_fare)-1
train['Fare'] = 2*(((train['Fare']-train['Fare'].mean())/train['Fare'].std())-train_normalize_min_fare)/(train_normalize_max_fare-train_normalize_min_fare)-1
"""





#get the survived array
y_train=train.loc[:,['Survived']]

passenger=test.loc[:,['PassengerId']]

#drop the unused data of training and testing(ID與生存率不該出現在要拿來訓練的資料中)
train.drop(['PassengerId','Survived'], axis=1,inplace=True)
test.drop(['PassengerId'], axis=1,inplace=True)


#change to array to compute,要把dataframe轉成array的形式方便訓練模型
y_train=y_train.to_numpy()
test=test.to_numpy()
train=train.to_numpy()
passenger=passenger.to_numpy()

#reshape y
y_train=np.squeeze(y_train)

# ============================================================================
# %% split data set to cross validation
# ============================================================================
def train_split(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

# ============================================================================
# %% Additional function
# ============================================================================

#最後拿來計算準確度的函數
def accuracy_calculate(preds,y):
    count=0
    for i in range(len(preds)):
        if y_train[i]==preds[i]:
            count+=1
    accuracy=count/len(preds)
    return accuracy

#最後拿來存取資料的函數
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

# igmoid function，最後用y去計算使用sigmoid後的值
def sigmoid(input):    
    output = 1 / (1 + np.exp(-input))
    return output

# 存取weight和bias的變化過程
weight_history=[]
bias_history=[]

# 訓練過程的function
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

# 用init_parameter同時存取weight與bias
init_parameters = {} 
init_parameters["weight"] = np.random.randn(train.shape[1])
init_parameters["bias"] = 0

# 把訓練與存取參數的過程整理成train_process
def train_process(x, y, learning_rate,iterations):
    parameters_out = optimize(x, y, learning_rate, iterations ,init_parameters)
    return parameters_out

# ============================================================================
# %% Cross validation 
# ============================================================================

def cross_validation(x,y,num,lr):
    train_set_split=train_split(x,num)
    ytrain_set_split=train_split(y,num)
    validation_accuracy=[]
    output_history=[]
    for i in range(num):
        train_set=np.empty((0, train.shape[1]))
        ytrain_set=[]
        for j in range(num):
            if i!=j:
                train_set=np.append(train_set,train_set_split[i],axis=0)
                ytrain_set=np.append(ytrain_set,ytrain_set_split[i],axis=0)
        parameters_out = train_process(train_set, ytrain_set, learning_rate = lr, iterations = 50000)
        output_values=np.dot(train_set_split[i],parameters_out["weight"])+parameters_out["bias"]
        prediction=np.zeros(len(output_values))
        for k in range(len(output_values)):
            if sigmoid(output_values[k])>=1/2:
                prediction[k]=1
            else:
                prediction[k]=0

        #accuracy=accuracy_calculate(prediction,ytrain_set_split[i])
        #validation_accuracy=np.append(validation_accuracy,accuracy)
        output_history=np.append(output_history,prediction)
    average_accuracy=accuracy_calculate(output_history,y_train)
    return average_accuracy

average_accuracy=cross_validation(train,y_train,10,0.1)

# ============================================================================
# %% Training and Testing
# ============================================================================

# 將整理好的training資料進行訓練

parameters_out = train_process(train, y_train, learning_rate = 0.1, iterations = 50000)
output_values=np.dot(train,parameters_out["weight"])+parameters_out["bias"]
prediction=np.zeros(len(output_values))

# 把計算出的y帶入sigmoid function，大於0.5填1(生存)，小於0.5填0(死亡)
for i in range(len(output_values)):
    if sigmoid(output_values[i])>=1/2:
        prediction[i]=1
    else:
        prediction[i]=0
        
# 計算training set的準確率
print("Cross validation average accuracy=",average_accuracy*100,"%")
training_accuracy=accuracy_calculate(prediction,y_train)
print("Training set accuracy=",training_accuracy*100," %")

# ============================================================================
# %% Get the final result for testing set
# ============================================================================

# 將testing set丟入訓練好的模型中，得到想要的猜測結果

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

#存取最後的資料
save_data(passenger,result)