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

# Add "Family = SibSp + Parch + 1" Column
train.insert(8, "Family", train["SibSp"] + train["Parch"] + 1)
test.insert(7, "Family", test["SibSp"] + test["Parch"] + 1)

# %%% Fare

# 在還沒填補 Pclass 值之前先算各 Pclass 的 Fare 平均，避免因 Pclass 填補誤差造成 Fare 平均的誤差
Fare_Pclass1_mean = train.dropna(axis = 0, how = "all", subset = ["Fare"]).dropna(axis = 0, how = "all", subset = ["Pclass"])[train["Pclass"] == 1]["Fare"].mean()
Fare_Pclass2_mean = train.dropna(axis = 0, how = "all", subset = ["Fare"]).dropna(axis = 0, how = "all", subset = ["Pclass"])[train["Pclass"] == 2]["Fare"].mean()
Fare_Pclass3_mean = train.dropna(axis = 0, how = "all", subset = ["Fare"]).dropna(axis = 0, how = "all", subset = ["Pclass"])[train["Pclass"] == 3]["Fare"].mean()

# 補 train 的 Fare 缺值
train["Fare"].fillna(-1, inplace = True)
for i in range(len(train)):
    if train["Fare"][i] == -1: # 取相同 Ticket 的 Fare 眾數填補
        train_drop_nan = train.drop(train[train["Fare"] == -1].index)
        if train_drop_nan[train_drop_nan["Ticket"] == train["Ticket"].iloc[i]].shape[0] != 0:
            # Find Mode
            mode_counts = np.bincount(train_drop_nan[train_drop_nan["Ticket"] == train["Ticket"].iloc[i]]["Fare"])
            mode = np.argmax(mode_counts)
            # Fill Mode
            train["Fare"][i] = mode
    if train["Fare"][i] == -1: # 取其 Pclass 等級的 Fare 平均填補
        if train["Pclass"][i] == 1:
            train["Fare"][i] = Fare_Pclass1_mean
        if train["Pclass"][i] == 2:
            train["Fare"][i] = Fare_Pclass2_mean
        if train["Pclass"][i] == 3:
            train["Fare"][i] = Fare_Pclass3_mean

# 補 test 的 Fare 缺值
test["Fare"].fillna(-1, inplace = True)
for i in range(len(test)):
    if test["Fare"][i] == -1: # 取相同 Ticket 的 Fare 眾數填補
        train_drop_nan = train.drop(train[train["Fare"] == -1].index)
        if train_drop_nan[train_drop_nan["Ticket"] == test["Ticket"].iloc[i]].shape[0] != 0:
            # Find Mode
            mode_counts = np.bincount(train_drop_nan[train_drop_nan["Ticket"] == test["Ticket"].iloc[i]]["Fare"])
            mode = np.argmax(mode_counts)
            # Fill Mode
            test["Fare"][i] = mode
    if test["Fare"][i] == -1: # 取其 Pclass 等級的 Fare 平均填補
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

# Fill Train Pclass
for i in range(len(train)):
    if train["Pclass"][i] == -1: # 取相同 Ticket 的 Pclass 眾數填補
        train_drop_nan = train.drop(train[train["Pclass"] == -1].index)
        if train_drop_nan[train_drop_nan["Ticket"] == train["Ticket"].iloc[i]].shape[0] != 0:
            # Find Mode
            mode_counts = np.bincount(train_drop_nan[train_drop_nan["Ticket"] == train["Ticket"].iloc[i]]["Pclass"])
            mode = np.argmax(mode_counts)
            # Fill Mode
            train["Pclass"][i] = mode
    if train["Pclass"][i] == -1: # Cabin 出現 T, A, B, C, D, E 的 Pclass 都填 1
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
    if train["Pclass"][i] == -1: # Fare 超過 train 中 Pclass2 Pclass3 的 Fare 最大值的話 Pclass 填 1
        if train["Pclass"][i] > max_fare_pclass_2and3:
            train["Pclass"][i] = 1
    if train["Pclass"][i] == -1: # 從 Q 上船的話 Pclass 填 3
        if train["Embarked"][i] == "Q":
            train["Pclass"][i] = 3

# Fill Test Pclass
for i in range(len(test)):
    if test["Pclass"][i] == -1: # 取相同 Ticket 的 Pclass 眾數填補
        train_drop_nan = train.drop(train[train["Pclass"] == -1].index)
        if train_drop_nan[train_drop_nan["Ticket"] == test["Ticket"].iloc[i]].shape[0] != 0:
            # Find Mode
            mode_counts = np.bincount(train_drop_nan[train_drop_nan["Ticket"] == test["Ticket"].iloc[i]]["Pclass"])
            mode = np.argmax(mode_counts)
            # Fill Mode
            test["Pclass"][i] = mode
    if test["Pclass"][i] == -1: # Cabin 出現 T, A, B, C, D, E 的 Pclass 都填 1
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
    if test["Pclass"][i] == -1: # Fare 超過 train 中 Pclass2 Pclass3 的 Fare 最大值的話 Pclass 填 1
        if test["Pclass"][i] > max_fare_pclass_2and3:
            test["Pclass"][i] = 1
    if test["Pclass"][i] == -1: # 從 Q 上船的話 Pclass 填 3
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

# Find Age Mean for Each Title
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

# Fill Nan Data With Mean of Certain Title in train data
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

# Fill Nan Data With Mean of Certain Title in test data
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

# Fill Nan Embarked with S
test["Embarked"].fillna(-1, inplace = True)
for i in range(len(test)):
    if test["Embarked"][i] == -1:
        test["Embarked"][i] = "S"

# %%% Age Group

# Train Add New Column "Age Group"
train["Age Group"] = 0
for i in range(len(train)): # 0~12 : Age Group = 1 ; 12~60 : Age Group = 2 ; 60~ : Age Group = 3
    if train["Age"].iloc[i] < 12:
        train["Age Group"].iloc[i] = 1
    elif train["Age"].iloc[i] >= 12 and train["Age"].iloc[i] < 60:
        train["Age Group"].iloc[i] = 2
    elif train["Age"].iloc[i] >= 60:
        train["Age Group"].iloc[i] = 3

# Test Add New Column "Age Group"        
test["Age Group"] = 0
for i in range(len(test)): # 0~12 : Age Group = 1 ; 12~60 : Age Group = 2 ; 60~ : Age Group = 3
    if test["Age"].iloc[i] < 12:
        test["Age Group"].iloc[i] = 1
    elif test["Age"].iloc[i] >= 12 and test["Age"].iloc[i] < 60:
        test["Age Group"].iloc[i] = 2
    elif test["Age"].iloc[i] >= 60:
        test["Age Group"].iloc[i] = 3

# %% More Preprocessing
# 將cabin資料整理成有值與缺失值，有的填yes,沒有填no
def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

train=set_Cabin_type(train)
test=set_Cabin_type(test)

#因子化,選取要用的資料

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

# sigmoid function，最後用y去計算使用sigmoid後的值
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
        print("=== ",(i+1)," time cross validation """)
        for j in range(num):
            if i!=j:
                train_set=np.append(train_set,train_set_split[i],axis=0)
                ytrain_set=np.append(ytrain_set,ytrain_set_split[i],axis=0)
        parameters_out = train_process(train_set, ytrain_set, learning_rate = lr, iterations = 70000)
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

parameters_out = train_process(train, y_train, learning_rate = 0.1, iterations = 70000)
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

# ============================================================================
# %% 以下為貝氏分類器模型
# ============================================================================

'''
    
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
#將survived=1與=0的資料分開
data_train_survived = train[train["Survived"] == 1]
data_train_dead = train[train["Survived"] == 0]
#取age平均與標準差
Age_mean_target_survived = data_train_survived["Age"].mean()
Age_std_target_survived = data_train_survived["Age"].std()
#row_count_target_survived = data_train_survived.shape[0]
Age_mean_target_dead = data_train_dead["Age"].mean()
Age_std_target_dead = data_train_dead["Age"].std()
#row_count_target_dead = data_train_dead.shape[0]
  
#取fare平均與標準差
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
#定義family_i1函數為family=i除以總存活人數，family_i0同理
def family_i1(i):
    f1=(len(train[(train['Family']==i) & (train['Survived']==1)]))/Survived1
    #f0=(len(train[(train['Family']==i) & (train['Survived']==0)]))/Survived0
    return (f1)
def family_i0(i):
    #f1=(len(train[(train['Family']==i) & (train['Survived']==1)]))/Survived1
    f0=(len(train[(train['Family']==i) & (train['Survived']==0)]))/Survived0
    return (f0)
#將family=6以上的歸為一類，因為人數過少
family_atleast6_Survived1=0
family_atleast6_Survived0=0
for i in range(6,12):
    family_atleast6_Survived1 += family_i1(i)
    family_atleast6_Survived0 += family_i0(i)
       
#將各名稱之存活數與死亡數計算
Mr_Survived1=len(train[(train['Title']=='Mr') & (train['Survived']==1)])  
Mr_Survived0=len(train[(train['Title']=='Mr') & (train['Survived']==0)])   
Dr_Survived1=len(train[(train['Title']=='Dr') & (train['Survived']==1)])  
Dr_Survived0=len(train[(train['Title']=='Dr') & (train['Survived']==0)]) 
Master_Survived1=len(train[(train['Title']=='Master') & (train['Survived']==1)])  
Master_Survived0=len(train[(train['Title']=='Master') & (train['Survived']==0)]) 
Miss_Survived1=len(train[(train['Title']=='Miss') & (train['Survived']==1)])  
Miss_Survived0=len(train[(train['Title']=='Miss') & (train['Survived']==0)]) 
Mrs_Survived1=len(train[(train['Title']=='Mrs') & (train['Survived']==1)])  
Mrs_Survived0=len(train[(train['Title']=='Mrs') & (train['Survived']==0)])
    
Rev_Survived1=len(train[(train['Title']=='Rev') & (train['Survived']==1)])  
Rev_Survived0=len(train[(train['Title']=='Rev') & (train['Survived']==0)])
#3個age group之存活數與死亡數
AgeGroup1_Survived1=len(train[(train['Age Group']==1) & (train['Survived']==1)])
AgeGroup1_Survived0=len(train[(train['Age Group']==1) & (train['Survived']==0)])
AgeGroup2_Survived1=len(train[(train['Age Group']==2) & (train['Survived']==1)])
AgeGroup2_Survived0=len(train[(train['Age Group']==2) & (train['Survived']==0)])
AgeGroup3_Survived1=len(train[(train['Age Group']==3) & (train['Survived']==1)])
AgeGroup3_Survived0=len(train[(train['Age Group']==3) & (train['Survived']==0)])
    
Parch0_Survived1=len(train[(train['Parch']==0) & (train['Survived']==1)])
Parch0_Survived0=len(train[(train['Parch']==0) & (train['Survived']==0)])
Parch1_Survived1=len(train[(train['Parch']==1) & (train['Survived']==1)])
Parch1_Survived0=len(train[(train['Parch']==1) & (train['Survived']==0)])
Parch2_Survived1=len(train[(train['Parch']==2) & (train['Survived']==1)])
Parch2_Survived0=len(train[(train['Parch']==2) & (train['Survived']==0)])
Parch3_Survived1=len(train[(train['Parch']==3) & (train['Survived']==1)])
Parch3_Survived0=len(train[(train['Parch']==3) & (train['Survived']==0)])
Parch4_Survived1=len(train[(train['Parch']==4) & (train['Survived']==1)])
Parch4_Survived0=len(train[(train['Parch']==4) & (train['Survived']==0)])
Parch5_Survived1=len(train[(train['Parch']==5) & (train['Survived']==1)])
Parch5_Survived0=len(train[(train['Parch']==5) & (train['Survived']==0)])
Parch6_Survived1=len(train[(train['Parch']==6) & (train['Survived']==1)])
Parch6_Survived0=len(train[(train['Parch']==6) & (train['Survived']==0)])
Parch9_Survived1=len(train[(train['Parch']==9) & (train['Survived']==1)])
Parch9_Survived0=len(train[(train['Parch']==9) & (train['Survived']==0)])
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
def Title_PXCi(data):
    if data=='Mr':
        return [Mr_Survived1/Survived1,Mr_Survived0/Survived0]
    elif data=='Dr':
        return [Dr_Survived1/Survived1,Dr_Survived0/Survived0]
    elif data=='Master':
        return [Master_Survived1/Survived1,Master_Survived0/Survived0]
    elif data=='Miss':
        return [Miss_Survived1/Survived1,Miss_Survived0/Survived0]
    elif data=='Mrs':
        return [Mrs_Survived1/Survived1,Mrs_Survived0/Survived0]
    else:
        return [1,1]
def AgeGroup_PXCi(data):
    if data==1:
        return [AgeGroup1_Survived1/Survived1,AgeGroup1_Survived0/Survived0]
    elif data==2:
        return [AgeGroup2_Survived1/Survived1,AgeGroup2_Survived0/Survived0]
    elif data==3:
        return [AgeGroup3_Survived1/Survived1,AgeGroup3_Survived0/Survived0]
    else:
        return [1,1]
    
def Parch_PXCi(data):
    if data==0:
        return [Parch0_Survived1/Survived1,Parch0_Survived0/Survived0]
    elif data==1:
        return [Parch1_Survived1/Survived1,Parch1_Survived0/Survived0]
    elif data==2:
        return [Parch2_Survived1/Survived1,Parch2_Survived0/Survived0]
    elif data==3:
        return [Parch3_Survived1/Survived1,Parch3_Survived0/Survived0]
    elif data==4:
        return [Parch4_Survived1/Survived1,Parch4_Survived0/Survived0]
    elif data==5:
        return [Parch5_Survived1/Survived1,Parch5_Survived0/Survived0]
    elif data==6:
        return [Parch6_Survived1/Survived1,Parch6_Survived0/Survived0]
    elif data==9:
        return [Parch9_Survived1/Survived1,Parch9_Survived0/Survived0]
    else:
        return [1,1]
#拿來放feature之P(X|Ci)
z1=np.ones([len(test),10])
z0=np.ones([len(test),10])
#套用函數並將P(X|Ci)存至z1,z2    
for i in range(len(test)):
    [z1[i,0],z0[i,0]]=Pclass_PXCi(test.iloc[i].at["Pclass"])
    [z1[i,1],z0[i,1]]=Sex_PXCi(test.iloc[i].at["Sex"])
    [z1[i,2],z0[i,2]]=Age_PXCi(test.iloc[i].at["Age"])
    [z1[i,3],z0[i,3]]=Fare_PXCi(test.iloc[i].at["Fare"])
    [z1[i,4],z0[i,4]]=Embarked_PXCi(test.iloc[i].at["Embarked"])
    #[z1[i,5],z0[i,5]]=Family_PXCi(test.iloc[i].at["Family"])
    #[z1[i,6],z0[i,6]]=Title_PXCi(test.iloc[i].at["Title"])
    [z1[i,7],z0[i,7]]=AgeGroup_PXCi(test.iloc[i].at["Age Group"])
    [z1[i,8],z0[i,8]]=Parch_PXCi(test.iloc[i].at["Parch"])
    
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
    
Passenger=test['PassengerId'].values
output=np.transpose(np.vstack((Passenger,predict)))
output=output.astype(int)
output0=pd.DataFrame(output, columns=['PassengerId','Survived']) 
print (output0)
output0.to_csv('result_psagpaafe.csv',index=False)
'''