"""
Spyder Editor
This is a temporary script file.
"""
import matplotlib
import numpy as np
import pandas as pd
from math import sqrt, exp, pi
from matplotlib import pyplot as plt
import random

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# %% preprocessing
# ============================================================================
#fill cabin
def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

train=set_Cabin_type(train)
test=set_Cabin_type(test)

# replace Nan in Cabin row with 0
train["Pclass"].fillna(0, inplace=True)
test["Pclass"].fillna(0, inplace=True)

# Refer to Cabin & Fill Pclass
for i in range(len(train)):
    if train["Pclass"][i] == 0:
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

# Refer to Fare & Fill Pclass
max_fare_pclass_2and3 = train[train["Pclass"] > 1].max(skipna=True)["Fare"]
for i in range(len(train)):
    if train["Pclass"][i] == 0:
        if train["Pclass"][i] > max_fare_pclass_2and3:
            # Can We Use Chebyshev"s Theorem?
            train["Pclass"][i] = 1
    if train["Pclass"][i] == 0:
        if train["Embarked"][i] == "Q":
            train["Pclass"][i] = 3
            
# Refer to Cabin & Fill Pclass
for i in range(len(test)):
    if test["Pclass"][i] == 0:
        if "T" in str(test["Cabin"][i]):
            test["Pclass"][i] = 1
        if "A" in str(test["Cabin"][i]):
            test["Pclass"][i] = 1
        if "B" in str(test["Cabin"][i]):
            test["Pclass"][i] = 1
        if "C" in str(test["Cabin"][i]):
            test["Pclass"][i] = 1
        if "D" in str(test["Cabin"][i]):
            test["Pclass"][i] = 1
        if "E" in str(test["Cabin"][i]):
            test["Pclass"][i] = 1

# Refer to Fare & Fill Pclass
for i in range(len(test)):
    if test["Pclass"][i] == 0:
        if test["Pclass"][i] > max_fare_pclass_2and3:
            # Can We Use Chebyshev"s Theorem?
            test["Pclass"][i] = 1
    if test["Pclass"][i] == 0:
        if test["Embarked"][i] == "Q":
            test["Pclass"][i] = 3

#replace the 0 in Pcalss to 3
train['Pclass'] = train['Pclass'].replace(0,3)
test['Pclass'] = test['Pclass'].replace(0,3)

#fill Nan Age
mean_age=train['Age'].mean()
train['Age'].fillna(value=train['Age'].mean(), inplace=True)
test['Age'].fillna(value=test['Age'].mean(), inplace=True)

#fill Nan Fare
mean_Fare=train['Fare'].mean()
train['Fare'].fillna(value=train['Fare'].mean(), inplace=True)

#fill Nan Embarked
test['Embarked'].fillna(train['Embarked'].mode().iloc[0], inplace=True)

#Numeralization
d_Sex=pd.get_dummies(train['Sex'],prefix='Sex')
d_Pclass=pd.get_dummies(train['Pclass'],prefix='Pclass')
d_Embarked=pd.get_dummies(train['Embarked'],prefix='Embarked')
d_Cabin=pd.get_dummies(train['Cabin'],prefix='Cabin')
train=pd.concat([train,d_Sex,d_Pclass,d_Embarked,d_Cabin],axis=1)
train.drop(['Name','Sex','Pclass','Embarked','Cabin','Ticket'],axis=1,inplace=True)

d_Sex=pd.get_dummies(test['Sex'],prefix='Sex')
d_Pclass=pd.get_dummies(test['Pclass'],prefix='Pclass')
d_Embarked=pd.get_dummies(test['Embarked'],prefix='Embarked')
d_Cabin=pd.get_dummies(test['Cabin'],prefix='Cabin')
test=pd.concat([test,d_Sex,d_Pclass,d_Embarked,d_Cabin],axis=1)
test.drop(['Name','Sex','Pclass','Embarked','Cabin','Ticket'],axis=1,inplace=True)



#Normalize Age and Fare
test[['Age','Fare']] -= train[['Age','Fare']].min()
test[['Age','Fare']] /= train[['Age','Fare']].max()

train[['Age','Fare']] -= train[['Age','Fare']].min()
train[['Age','Fare']] /= train[['Age','Fare']].max()




#Split Training set value to train_of_train and train_of_test
train_of_train=train.sample(frac=0.7)
test_of_train=train.drop(train_of_train.index)
print (train_of_train,'\n',test_of_train)

y_train_of_train=train_of_train.loc[:,['Survived']]
y_test_of_train=test_of_train.loc[:,['Survived']]
y_train=train.loc[:,['Survived']]

test_of_train.drop(['PassengerId','Survived'], axis=1,inplace=True)
train_of_train.drop(['PassengerId','Survived'], axis=1,inplace=True)
train.drop(['PassengerId','Survived'], axis=1,inplace=True)
passenger=test.loc[:,['PassengerId']]

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

print(train)
print(test)

# ============================================================================
# %% Logistic Regression
# ============================================================================
def sigmoid(input):    
    output = 1 / (1 + np.exp(-input))
    return output


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
        bias -= learning_rate * db 
        if (i+1)%100==0:
            print('=== Iteration: %d ===' %(i+1))
            print('Training loss: %.4f' %loss)
    
    parameters["weight"] = weight
    parameters["bias"] = bias
    return parameters

init_parameters = {} 
init_parameters["weight"] = np.zeros(train_of_train.shape[1])
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
count=0
for i in range(len(output_values)):
    if y_train[i]==prediction[i]:
        count+=1
accuracy=count/len(output_values)
print("The Accuracy is", accuracy*100,"%")

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
result=np.reshape(result,(len(result),1))
final_result=np.concatenate((passenger, result),axis=1)
final_result=final_result.astype(int)
dataframe=pd.DataFrame(final_result, columns=['PassengerId','Survived']) 
print (dataframe)
dataframe.to_csv('result.csv',index=False)