"""
Spyder Editor
This is a temporary script file.
"""
import matplotlib
import numpy as np
import pandas as pd
from math import sqrt, exp, pi
from matplotlib import pyplot as plt

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# %% preprocessing
# ============================================================================
""" Fill Pclass Method 1
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

"""Fill Pclass Method 2"""
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
print(train)

"""
# %% predict multivariate
# ============================================================================
# split train for train and test
train_train = train.sample(frac = 0.7, axis = 0)
train_test = train.append(train_train)
train_test = train_test.drop_duplicates(list(train), keep = False)
train_train_survived = train_train[train_train["Survived"] == 1]
train_train_dead = train_train[train_train["Survived"] == 0]
def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent
def mean(numbers):
	return sum(numbers)/float(len(numbers))
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)
def calculate_probability_multivariate(d, x, mean, cov):
	temp=(x-mean).transpose()
	temp=temp.dot(np.linalg.inv(cov))
	exponent = exp(-(temp.dot(x-mean))/2)
	return (1 / (sqrt(2 * pi)**d * np.linalg.det(cov))) * exponent
def calculate_probability_log(x, mean, stdev):
	exponent = exp(-((log(x)-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) *x * stdev)) * exponent
"""

# =============================================================================
# Trying PCA
# =============================================================================
from sklearn.decomposition import PCA

train_no_null=train.drop(['PassengerId','Age','Name','Ticket'],axis=1)    #create a temporary testing set without Nan
train_no_null["Cabin"].fillna(0, inplace=True)
train_no_null.loc[train_no_null["Cabin"] == 0, "Cabin"] = 0
train_no_null.loc[train_no_null["Cabin"] != 0, "Cabin"] = 1
train_no_null["Fare"].fillna(0, inplace=True)
train_no_null['Embarked'] = pd.factorize(train_no_null['Embarked'])[0]
train_no_null['Sex'] = pd.factorize(train_no_null['Sex'])[0]

alive=train_no_null.loc[:,'Survived']
train_no_null=train_no_null.drop(['Survived'],axis=1)               #get Survived as Y 
Y=alive.to_numpy()
print(train_no_null)

pca_training = PCA().fit(train_no_null)
plt.figure()
accumulate_pca=np.cumsum(pca_training.explained_variance_ratio_)
plt.plot(accumulate_pca)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

pca_training3 = PCA(n_components = 3)       #PCA using 3 components
pca_training3_feature = pca_training3.fit_transform(train_no_null)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('PCA(component = 3)')
ax.scatter(pca_training3_feature[:,0],pca_training3_feature[:,1], 
           pca_training3_feature[:,2], c=Y)
plt.show()

pca_training2 = PCA(n_components = 2)       #PCA using 2 components
pca_training2_feature = pca_training2.fit_transform(train_no_null)
plt.figure()
plt.title('PCA(component = 2)')
plt.scatter(pca_training2_feature[:,0],pca_training2_feature[:,1],c=Y)
plt.show()

# =============================================================================
# Trying LDA
# =============================================================================
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda_train = lda.fit(train_no_null, Y)

lda1 = LinearDiscriminantAnalysis(n_components=1)
lda1_train = lda1.fit_transform(train_no_null, Y)
plt.figure()
plt.title('LDA(component = 1)')
plt.scatter(lda1_train[:,0],np.zeros([1,len(lda1_train)]),c=Y)
plt.show()