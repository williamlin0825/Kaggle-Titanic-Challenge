# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 20:45:36 2021

@author: user
"""

#%%

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