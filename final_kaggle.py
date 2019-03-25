# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 18:25:35 2018

@author: Debaditya
"""
 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')

dataset=[train_data,test_data]

train_data.describe()

train_data.info()

train_data[['Sex','Survived']].groupby(['Sex']).mean()

train_data[['Pclass','Survived']].groupby(['Pclass']).mean()

for i in dataset:
    print(i['Embarked'].mode())
    
for i in dataset:
    i['Embarked'].fillna('S',inplace=True)
    
train_data.info()
test_data.info()

train_data[['Embarked','Survived']].groupby(['Embarked']).mean()

train_data.info()
test_data.info()

for i in dataset:
    i['family']=i['SibSp']+i['Parch']+1

train_data[['family','Survived']].groupby(['family']).mean()

train_data.info()
test_data.info()

for i in dataset:
    print(i['Age'].mean())
for i in dataset:
    print(i['Age'].median())
for i in dataset:
    i['Age'].fillna(i['Age'].median(),inplace=True)

train_data.info()
test_data.info()


train_data['age_range']=pd.qcut(train_data['Age'],4)

train_data[['age_range','Survived']].groupby(['age_range']).mean()

train_data.info()

train_data['fare_range']=pd.qcut(train_data['Fare'],4)

train_data[['fare_range','Survived']].groupby(['fare_range']).mean()

train_data=train_data.drop(train_data[['age_range','fare_range']],axis=1)

train_data.drop(train_data[['Name','SibSp','Parch','Cabin','Ticket']],axis=1,inplace=True)
test_data.drop(test_data[['Name','SibSp','Parch','Cabin','Ticket']],axis=1,inplace=True)

train_data.info()
test_data.info()

test_data['Fare'].fillna(test_data['Fare'].median(),inplace=True)

test_data.info()

'''cannot do using for loop as i will come out changing and keeping the test_data
but it will not change the train_data or test_data itself
its the i which is changing
but we wanna change the train_data and test_data
as both of them will be used in train,test sets and classifiers
for i in dataset: 
    i = pd.get_dummies(i, columns = ['Sex'],drop_first=True)
See for yourself
'''
train_data = pd.get_dummies(train_data, columns = ['Sex'],drop_first=True)
train_data = pd.get_dummies(train_data, columns = ['Embarked'],drop_first=True)

test_data = pd.get_dummies(test_data, columns = ['Sex'],drop_first=True)
test_data = pd.get_dummies(test_data, columns = ['Embarked'],drop_first=True)

train_data.info()
test_data.info()

X_train=train_data.iloc[:,2:9].values
Y_train=train_data.iloc[:,1].values

X_test=test_data.iloc[:,1:9].values

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.fit_transform(X_test)

from sklearn.svm import SVC
svm=SVC(C=6,kernel='rbf',gamma=0.1328,random_state=0)
svm.fit(X_train,Y_train)

Y_pred1=svm.predict(X_test)

from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=svm,X=X_train,y=Y_train,cv=10)
accuracies.mean()

from sklearn.model_selection import GridSearchCV
parameter=[{'C':[1,2,3,4,5,6,7],'gamma':[0.1428,0.1228,0.1328,0.1528,0.1628,0.1728]}]
grid_search=GridSearchCV(estimator=svm,param_grid=parameter,scoring='accuracy',cv=10)
grid_search=grid_search.fit(X_train,Y_train)
accuarcy=grid_search.best_score_
best_param=grid_search.best_params_

test_data['Survived']=Y_pred1
answer=test_data[['PassengerId','Survived']]
answer.to_csv('titanic_submission9.csv', index = False)
