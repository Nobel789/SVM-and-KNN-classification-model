#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 13:00:40 2023

@author: myyntiimac
"""

#Support vector classificatio
#import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv("/Users/myyntiimac/Desktop/logit classification.csv")
df.head()
#define variable
X = df.iloc[:,[2,3]].values
X
Y = df.iloc[:,-1]
#spliting the variable
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, Y, test_size = 0.20,random_state = 0 )

#scalized the test variable
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred 

#Check the confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score 
ac = accuracy_score(y_test, y_pred)
ac 
#check the vclassification report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
cr
#check the bias
bias = classifier.score(X_train, y_train)
bias
#check the variance
variance = classifier.score(X_test, y_test)
variance

#future prediction
df1=pd.read_csv("/Users/myyntiimac/Desktop/50 observation dataset for future prediction.csv")
df1.head()

df1.shape
df.shape
#copy df1 for assignn the prediction value after prediction
FD=df1.copy()
FD.head()
#Then the defined the dataset for futyre prediction
X1= df1.iloc[:,[2,3]].values
X1
#scalize the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
C= sc.fit_transform(X1)
C

y_pred2=classifier.predict(C)
y_pred2

FD['predict'] = pd.Series(y_pred2)
FD
FD.to_csv("Svmpred.csv")

####
#aply in console get file location
#import os

#os.getcwd()
# some important inforamtion foe parameteter tuning, by default c=1, it regularize parameter
#kernel(linear,rbf,,sigmoid,prcomputed),last three is non linear, use for not linearly seperable data
#Degree=3 for linear
#gama- is kernel coefficient, (auto and scale )by default scale
#kernel _it helps to classify data by changing 1d to  2d dimensional space , 2D to 3D, so that its easly separate nonlinear data




