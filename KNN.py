#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 00:34:53 2023

@author: myyntiimac
"""

##KNN Classifier
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

#Training the KNN classifier model with training data
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
# Train the KNN Classifier
knn.fit(X_train, y_train)
# Make predictions on the test set
y_pred = knn.predict(X_test)

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
bias = knn.score(X_train, y_train)
bias
#check the variance
variance = knn.score(X_test, y_test)
variance
#Future prediction
df1=pd.read_csv("/Users/myyntiimac/Desktop/50 observation dataset for future prediction.csv")
df1.head()
df1.shape
df.shape
#copy df1 for assignn the prediction value after prediction
FD=df1.copy()
#Then the defined the dataset for futyre prediction
X1= df1.iloc[:,[2,3]].values
X1
#scalize the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
C= sc.fit_transform(X1)
C
y_pred2=knn.predict(C)
y_pred2
FD['predict'] = pd.Series(y_pred2)
FD
FD.to_csv("KNNpred.csv")