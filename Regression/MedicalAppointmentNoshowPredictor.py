p#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 22:48:22 2019

@author: Rajneesh Sharma
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the dataset
dataset = pd.read_excel('Medical-Appointment-noshow-data.xlsx')
dataset.iloc[:, [5]] = dataset.iloc[:, [5]].replace(0, np.NaN)
dataset.iloc[:, [5]] = dataset.iloc[:, [5]].replace(-1, np.NaN) 
dataset.iloc[:, [5]] = dataset.iloc[:, [5]].replace(115, np.NaN) 

#to count the types of values
no_show = dataset["No-show"].value_counts()
handcap = dataset["Handcap"].value_counts()
neighbourhood = dataset["Neighbourhood"].value_counts()

#to count the no of values
age = (dataset["Age"] == 100).sum()

#Feature Selection
X = dataset.iloc[:, 5:13].values
y = dataset.iloc[:, -1].values

#Missing data handling
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
X[:, [0]] = imputer.fit_transform(X[:, [0]])

#To check the no of rows with 0 value in each column
(X == 0).astype(int).sum(axis = 0)

# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 1] = labelEncoder_X.fit_transform(X[:, 1])

# to convert object array to dataset
X = pd.DataFrame(X)

#Using one hot encoder
OneHotEncoder_X = OneHotEncoder(categorical_features = [1, 6])
X = OneHotEncoder_X.fit_transform(X).toarray()

#Label encoder on y
labelEncoder_Y = LabelEncoder()
y = labelEncoder_Y.fit_transform(y)

# Searching for the optimal features
import statsmodels.formula.api as sm
X_opt = X[:, 0:89]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, 1:89]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#Scaling up the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_opt)
rescaledX2 = scaler.transform(X_opt)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(rescaledX2, y, test_size = 0.25, random_state = 0)

#Fitting logistic regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predicting a test set result
y_pred = classifier.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred)