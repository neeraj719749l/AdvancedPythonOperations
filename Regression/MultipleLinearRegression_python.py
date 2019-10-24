#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: rajneeshsharma
# Several indepedent variables
"""

#Importing the liberaries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
ds = pd.read_csv('50_startups.csv')
X = ds.iloc[:, :-1].values
Y = ds.iloc[:, 4].values

#Encoding categorical variable
#Encoding the independent variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 3] = labelEncoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy varible trap
X = X[:, 1:]

#Building the optimal model using backward elimination
import statsmodels.regression.linear_model as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis=1)
X_opt = X[:, [0,1,2,3,4,5]] 
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1,3,4,5]] 
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5]] 
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,5]] 
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

# Removing columns furhter leads to decrease in r-squared value 
# R&D is the most important feature contributing towards profit

#Splitting the dataset into training and testing datasets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_opt, Y, test_size=0.2, random_state=0)

#Fitting Multiple linear regression to the training set
from sklearn.linear_model import LinearRegression as lr
regressor = lr()
regressor.fit(X_train, Y_train)

#Predicting the test set results
y_pred = regressor.predict(X_test)

