# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 23:51:23 2019

@author: hp
"""

#Import Libraries
import pandas as pd
#----------------------------------------------------
#reading data
df = pd.read_csv('Concrete_Data_Yeh.csv')

df.describe()
df.isnull()

print('Original df Shape is ' , df.shape)

df.corr(method = 'pearson')

# Rename the columns
df.columns = ['cement', 'slag', 'ash', 'water', 'splast', 'coarse', 'fine', 'age', 'strength']

# Extract the X and y data from the DataFrame 
X = df.drop('strength', axis=1)
y = df['strength']


#split the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=44, shuffle =True)

#Splitted Data
print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)

from sklearn.linear_model import LinearRegression

#Applying Linear Regression Model 

LinearRegressionModel = LinearRegression(fit_intercept=True, normalize=True,copy_X=True,n_jobs=-1)
LinearRegressionModel.fit(X_train, y_train)

#Calculating Details
print('Linear Regression Train Score is : ' , LinearRegressionModel.score(X_train, y_train))
print('Linear Regression Test Score is : ' , LinearRegressionModel.score(X_test, y_test))
print('Linear Regression Coef is : ' , LinearRegressionModel.coef_)
print('Linear Regression intercept is : ' , LinearRegressionModel.intercept_)
print('----------------------------------------------------')

#Calculating Prediction
y_pred = LinearRegressionModel.predict(X_test)
print('Predicted Value for Linear Regression is : ' , y_pred[:10])


import matplotlib.pyplot as plt

# Adjusting Figure Size . .  :  
plt.figure(figsize = (7, 4))
plt.title('Graph')
plt.xlabel('X Data')
plt.ylabel('y Data')
#----------------------------------------------------
#Defining Data
XGraph = X_train[:,0]
yGraph = y_train

#----------------------------------------------------
# Drawing plot graph No 1 . .  :   
plt.plot(XGraph,yGraph,linewidth=1,color= 'r', # can be : b ,g ,y ,c ,m ,k ,w 
         alpha=1,linestyle = 'solid') # can be  : dotted, dashed , dashdot




plt.show()






































