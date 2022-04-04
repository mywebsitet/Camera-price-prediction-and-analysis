# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 19:59:18 2021

@author: SUBHAM
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white',color_codes=True)
sns.set(font_scale=1.5)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score,mean_squared_error
from math import sqrt


Camera_data = pd.read_csv("C:/Users/SUBHAM/Desktop/Python Projects/Python Project 1/Project 1/Camera.csv")
Camera_data.head()

#percentage of blank values 
Camera_data.isnull().sum()/Camera_data.count().sum()

# statistical summary of the data

Camera_data.describe()

# Replace all the blank values with NaN

Camera_data = Camera_data.replace(0,np.nan)

# Now replace all the Blank values with the column median.

Camera_data = Camera_data.fillna(Camera_data.median())

# Add a new column “Discounted_Price” in which give a discount of
# 5% in the Price column.

Camera_data['Discounted_Price'] = Camera_data['Price']*.05

Camera_data[['Price','Discounted_Price']]

# Drop the columns Zoom Tele & Macro Focus range

Camera_data = Camera_data.drop(['Zoom tele (T)','Macro focus range'],axis = 1)

#Replace the Model Name “Agfa ePhoto CL50” with “Agfa ePhoto CL250”
Camera_data['Model'] = Camera_data['Model'].replace("Agfa ePhoto CL50","Agfa ePhoto CL250")

# Rename the column name from Release Date to Release Year.

Camera_data=Camera_data.rename(columns={'Release date':'Release Year'},inplace = False)

Camera_data.head()
# moCamera_datast expensive Camera

Camera_data[Camera_data['Price']==np.max(Camera_data['Price'])]['Model']

# Which camera have the least weight?
Camera_data[Camera_data['Weight (inc. batteries)']==np.max(Camera_data['Weight (inc. batteries)'])]['Model']

# Group the data on the basis of their release year.
groupby_release_year=Camera_data.groupby(['Release Year'])

print(groupby_release_year.first())

Camera_data.groupby(['Release Year']).count()

# Extract the Name, Storage Include, Price, Disounted_Price & Dimensions columns

Camera_data[['Model','Storage included','Price','Discounted_Price','Dimensions']]

# Extract the records for the cameras released in the year 2005 & 2006

data_05_06=Camera_data['Release Year'].isin([2005,2006])

print(Camera_data[data_05_06])

# Find out 2007’s expensive & Cheapest Camera

Camera_data[Camera_data['Price']==np.max(Camera_data['Price']) and Camera_data['Release date']=='2007']['Model']

Camera_data[Camera_data['Price']==np.max(Camera_data['Price'])]
Camera_data[Camera_data['Release date']==2007]

Camera_data.head(10)
Camera_data.info()

Camera_data['Model'].value_counts()

# Correlation Plot

plt.figure(figsize=(10,10))
sns.heatmap(Camera_data.corr())
plt.show()


#Split the original dataset into training and testing datasets for the model

X = Camera_data.drop(['Model','Price'],axis=1)
Y = Camera_data['Price']

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=1)

#Linear Regression Model creation 

model = LinearRegression()

model.fit(x_train,y_train)

print(model.coef_)
model.intercept_

y_pred = model.predict(x_test)

print(sqrt(mean_squared_error(y_test,y_pred)))

model.score(x_test, y_test)

## Ridge Regression Model Creation
from sklearn.linear_model import Ridge
Ridge_model = Ridge(alpha=0.001,normalize=True)
Ridge_model.fit(x_train,y_train)
print(sqrt(mean_squared_error(y_train,Ridge_model.predict(x_train))))
print(sqrt(mean_squared_error(y_test,Ridge_model.predict(x_test))))
print("R2 Value/Coefficient of Determination : {}",format(Ridge_model.score(x_test,y_test)))

## Lasso Regression Model 
from sklearn.linear_model import Lasso
Lesso_reg = Lasso(alpha=0.001,normalize=True)
Lesso_reg.fit(x_train,y_train)
print(sqrt(mean_squared_error(y_train,Lesso_reg.predict(x_train))))
print(sqrt(mean_squared_error(y_test,Lesso_reg.predict(x_test))))
print("R2 value/coefficient of Determination: {}",format(Lesso_reg.score(x_test,y_test)))

## ElasticNet Regression model
from sklearn.linear_model import ElasticNet
Ele_reg = Lasso(alpha=0.001,normalize=True)
Ele_reg.fit(x_train,y_train)
print(sqrt(mean_squared_error(y_train,Ele_reg.predict(x_train))))
print(sqrt(mean_squared_error(y_test,Ele_reg.predict(x_test))))
print("R2 value/coefficient of Determination: {}",format(Ele_reg.score(x_test,y_test)))




