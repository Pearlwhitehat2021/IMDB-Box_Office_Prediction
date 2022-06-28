# -*- coding: utf-8 -*-
"""
Created on Wed May  4 21:56:03 2022

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 21:32:58 2022

@author: pavilion
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, 0:1]. values
Y = dataset.iloc[:, 1]. values
X_train, X_test, Y_train, Y_test=train_test_split(X,Y, test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

Y_predict = regressor.predict(X_test)

plt.scatter(X_train, Y_train, color='Red')
plt.plot(X_train, regressor.predict(X_train), color='Blue')
plt.title('IMDB Movie Prdiction Analysis')
plt.xlabel('Release Year')
plt.ylabel('Movie Rating')



from sklearn import metrics
print(metrics.mean_absolute_error(Y_test, Y_predict))
print(metrics.mean_squared_error(Y_test, Y_predict))
print(np.sqrt(metrics.mean_squared_error(Y_test, Y_predict)))


