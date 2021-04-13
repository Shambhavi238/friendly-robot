# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:11:51 2019

@author: ankit
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data=pd.read_csv('nifty50.csv')
print(data.head(10))
real_x=data['date'].values;
real_y=data['close'].values;
real_x=real_x.reshape(-1,1)
real_y=real_y.reshape(-1,1)
print(real_x)
print(real_y)
training_x,testing_x,training_y,testing_y=train_test_split(real_x,real_y,test_size=0.3,random_state=0)
Lin=LinearRegression()
Lin.fit(training_x,training_y)
Pred_y=Lin.predict(testing_x)
print("Testing data=",testing_y[1])
print("Pred data=",Pred_y[1])
plt.scatter(training_x,training_y,color='green')
plt.plot(training_x,Lin.predict(training_x),color='blue')
plt.xlabel('date')
plt.ylabel('capital')
plt.title('stock market prediction plot')
plt.show()