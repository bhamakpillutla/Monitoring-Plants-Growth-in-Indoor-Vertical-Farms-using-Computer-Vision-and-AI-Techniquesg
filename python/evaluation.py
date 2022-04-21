#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 19:14:04 2022

@author: Bhama Pillutla

Compare ground truth areas with detected areas and calculate R2 score and MSE.

"""
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score,mean_squared_error

# line 1 points

df = pd.read_csv("/home/psych256lab/Documents/groundtruths_test.csv")
scaler = MinMaxScaler()

df[["ComputedArea"]] = scaler.fit_transform(df[["ComputedArea"]])
df[["ActualArea"]] = scaler.fit_transform(df[["ActualArea"]])
# plotting the line 1 points 
plt.plot(df["ActualArea"], label = "groundtruth")

plt.plot(df["ComputedArea"], label = "computed")

plt.xlabel('x -  - No. of Leaves Compared')
# Set the y axis label of the current axis.
plt.ylabel('y - axis')
plt.legend()
# Set a title of the curr
plt.show()


df.fillna(0, inplace=True)
r2 = r2_score(df["ComputedArea"], df["ActualArea"])
print('r2 score:', r2)
#df.plot(x='GroundTruth_Area', y='ComputedArea')
# Mean Squared Error
MSE = mean_squared_error(df["ActualArea"],df["ComputedArea"])
print("MSE:",MSE)



