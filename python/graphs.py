#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 12:09:59 2021

@author: psych256lab

"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/home/psych256lab/Downloads/Scripts/area_output.csv")
print("mean:\n",df.groupby(['ImageDate'], as_index=False).agg({'Area': 'mean'}))
print("median:\n",df.groupby(['ImageDate'], as_index=False).agg({'Area': 'median'}))
print("mode:\n",df.groupby(['ImageDate'], as_index=False)['Area'].apply(lambda x: x.mode().iloc[0]))
print("standard deviation:\n",df.groupby('ImageDate').std())


mean = df.groupby(['ImageDate'], as_index=False).agg({'Area': 'mean'})
median = df.groupby(['ImageDate'], as_index=False).agg({'Area': 'median'})
mode = df.groupby(['ImageDate'], as_index=False)['Area'].apply(lambda x: x.mode().iloc[0])
std = df.groupby(['ImageDate'],as_index=False)['Area'].std()


mean['ImageDate'] = pd.to_datetime(mean['ImageDate'], format = '%Y-%m-%d %H:%M:%S')
median['ImageDate'] = pd.to_datetime(median['ImageDate'], format = '%Y-%m-%d %H:%M:%S')
mode['ImageDate'] = pd.to_datetime(mode['ImageDate'], format = '%Y-%m-%d %H:%M:%S')
std['ImageDate'] = pd.to_datetime(std['ImageDate'], format = '%Y-%m-%d %H:%M:%S')

mean['ImageDate'] = mean['ImageDate'].dt.strftime('%Y-%m-%d %H:%M:%S')
median['ImageDate'] = median['ImageDate'].dt.strftime('%Y-%m-%d %H:%M:%S')
mode['ImageDate'] = mode['ImageDate'].dt.strftime('%Y-%m-%d %H:%M:%S')
std['ImageDate'] = std['ImageDate'].dt.strftime('%Y-%m-%d %H:%M:%S')

from datetime import datetime 
import matplotlib.dates as mdates
dates = []
for ts in mean["ImageDate"]:
   local_d = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
   dates.append(local_d)

fig, ax = plt.subplots()
plt.setp( ax.xaxis.get_majorticklabels(), rotation=90)
plt.plot(dates, mean["Area"])
plt.xlabel('Date')
plt.ylabel('Area')
plt.title('MEAN!')
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=350))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.figure(figsize = (50,50))
fig.set_size_inches(30.,18.)
fig.savefig("mean.jpg",dpi=1000)
plt.show()

dates2 = []
for ts in median["ImageDate"]:
   local_d = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
   dates2.append(local_d)
fig, ax = plt.subplots()
plt.setp( ax.xaxis.get_majorticklabels(), rotation=90)
plt.plot(dates2, median["Area"])
plt.xlabel('Date')
plt.ylabel('Area')
plt.title('MEDIAN!')
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=350))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.figure(figsize = (50,50))
fig.set_size_inches(30.,18.)
fig.savefig("median.jpg",dpi=1000)
plt.show()


dates3 = []
for ts in mode["ImageDate"]:
   local_d = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
   dates3.append(local_d)
fig, ax = plt.subplots()
plt.setp( ax.xaxis.get_majorticklabels(), rotation=90)
plt.plot(dates3, mode["Area"])
plt.xlabel('Date')
plt.ylabel('Area')
plt.title('MODE!')
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=350))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.figure(figsize = (50,50))
fig.set_size_inches(30.,18.)
fig.savefig("mode.jpg",dpi=1000)
plt.show()

dates4 = []
for ts in std["ImageDate"]:
   local_d = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
   dates4.append(local_d)
fig, ax = plt.subplots()
plt.setp( ax.xaxis.get_majorticklabels(), rotation=90)
plt.plot(dates4, std["Area"])
plt.xlabel('Date')
plt.ylabel('Area')
plt.title('STD!')
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=120))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.figure(figsize = (50,50))
fig.set_size_inches(30.,18.)
fig.savefig("std.jpg",dpi=1000)
plt.show()
print("Completed")
