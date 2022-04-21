#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 11:15:49 2021

@author: psych256lab
"""
import pandas as pd
df = pd.read_csv("/home/psych256lab/Downloads/Scripts/area_output(daytrack).csv")

#df.drop_duplicates(subset=['Day', 'Type'], keep='first')
#df.groupby(['col5', 'col2'])
df = df.groupby(['Day','Time','LeafIndex'], group_keys=False).apply(lambda x: x.loc[x.Area.idxmax()])
df = df.drop(columns=['Day', 'Time','Unnamed: 0','LeafIndex'])
print(df.groupby(['Day','Time','LeafIndex'])['Area'])
if df.groupby(['Day','Time','LeafIndex'])['Area'] < df.groupby(['Day','Time','LeafIndex'])['Area'].mean():
   df.groupby(['Day','Time','LeafIndex'])['Area']= df.groupby(['Day','Time','LeafIndex'])['Area'].mean()
df.to_csv('/home/psych256lab/Downloads/Scripts/area_output(updateddayrack).csv')