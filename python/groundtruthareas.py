#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 15:57:51 2022

@author: psych256lab
"""
import os, json
import pandas as pd
#from sympy import Point, Polygon
from shapely.geometry import Polygon
def Area(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area
path_to_json = '/home/psych256lab/Downloads/groundtruths_test'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
print(json_files)  # for me this prints ['foo.json']
csvDate = []
csvID = []
csvArea = []
for j in json_files:
    # Opening JSON file
    with open(path_to_json+"/"+j) as json_file:
        data = json.load(json_file)
        print(j)
        
        # Print the type of data variable
        print("data:", len(data["shapes"]))
        shapes = data["shapes"]
        for i in shapes:
            #print(i["group_id"],i["points"])
            # Convert List of Lists to Tuple of Tuples
            # Using tuple + list comprehension
            res = tuple(tuple(sub) for sub in i["points"])
            #print(i["group_id"],list(res))
            #area = float(abs(Polygon(*i["points"]).area))
            polygon = Polygon(res)
            area = polygon.area
            
            csvID.append(i["group_id"])
            csvArea.append(area)
            date = j[6:25]
            csvDate.append(date)
            #list_of_areas.append(area)
            #print(i["group_id"],Area(res))
            
       
    print("file done")


    
    
dt = {'Date':csvDate,'LeafIndex': csvID,'Area':csvArea}
df = pd.DataFrame(dt)
df.to_csv('/home/psych256lab/Documents/groundtruths_test.csv')