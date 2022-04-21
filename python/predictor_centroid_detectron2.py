#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 11:43:48 2021

@author: Bhama Pillutla

centroid tracking code - checking code file

"""

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt
import cv2

from detectron2.utils.visualizer import ColorMode, Visualizer
import glob
import numpy as np

import imantics
from imantics import Polygons, Mask
import regex as re
import pandas as pd
import numpy
from datetime import datetime




import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import os
import numpy as np
import json
from detectron2.structures import BoxMode
import time
from PIL import Image


from sympy import Point, Polygon
from imantics import Polygons, Mask

from math import sqrt


cfg= get_cfg()
cfg.merge_from_file("/home/psych256lab/Downloads/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.MODEL.WEIGHTS = "/home/psych256lab/Downloads/Scripts/output/model_final.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.10

dataset_metadata = MetadataCatalog.get("august_train")
cfg.TEST.DETECTIONS_PER_IMAGE = 1000

predictor = DefaultPredictor(cfg)


leaf_id1 = []
image_date1 = []
image_areas1 = []

leaf_id2 = []
image_date2 = []
image_areas2 = []

dir_name = '/home/psych256lab/Downloads/Scripts/august/test_updated/'
#all_images = [file_name for file_name in glob.glob('/home/psych256lab/Downloads/Scripts/septtest/*.jpg')]

all_images = sorted( filter( os.path.isfile,
                        glob.glob(dir_name + '*.jpg') ) )
print(all_images)

def calc_distance(p1, p2): # simple function, I hope you are more comfortable 
  return sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2) # Pythagorean theorem

print("Started")

value = 0
matching_check = {}
#for im_path in range(len(all_images)-1):
for im_path in range(10,20):
  print(all_images[im_path])
  print(all_images[im_path+1])
  

  
  list_of_centroids1 = []
  list_of_areas1 = []
  list_of_centroids2 = []
  list_of_areas2 = []
  list_of_dates1 = []
  list_of_dates2 = []

  im_1 = cv2.imread(all_images[im_path])
  im_2 = cv2.imread(all_images[im_path+1])
    
  #print("entered")
  outputs_1 = predictor(im_1)
  outputs_2 = predictor(im_2)

  v1 = Visualizer(im_1[:, :, ::-1],
                   metadata=dataset_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
  )
  v2 = Visualizer(im_2[:, :, ::-1],
                   metadata=dataset_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
  )
  
  instances1 = outputs_1["instances"].to("cpu")
 # instances1.remove('pred_boxes')
  instances2 = outputs_2["instances"].to("cpu")
 # instances2.remove('pred_boxes')
  
  
 
  out1 = v1.draw_instance_predictions(instances1.to("cpu"))

  out2 = v2.draw_instance_predictions(instances2.to("cpu"))
    
    
    
  mask_array1 = outputs_1['instances'].pred_masks.to('cpu').numpy()
  mask_array2 = outputs_2['instances'].pred_masks.to('cpu').numpy()

  num_instances1 = mask_array1.shape[0]
  num_instances2 = mask_array2.shape[0]

  mask_array1 = np.moveaxis(mask_array1, 0, -1)
  mask_array2 = np.moveaxis(mask_array2, 0, -1)

  mask_array_instance1 = []
  mask_array_instance2 = []
  
  date1 = re.search(r'\d{2}-\d{2}-\d{4} \d{2}_\d{2}_\d{2}',all_images[im_path])   
  date2 = re.search(r'\d{2}-\d{2}-\d{4} \d{2}_\d{2}_\d{2}',all_images[im_path+1])   
  d1 = datetime.strptime(date1.group(), '%d-%m-%Y %H_%M_%S')
  d2 = datetime.strptime(date2.group(), '%d-%m-%Y %H_%M_%S')
  print(d1)
  print(d2)

  for i in range(num_instances1):
    mask_array_instance1.append(mask_array1[:, :, i:(i+1)])
    array = mask_array1[:,:,i:(i+1)]
    polygons = Mask(array).polygons()
    polygon_points = polygons.points
    for poly in polygon_points:
      array_of_tuples = map(tuple, poly)
      tuple_of_tuples = tuple(array_of_tuples)
      try:
          area = float(abs(Polygon(*tuple_of_tuples).area))
          list_of_areas1.append(area)

          tuple_poly = tuple([int(i) for i in tuple(Polygon(*tuple_of_tuples).centroid)])
          list_of_centroids1.append(tuple_poly)
          
          list_of_dates1.append(d1)
            #out = v.draw_text(str(value),tuple_poly)
      except:
          pass

  for i in range(num_instances2):
    mask_array_instance2.append(mask_array2[:, :, i:(i+1)])
    array = mask_array2[:,:,i:(i+1)]
    polygons = Mask(array).polygons()
    polygon_points = polygons.points
    for poly in polygon_points:
      array_of_tuples = map(tuple, poly)
      tuple_of_tuples = tuple(array_of_tuples)
      try:
          area = float(abs(Polygon(*tuple_of_tuples).area))
          list_of_areas2.append(area)

          tuple_poly = tuple([int(i) for i in tuple(Polygon(*tuple_of_tuples).centroid)])
          list_of_centroids2.append(tuple_poly)
          
          list_of_dates2.append(d2)
            #out = v.draw_text(str(value),tuple_poly)
      except:
          pass


    
  # prev_image = out1.get_image()[:,:,::-1]
  # curr_image = out2.get_image()[:,:,::-1]
  

  #cv2.imwrite('/content/dummy/'+str(d)+".jpg",curr_image)
  # curr_image = cv2.imread('/content/dummy/'+str(d)+".jpg")
    
  
  
  for prev_cntd,prev_area in zip(list_of_centroids1,list_of_areas1):
    pX, pY = prev_cntd
    for curr_cntd,curr_area in zip(list_of_centroids2,list_of_areas2):
        
      cX, cY = curr_cntd
      distance = calc_distance((pX,pY), (cX,cY) ) 
      if (distance <= 3 and abs(prev_area-curr_area) <= 50):
        print("\n FIXED X AND Y: ",pX,pY)
        print("\nnew point is: ",cX,cY)
        print("leaves have same centroid")
        out1 = v1.draw_text(str(value),(pX,pY))
        out2 = v2.draw_text(str(value),(cX,cY))
        image_date1.append(d1)
        leaf_id1.append(value)
        image_areas1.append(prev_area)
        image_date2.append(d2)
        leaf_id2.append(value)
        image_areas2.append(curr_area)

        #if (cX,cY) in matching_check:
         #   vid =  matching_check[(cX,cY)]
           
         #   out1 = v1.draw_text(str(vid),(pX,pY))
          #  out2 = v2.draw_text(str(vid),(cX,cY))
        #else:
         #   matching_check[(cX,cY)] = value
          #  value = value + 1 


        
          
        #out1 = v1.draw_text(str(value),(pX,pY))
       #out2 = v2.draw_text(str(value),(cX,cY))
        
       

          
  


  
  print(matching_check)

  print("Previous Image")
  cv2.imwrite('/home/psych256lab/Downloads/Scripts/test_results/test_sept20/'+str(d1)+"prev.jpg",out1.get_image()[:,:,::-1])

  print("Current Image")
  cv2.imwrite('/home/psych256lab/Downloads/Scripts/test_results/test_sept20/'+str(d2)+"curr.jpg", out2.get_image()[:,:,::-1])
"""
data_dict = {'ImageDate1':image_date1,'LeafID1':leaf_id1,'Area1':image_areas1,'ImageDate2': image_date2,'LeafID2':leaf_id2,'Area2':image_areas2}
df = pd.DataFrame(data_dict)

df.to_csv('/home/psych256lab/Downloads/Scripts/area_output(test_sept20).csv')

"""

    
    



    
   
