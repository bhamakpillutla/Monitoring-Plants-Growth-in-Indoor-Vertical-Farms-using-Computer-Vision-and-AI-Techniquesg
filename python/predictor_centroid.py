#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 11:57:14 2021

@author: Bhama Pillutla

Centroid tracking using OpenCV
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


cfg= get_cfg()
cfg.merge_from_file("/home/psych256lab/Downloads/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.MODEL.WEIGHTS = "/home/psych256lab/Downloads/Scripts//output/model_final.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.10

dataset_metadata = MetadataCatalog.get("august_train")
cfg.TEST.DETECTIONS_PER_IMAGE = 1000

predictor = DefaultPredictor(cfg)


from math import sqrt
prev_image = " "
prev_image_centroids = []
prev_image_areas = []
prev_date = ""
all_images = [file_name for file_name in glob.glob('/home/psych256lab/Downloads/Scripts/septtest/*.jpg')]
all_images.sort(key=lambda k: k.split("_")[-1].split('.')[0])
def calc_distance(p1, p2): # simple function, I hope you are more comfortable 
  return sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2) # Pythagorean theorem

print("Started")
for im_path in all_images:
    print(im_path)
    list_of_centroids = []
    list_of_areas = []

    im = cv2.imread(im_path)
    
    #print("entered")
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=dataset_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    mask_array = outputs['instances'].pred_masks.to('cpu').numpy()
    num_instances = mask_array.shape[0]

    mask_array = np.moveaxis(mask_array, 0, -1)

    mask_array_instance = []
  

    for i in range(num_instances):
      mask_array_instance.append(mask_array[:, :, i:(i+1)])
      array = mask_array[:,:,i:(i+1)]
      polygons = Mask(array).polygons()
      polygon_points = polygons.points
      for poly in polygon_points:
        array_of_tuples = map(tuple, poly)
        tuple_of_tuples = tuple(array_of_tuples)
        try:
            area = float(abs(Polygon(*tuple_of_tuples).area))
            list_of_areas.append(area)

            tuple_poly = tuple([int(i) for i in tuple(Polygon(*tuple_of_tuples).centroid)])
            list_of_centroids.append(tuple_poly)
            #out = v.draw_text(str(value),tuple_poly)
        except:
            pass

    date = re.search(r'\d{2}-\d{2}-\d{4} \d{2}_\d{2}_\d{2}',im_path)    
    d = datetime.strptime(date.group(), '%d-%m-%Y %H_%M_%S')
    print(d)
    
    curr_image = out.get_image()[:,:,::-1]
    cv2.imwrite('/home/psych256lab/Downloads/Scripts/septtest/dummy/'+str(d)+".jpg",curr_image)
    curr_image = cv2.imread('/home/psych256lab/Downloads/Scripts/septtest/dummy/'+str(d)+".jpg")
    
    value = 0
    if (prev_image != " " and len(prev_image_centroids)!= 0 and len(prev_image_areas)!= 0):
      for prev_cntd in prev_image_centroids:
        pX, pY = prev_cntd
        for curr_cntd in list_of_centroids:
          cX, cY = curr_cntd
          distance = calc_distance((pX,pY), (cX,cY) ) 
          if (distance == 0 or distance <=10):
            print("\n FIXED X AND Y: ",pX,pY)
            print("\nnew point is: ",cX,cY)
            print("leaves have same centroid")
            #curr_image_centroids = v.draw_text(str(value),(cX,cY))
           # prev_image = v.draw_text(str(value), (pX,pY))
            cv2.circle(curr_image, (cX, cY), 3, (209, 80, 0, 255), -1)
            cv2.putText(curr_image, "id" + str(value),(cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 1, cv2.LINE_AA)
            cv2.circle(prev_image,(pX, pY), 3, (209, 80, 0, 255), -1)
            cv2.putText(prev_image, "id" + str(value),(pX, pY), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 1, cv2.LINE_AA)
            value = value + 1

    

      print("Previous Image")
      #print("Centroids :",prev_image_centroids)
      #print("Areas : ",prev_image_areas)
      #plt.imshow(prev_image)  
      #cv2.imwrite("'/home/psych256lab/Downloads/Scripts/septtest/output_centroids/",prev_image,".jpg")
      cv2.imwrite('/home/psych256lab/Downloads/Scripts/septtest/output_centroids/'+str(prev_date)+"prev.jpg",prev_image)

      print("Current Image")
      #print("Centroids : ",list_of_centroids)
      #print("Areas : ",list_of_areas)   
      #cv2.imwrite("'/home/psych256lab/Downloads/Scripts/septtest/output_centroids/",curr_image_centroids,".jpg")
      cv2.imwrite('/home/psych256lab/Downloads/Scripts/septtest/output_centroids/'+str(d)+"curr.jpg",curr_image)

    
   
  
    prev_image = curr_image
    prev_date = d
    prev_image_centroids = list_of_centroids
    prev_image_areas = list_of_areas