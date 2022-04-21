#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 11:21:26 2021

@author: Bhama Pillutla

Code to predict leaves in Binary format 
using CV

"""
import glob
import cv2
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt
import cv2

from detectron2.utils.visualizer import ColorMode, Visualizer

import numpy as np

import imantics
from imantics import Polygons, Mask
import regex as re
import pandas as pd
import numpy
from datetime import datetime
from PIL import Image



cfg= get_cfg()
cfg.merge_from_file("/home/psych256lab/Downloads/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.MODEL.WEIGHTS = "/home/psych256lab/Downloads/Scripts//output/model_final.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.10

dataset_metadata = MetadataCatalog.get("july_train")
cfg.TEST.DETECTIONS_PER_IMAGE = 1000
cfg.DATASETS.TEST = ("august_test" )
predictor = DefaultPredictor(cfg)


imageDate = []
area = []
print("Started")
for im_path in glob.glob('/content/july/test/*.jpg'):    
    im = cv2.imread(im_path)
    #print("entered")
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=dataset_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #print(outputs["instances"])   
    ## get the boxes and masks data

    # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
    #print(outputs["instances"].pred_classes)
    #print(outputs["instances"].pred_boxes.tensor.cpu().numpy())
    # print(outputs["instances"].pred_masks.to('cpu').numpy())


    #masks = outputs['instances'].get('pred_masks').cpu().numpy()
    # # print(masks)
    # array = (masks > 126) * 255
    # print(array)
    # positive_pixel_count = masks.sum() # assumes binary mask (True == 1)
    # h, w = masks.shape[1:3] # assumes NHWC data format, adapt as needed
    # area = positive_pixel_count / (w*h)
    # print(area)
    
    mask_array = outputs['instances'].pred_masks.to('cpu').numpy()
    num_instances = mask_array.shape[0]
    mask_array = np.moveaxis(mask_array, 0, -1)
    
    mask_array_instance = []
    output = np.zeros_like(im) #black

    for i in range(num_instances):
      mask_array_instance.append(mask_array[:, :, i:(i+1)])
      array = mask_array[:,:,i:(i+1)]

      # polygons code

      polygons = Mask(array).polygons()
      polygon_points = polygons.points
      #print(polygon_points)
      contour_sizes = []
      for poly in polygon_points:
        contour_sizes.append(cv2.contourArea(poly))
        #date=re.search("([0-9]{2}\-[0-9]{2}\-[0-9]{4})", im_path)  
        date = re.search(r'\d{2}-\d{2}-\d{4} \d{2}_\d{2}_\d{2}',im_path) 
        d = datetime.strptime(date.group(), '%d-%m-%Y  %H_%M_%S')
        #print("file name: {},date and time : {}".format(im_path, d))
        #print(dt)
        # print(date.group(1))
        imageDate.append(d)
        #imageDate.append(date.group(1))
        area.append(contour_sizes.pop())


      #contour_sizes = [(cv2.contourArea(poly)) for poly in polygon_points]
      # area = cv2.contourArea(polygon_points)
      #print(contour_sizes) 
      # print(area)
     
      output = np.where(mask_array_instance[i] == True, 255, output)
    im = Image.fromarray(output)
    plt.imshow(im)
    im.save("/content/outputdetected/"+str(d)+".jpg")
    #cv2.imwrite("/content/outputdetected/"+str(d)+".jpg",im)
  
    #print("file name: {},date and time : {}".format(im_path,d))
    plt.figure(figsize = (14, 10))
    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
  
    #cv2.imwrite("/content/outputdetected/"+str(d)+'.jpg',v.get_image()[:, :, ::-1])#mask
   #plt.show()
    
    

d = {'ImageDate':imageDate,'Area':area}
df = pd.DataFrame(d)




mean = df.groupby(['ImageDate'], as_index=False).agg({'Area': 'mean'})
median = df.groupby(['ImageDate'], as_index=False).agg({'Area': 'median'})
mode = df.groupby(['ImageDate'], as_index=False)['Area'].apply(lambda x: x.mode().iloc[0])
std = df.groupby(['ImageDate'],as_index=False)['Area'].std()





#df.to_csv('/home/psych256lab/Downloads/Scripts/area_output.csv')