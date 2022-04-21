# -*- coding: utf-8 -*-
"""
Spyder Editor
@author: Bhama Pillutla

Predictor file

Pure predictor without tracking code.
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



cfg= get_cfg()
cfg.merge_from_file("/home/psych256lab/Downloads/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.MODEL.WEIGHTS = "/home/psych256lab/Downloads/Scripts/output/model_final.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.10

dataset_metadata = MetadataCatalog.get("august_train")
cfg.TEST.DETECTIONS_PER_IMAGE = 1000
cfg.DATASETS.TEST = ("august_val" )
predictor = DefaultPredictor(cfg)



imageDate = []
area = []
count = 0
print("Started")
for im_path in glob.glob('/home/psych256lab/Downloads/Scripts/august/val/*.jpg'):    
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


    masks = outputs['instances'].get('pred_masks').cpu().numpy()
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
    #print('output',output)
    date = re.search(r'\d{2}-\d{2}-\d{4} \d{2}_\d{2}_\d{2}',im_path) 
    for i in range(num_instances):
      mask_array_instance.append(mask_array[:, :, i:(i+1)])
      array = mask_array[:,:,i:(i+1)]
      polygons = Mask(array).polygons()
      polygon_points = polygons.points
      #print(polygon_points)
      contour_sizes = []
      for poly in polygon_points:
        contour_sizes.append(cv2.contourArea(poly))
        #date=re.search("([0-9]{2}\-[0-9]{2}\-[0-9]{4})", im_path)  
        date = re.search(r'\d{2}-\d{2}-\d{4} \d{2}_\d{2}_\d{2}',im_path) 
        d = datetime.strptime(date.group(), '%d-%m-%Y  %H_%M_%S')
        #print("file name: {},date and time : {}".format(im_path, d))sem_seg
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
    
    date = re.search(r'\d{2}-\d{2}-\d{4} \d{2}_\d{2}_\d{2}',im_path) 
    d = datetime.strptime(date.group(), '%d-%m-%Y  %H_%M_%S')
    print("file name: {},date and time : {}".format(im_path,d))
    
    im = Image.fromarray(output)
    plt.imshow(im)
    #im.save("/home/psych256lab/Downloads/centroid_test/binary_results/"+str(d)+".jpg")
    
    #plt.figure(figsize = (14, 10))
    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    
    count = count + 1
    #print(count)
    print(d)
    #success = cv2.imwrite("/home/psych256lab/Downloads/Scripts/test_results/test_sept20/"+str(d)+'.jpg',v.get_image()[:, :, ::-1])#mask
    print(count)
   #print(success)
    #plt.show()
    
    

d = {'ImageDate':imageDate,'Area':area}
df = pd.DataFrame(d)

#import the COCO Evaluator to use the COCO Metrics
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

#Call the COCO Evaluator function and pass the Validation Dataset
evaluator = COCOEvaluator("august_val", cfg, False, output_dir="/home/psych256lab/Downloads/Scripts/output/")
val_loader = build_detection_test_loader(cfg, "august_val")

#Use the created predicted model in the previous step
inference_on_dataset(predictor.model, val_loader, evaluator)
"""

mean = df.groupby(['ImageDate'], as_index=False).agg({'Area': 'mean'})
median = df.groupby(['ImageDate'], as_index=False).agg({'Area': 'median'})
mode = df.groupby(['ImageDate'], as_index=False)['Area'].apply(lambda x: x.mode().iloc[0])
std = df.groupby(['ImageDate'],as_index=False)['Area'].std()


"""


#df.to_csv('/home/psych256lab/Downloads/Scripts/area_output(test_sept20).csv')


    # if masks.shape[2] != 0:
    #     for i in range(masks.shape[2]):
    #         polygons = Mask(masks[:, :, i]).polygons()
    # else:
    #     polygons = Mask(masks[:, :, 0]).polygons()
    #print(polygons.points)


    # get the polygons from the masks
    # polygons = Mask(array).polygons() 
    # print(polygons.points)
    # # area = cv2.contourArea(np.array(polygons.points))
    # print(area)

    
    # print(polygons.segmentation)
    


    ## get the detected images
    #cv2.imshow(v.get_image())
   
"""
print("mean:\n",df.groupby(['ImageDate'], as_index=False).agg({'Area': 'mean'}))
print("median:\n",df.groupby(['ImageDate'], as_index=False).agg({'Area': 'median'}))
print("mode:\n",df.groupby(['ImageDate'], as_index=False)['Area'].apply(lambda x: x.mode().iloc[0]))
print("standard deviation:\n",df.groupby('ImageDate').std())

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
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=250))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.figure(figsize = (100,50))
#plt.savefig("/home/psych256lab/Downloads/Scripts/graphs/mean.jpg")
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
plt.title('MEAN!')
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=250))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.figure(figsize = (100,50))
#plt.savefig("/home/psych256lab/Downloads/Scripts/graphs/median.jpg")
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
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=250))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.figure(figsize = (100,50))
plt.savefig("/home/psych256lab/Downloads/Scripts/graphs/mode.jpg")
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
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=250))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
plt.figure(figsize = (100,50))
#plt.savefig("/home/psych256lab/Downloads/Scripts/graphs/std.jpg")
plt.show()
print("Completed")
"""