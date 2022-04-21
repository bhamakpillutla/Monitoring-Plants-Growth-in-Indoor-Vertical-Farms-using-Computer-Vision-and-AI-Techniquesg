
"""
Created on Wed Sep  1 11:57:14 2021

@author: Bhama Pillutla

Code to detect only leaves that are present in all timestamps of the day
with zero center


"""

import matplotlib.pyplot as plt
import cv2

from detectron2.utils.visualizer import ColorMode, Visualizer
import glob



import regex as re
import pandas as pd

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
import time

from sympy import Point, Polygon
from imantics import Polygons, Mask

from math import sqrt
from shapely.geometry import Polygon as shp
#import torch
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg= get_cfg()
cfg.merge_from_file("/home/psych256lab/Downloads/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
#from detectron2.modeling import build_model
#model = build_model(cfg)
#cfg.MODEL.DEVICE='cuda:0'
cfg.MODEL.WEIGHTS = "/home/psych256lab/Downloads/Scripts/output/model_final.pth"
#torch.save(model.state_dict(), cfg.MODEL.WEIGHTS)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.10

dataset_metadata = MetadataCatalog.get("august_train")
cfg.TEST.DETECTIONS_PER_IMAGE = 1000

predictor = DefaultPredictor(cfg)



dir_name = '/home/psych256lab/Downloads/Scripts/august/test_last/'
#all_images = [file_name for file_name in glob.glob('/home/psych256lab/Downloads/Scripts/septtest/*.jpg')]

all_images = sorted( filter( os.path.isfile,
                        glob.glob(dir_name + '*.jpg') ) )
print(all_images)

def calc_distance(p1, p2): # simple function, I hope you are more comfortable 
  return sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2) # Pythagorean theorem

print("Started")

csvDATE = []
csvDAY = []
csvAREA = []
csv_IDS = []
csvCENTROID = []
csvTIME = []

ix = 0
day = 1

while ix < len(all_images):

   images = all_images[ix : ix+6]
   #images = [ all_images[i] for i in range(0,5)]
   #print(images)
   #while(ix <= 6):
   print("set of images: \n",images)
   firstCentroidData = []
   firstAreaData = []
   firstValues = []
   firstPolys = []
 
   ## first phase of image iteration and detections
   for im_path in images:
       print("current image path: ",im_path)
     
       last_of_centroids = []
       last_of_areas = []
       last_of_values = []
       last_of_polys = []
       im = cv2.imread(im_path)
       #print("entered")
       outputs = predictor(im)
       v = Visualizer(im[:, :, ::-1],
             metadata=dataset_metadata, 
             scale=0.8, 
             instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
             )
       #out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
       instances = outputs["instances"].to("cpu")
       instances.remove('pred_boxes')
       instances.remove('pred_classes')
       instances.remove('scores')
       
       #out = v.draw_instance_predictions(instances.to("cpu"))
    
       mask_array = outputs['instances'].pred_masks.to("cpu").numpy()
        #mask_array = torch.from_numpy(mask_array).float().to(device)
        
       num_instances = mask_array.shape[0]
       
       mask_array = np.moveaxis(mask_array, 0, -1)
        
       mask_array_instance = []
      
       for i in range(num_instances):
           mask_array_instance.append(mask_array[:, :, i:(i+1)])
           array = mask_array[:,:,i:(i+1)]
           polygons = Mask(array).polygons()
           polygon_points = polygons.points
           #print(polygon_points)# [([[401,873],[890,789],.....]])]
   
           for poly in polygon_points:
               #print(poly) # [[234 567],[7893 678],....]
               array_of_tuples = map(tuple, poly)
               tuple_of_tuples = tuple(array_of_tuples)
               #print(tuple_of_tuples)
 
               try:
                   area = float(abs(Polygon(*tuple_of_tuples).area))
                   last_of_areas.append(area)
                   
                   tuple_poly = tuple([int(i) for i in tuple(Polygon(*tuple_of_tuples).centroid)])
                   last_of_centroids.append(tuple_poly)
               except AttributeError:
                   continue
               #list_of_polys.append((Polygon(*tuple_of_tuples)))all_images
                        
               #print(shp(list(tuple_of_tuples)))
               last_of_polys.append(list(tuple_of_tuples))
               
               #list_of_values.append(val)
               #val = val + 1
               #out = v.draw_text(str(value),tuple_poly)
            
 
       date = re.search(r'\d{2}-\d{2}-\d{4} \d{2}_\d{2}_\d{2}',im_path)    
       d = datetime.strptime(date.group(), '%d-%m-%Y %H_%M_%S')
       print("image date: ",d)
       dt = datetime.strptime(str(d), "%Y-%m-%d %H:%M:%S")
       #print(type(dt))
       curr_day = "Day - "+str(day)
       curr_time = str(dt.hour)+":"+str(dt.minute)
       #print("previous centroid length = ",len(firstCentroidData))
       
       
       
       if(len(firstCentroidData) == 0) :    
            value = 0
        
            for centroid,area,poly in zip(last_of_centroids,last_of_areas,last_of_polys):
                pX,pY = centroid  
                #out = v.draw_polygon(poly, 'r', edge_color=None, alpha=0.5)
                #out = v.draw_text(str(value),(pX,pY))
                
                firstCentroidData.append(centroid)
                firstAreaData.append(area)
                firstValues.append(value)
                firstPolys.append(poly)
                
                
                value = value + 1
            print("\n first data updated")
            #success = cv2.imwrite('/home/psych256lab/Downloads/Scripts/test_results/test_sept20/'+str(d)+".jpg",out.get_image()[:,:,::-1])
            #print("output file written : ",success)
       else:
            #print("first centroids",firstCentroidData)
            #print("first values: ",firstValues)
            #print("current centroids: ",list_of_centroids)
            
            repcentroid = []
            repareas = []
            repvalues = []
            reppolys = []
            
            for curr_centroid,curr_area,curr_poly in zip(last_of_centroids,last_of_areas,last_of_polys):
                cX,cY = curr_centroid
                for prev_centroid,prev_area,prev_poly in zip(firstCentroidData,firstAreaData,firstPolys):
                    pX,pY = prev_centroid
                    distance = calc_distance((pX,pY), (cX,cY) ) 
                    if (abs(distance) <= 20  ):
                        start = time.process_time()
                        #isIntersection = Polygon(*curr_poly).intersection(Polygon(*prev_poly))             
                        #print("len of intersection:",len(isIntersection))
                        #print("entered shp")
                        #print("current poly: ",curr_poly)
                        #print("prev poly: ",prev_poly)
                        #shIntersection = shp(curr_poly).intersection(shp(prev_poly))
                            
                        shIntersection = shp(curr_poly).intersects(shp(prev_poly))
                        #print("intersection",shIntersection)
                        #print(time.process_time() - start)
                        
                        
                        #if  len(isIntersection)!=0:
                        if (shIntersection!= False ):
                            
                            
                            
                            cen_x,cen_y = curr_centroid
                            cen_px,cen_py = prev_centroid
                          
                            zeroecentered_curr_poly = [(x-cen_x,y-cen_y) for x,y in curr_poly]
                            zerocentered_prev_poly = [(x-cen_px,y-cen_py) for x,y in prev_poly]
                            
                            try:
                                zero_cen_shIntersection = shp(zeroecentered_curr_poly).intersects(shp(zerocentered_prev_poly))
                                
                                if(zero_cen_shIntersection != False):
                                    #print("intersects : length is :",len(shIntersection))
                                    idx= firstCentroidData.index(prev_centroid)     
                                    value = firstValues[idx]
                                    
                                    #out = v.draw_polygon(curr_poly, 'r', edge_color=None, alpha=0.5)
                                    #out = v.draw_text(str(value),(cX,cY))
                                    repcentroid.append(curr_centroid)
                                    repareas.append(curr_area)
                                    repvalues.append(value)
                                    reppolys.append(curr_poly)
                                    
                            except:
                                    continue 
                    
                    #for idxc,idxa,idxv in zip(range(len(firstCentroidData)),range(len(firstAreaData)),range(len(firstValues))):   
                        #   if( centroid == firstCentroidData[idxc] or abs(area-firstAreaData[idxa]) <= 50 and valueex != firstValues[idxv]):
                            #      print("both are same leaves",centroid,firstCentroidData[idxc],firstValues[idxv])
                            
                            
                            # valueex = value
            if len(repcentroid)!=0:
                del firstValues[:]
                del firstCentroidData[:]
                del firstAreaData[:]
                del firstPolys[:]
                # print(repcentroid)
                print("\n updating previous data")
                firstCentroidData = repcentroid
                firstAreaData = repareas
                firstValues = repvalues
                firstPolys = reppolys
               
                #  print("updated values: ",firstCentroidData)
            elif len(repcentroid) == 0 and last_of_centroids!=0:
                del firstValues[:]
                del firstCentroidData[:]
                del firstAreaData[:]
                del firstPolys[:]
                # print(list_of_centroids)cfg.MODEL.DEVICE='cuda:0'
                print("\n updation of previous data not possible, taking current data as new previous")
                value = 0
                for centroid,area,poly in zip(last_of_centroids,last_of_areas,last_of_polys):
                    pX,pY = centroid 
                    
                    #out = v.draw_polygon(poly, 'r', edge_color=None, alpha=0.5)
                    #out = v.draw_text(str(value),(pX,pY))
                    
                    firstCentroidData.append(centroid)
                    firstAreaData.append(area)
                    firstValues.append(value)
                    firstPolys.append(poly)
                 
                    
                    value = value + 1
      
     
                    
   print("\n ***************\nlast image data:\n",len(firstCentroidData))
   
   
   prev_image_data = []
   prev_image_ids = []
   ## second phase of image iteration, detection and matching
   for im_path in images:
       print(im_path)
        
       list_of_centroids = []
       list_of_areas = []
       list_of_values = []
       list_of_polys = []
       
      
       
       im = cv2.imread(im_path)
       #print("entered")
       outputs = predictor(im)
       v = Visualizer(im[:, :, ::-1],
             metadata=dataset_metadata, 
             scale=0.8, 
             instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
             )
       #out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
       #instances = outputs["instances"].to("cpu")
       #instances.remove('pred_boxes')
       #instances.remove('pred_classes')
       #instances.remove('scores')
       
       #out = v.draw_instance_predictions(instances.to("cpu"))
    
       mask_array = outputs['instances'].pred_masks.to("cpu").numpy()
        #mask_array = torch.from_numpy(mask_array).float().to(device)
        
       num_instances = mask_array.shape[0]
       
       mask_array = np.moveaxis(mask_array, 0, -1)
        
       mask_array_instance = []
 
       for i in range(num_instances):
           mask_array_instance.append(mask_array[:, :, i:(i+1)])
           array = mask_array[:,:,i:(i+1)]
           polygons = Mask(array).polygons()
           polygon_points = polygons.points
           #print(polygon_points)# [([[401,873],[890,789],.....]])]
   
           for poly in polygon_points:
               #print(poly) # [[234 567],[7893 678],....]
               array_of_tuples = map(tuple, poly)
               tuple_of_tuples = tuple(array_of_tuples)
               #print(tuple_of_tuples)
 
               try:
                   area = float(abs(Polygon(*tuple_of_tuples).area))
                   list_of_areas.append(area)
                   
                   tuple_poly = tuple([int(i) for i in tuple(Polygon(*tuple_of_tuples).centroid)])
                   list_of_centroids.append(tuple_poly)
                   
                   list_of_polys.append(list(tuple_of_tuples))
               except AttributeError:
                   continue 
    
               #list_of_polys.append((Polygon(*tuple_of_tuples)))all_images
                        
               #print(shp(list(tuple_of_tuples)))
               
               
               #list_of_values.append(val)
               #val = val + 1
               #out = v.draw_text(str(value),tuple_poly)
            
 
       date = re.search(r'\d{2}-\d{2}-\d{4} \d{2}_\d{2}_\d{2}',im_path)    
       d = datetime.strptime(date.group(), '%d-%m-%Y %H_%M_%S')
       print("image date: ",d)
       dt = datetime.strptime(str(d), "%Y-%m-%d %H:%M:%S")
       print(type(dt))
       curr_day = "Day - "+str(day)
       curr_time = str(dt.hour)+":"+str(dt.minute)
       
       
       #ids = [i for i in range(len(ct))]
       
       print("len of polys to check: current :",len(list_of_centroids))
       print("len of polys to check: previous :",len(firstCentroidData))
       
       if len(prev_image_data) == 0:
          value = 0
          for curr_centroid,curr_area,curr_poly in zip(list_of_centroids,list_of_areas,list_of_polys):
               cX,cY = curr_centroid
               for prev_centroid,prev_area in zip(firstCentroidData,firstAreaData):
                   pX,pY = prev_centroid
                           
                   distance = calc_distance((pX,pY), (cX,cY)) 
                   if (abs(distance) <= 20 ):
                       
                       
                       out = v.draw_polygon(curr_poly, 'r', edge_color=None, alpha=0.5)
                       out = v.draw_text(str(value),(cX,cY))
                       
                       print("matched")
                       print(value)         
                       prev_image_data.append(curr_centroid)
                       prev_image_ids.append(value)
                       
                       csvDAY.append(curr_day)
                       csvDATE.append(dt)
                       csvTIME.append(curr_time)
                       csvAREA.append(curr_area)
                       csvCENTROID.append(curr_centroid)
                       csv_IDS.append(value)
                       
                       value = value + 1
                           
       else:
           temp_centroid = []
           temp_ids = []
           for curr_centroid,curr_area,curr_poly in zip(list_of_centroids,list_of_areas,list_of_polys):
               cX,cY = curr_centroid
               #if curr_centroid in prev_image_data:
               for prev_centroid,idx in zip(prev_image_data,prev_image_ids):
                   pX,pY = prev_centroid
                       #print("prev: ",prev_centroid)
                   distance = calc_distance((pX,pY), (cX,cY)) 
                   if (abs(distance) <= 20 ):
                           print("matched")
                           out = v.draw_polygon(curr_poly, 'r', edge_color=None, alpha=0.5)
                           out = v.draw_text(str(idx),(cX,cY))
                           #value = value + 1
                           print(idx)
                           
                           temp_centroid.append(curr_centroid)
                           temp_ids.append(idx)
                           
                           csvDAY.append(curr_day)
                           csvDATE.append(dt)
                           csvTIME.append(curr_time)
                           csvAREA.append(curr_area)
                           csvCENTROID.append(curr_centroid)
                           csv_IDS.append(idx)
           if(len(temp_centroid)!=0):          
               del prev_image_data[:]
               del prev_image_ids[:]
               
               prev_image_data = temp_centroid
               prev_image_ids = temp_ids
          
               
                           
       
       cv2.imwrite('/home/psych256lab/Downloads/Scripts/test_results/april13/'+str(d)+".jpg", out.get_image()[:,:,::-1])     
   ix = ix+6
   day = day + 1
dt = {'Date':csvDATE,'Day':csvDAY,'Time':csvTIME,'LeafIndex': csv_IDS,'Area':csvAREA,'Centroid':csvCENTROID}
df = pd.DataFrame(dt)
df.to_csv('/home/psych256lab/Downloads/Scripts/area_output(april13).csv')