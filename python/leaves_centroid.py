#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 11:12:04 2021

@author: Bhama Pillutla

calculating centroids using openCV functions


"""
import os
import cv2
import numpy as np
import glob
#from google.colab.patches import cv2_imshow
from PIL import Image
import regex as re
from datetime import datetime
# Reading files and sorting them in the right order

all_images = [file_name for file_name in glob.glob('/home/psych256lab/Downloads/centroid_test/binary_results/*.jpg') ]
all_images.sort(key=lambda k: k.split(".")[0][-1])
print(all_images) # 

# Initially, no centroid information is available. 
previous_centroid_x = -1
previous_centroid_y = -1
 
DIST_THRESHOLD = 20
for i, image_name in enumerate(all_images):
    rgb_image = cv2.imread(image_name)
    height, width = rgb_image.shape[:2]
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_image, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
    blankImage = np.zeros_like(rgb_image)
    for cnt in contours:
        # Refer to https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#moments
        M = cv2.moments(cnt)
        if M["m00"] != 0:
          cX = int(M["m10"] / M["m00"])
          cY = int(M["m01"] / M["m00"])
        else:
          # set values as what you need in the situation
          cX, cY = 0, 0
        # Refer to https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_properties/py_contour_properties.html#contour-properties
        
        extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
        extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
        extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
        extBot = tuple(cnt[cnt[:, :, 1].argmax()][0])
        
        color = (0, 0, 255)
        if i == 0: # First frame - Assuming that you can find the correct blob accurately in the first frame
            # Here, I am using a simple logic of checking if the blob is close to the centre of the image. 
            if abs(cY - (height / 2)) < DIST_THRESHOLD: # Check if the blob centre is close to the half the image's height
                previous_centroid_x = cX # Update variables for finding the next blob correctly
                previous_centroid_y = cY
                DIST_THRESHOLD = (extBot[1] - extTop[1]) / 2 # Update centre distance error with half the height of the blob
                color = (0, 255, 0) 
        else:
            if abs(cY - previous_centroid_y) < DIST_THRESHOLD: # Compare with previous centroid y and see if it lies within Distance threshold
                previous_centroid_x = cX
                previous_centroid_y = cY
                color = (0, 255, 0) 

        cv2.drawContours(blankImage, [cnt], 0, color, -1) 
        cv2.circle(blankImage, (cX, cY), 3, (255, 0, 0), -1)

    date = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',image_name) 
    
    d = datetime.strptime(date.group(), '%Y-%m-%d %H:%M:%S')
    #d = datetime.strptime('%Y-%m-%d %H:%M:%S')
    print(d)

    #im = Image.fromarray(blankImage)
    #im.save("/contents/results/"+str(d)+".jpg")
    cv2.imwrite('/home/psych256lab/Downloads/centroid_test/results/'+str(d)+".jpg",blankImage)
    print("..")
    #cv2.imshow(blankImage)
        #cv2.imwrite("result_" + image_name, blankImage)
    