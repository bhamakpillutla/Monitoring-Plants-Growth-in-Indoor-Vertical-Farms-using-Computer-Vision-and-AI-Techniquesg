import numpy as np
import cv2 as cv

img = cv.imread('/home/psych256lab/Downloads/Scripts/centroid.jpg',0)
ret,thresh = cv.threshold(img,127,255,0)
contours,hierarchy = cv.findContours(thresh, 1, 2)

if len(contours) > 0:
    for c in contours:
        M = cv.moments(c)
        print( M)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        
        cv.circle(img, (cX, cY), 5, (0, 0, 255), -1)
        cv.imshow("Image", img)
        cv.waitKey(0)
        
else:
    print("Sorry No contour Found.")

