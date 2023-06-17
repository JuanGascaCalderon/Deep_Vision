# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 19:13:32 2023

@author: JUAN PABLO GASCA
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

#Read method for pic path
blueberries = cv2.imread(r"C:\Users\JUANPABLOGASCA\Desktop\Deep_Vision\recursos\Arandanos.webp")
#RGB = BGR (scales of three color - blue, green and red)
b = blueberries[:,:,0] #rows x columns of color blue
g = blueberries[:,:,1] #rows x columns of color green
r = blueberries[:,:,2] #rows x columns of color red

#Intensity Color (Average Channels - Gray's Scales)
img_gray = cv2.cvtColor(blueberries, cv2.COLOR_BGR2GRAY) #BGT to GRAY
#cv2.imshow('', img_gray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#Binary Image upper 100
binary = np.uint8(255* (img_gray<233)) #Convert bool to image (Unbralization)
"""cv2.imshow('', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

#Multiplication - Segmentation throught scales grays and binary
gray_segmentada = np.uint8(img_gray*(binary/255))

"""cv2.imshow('', gray_segmentada)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

#Multiplication - Segmentation throught orignal color scale and binary
seg_color = blueberries.copy()
seg_color[:,:,0] = np.uint8(b*(binary/255))
seg_color[:,:,1] = np.uint8(g*(binary/255))
seg_color[:,:,2] = np.uint8(r*(binary/255))

cv2.imshow('', seg_color)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Frequency Histograms of the Images
#bins = intervals
plt.hist(img_gray.flatten(), bins=15) #Array without dimentions
plt.show()




