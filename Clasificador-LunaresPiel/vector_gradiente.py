# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:51:59 2023

@author: JUAN PABLO GASCA
"""

#Import library's
import cv2
import numpy as np

#Read the picture
blueberries = cv2.imread(r"C:\Users\JUANPABLOGASCA\Desktop\Deep_Vision\recursos\Arandanos.webp")
#Show the picture
#cv2.imshow('',blueberries)

#Intensity Color (Average Channels - Gray's Scales)
img_gray = cv2.cvtColor(blueberries, cv2.COLOR_BGR2GRAY) #BGT to GRAY

#GRADIENTE VECTOR 
gx = cv2.Sobel(img_gray, cv2.CV_64F,1,0,5) #Visualize the y vector
# cv2.imshow('',gx)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
gy = cv2.Sobel(img_gray, cv2.CV_64F,0,1,5) #Visualize the x vector
# cv2.imshow('',gy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

mag,_ = cv2.cartToPolar(gx,gy) #Visualize the magnitud of x,y vector
mag = np.uint8(255*mag/np.max(mag)) #Normalize (0,255) in order to see edge of the pic
cv2.imshow('',mag)
cv2.waitKey(0)
cv2.destroyAllWindows()