# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 17:44:34 2023

@author: JUAN PABLO GASCA
"""

import cv2
import numpy as np

#Read method for pic path
blueberries = cv2.imread(r"C:\Users\JUANPABLOGASCA\Desktop\Deep_Vision\recursos\Arandanos.webp")
#RGB = BGR (scales of three color - blue, green and red)
b = blueberries[:,:,0] #rows x columns of color blue
g = blueberries[:,:,1] #rows x columns of color green
r = blueberries[:,:,2] #rows x columns of color red

bananas = cv2.imread(r"C:\Users\JUANPABLOGASCA\Desktop\Deep_Vision\recursos\bananos.jpg")
#RGB = BGR (scales of three color - blue, green and red)
b2 = bananas[:,:,0] #rows x columns of color blue
g2 = bananas[:,:,1] #rows x columns of color green
r2 = bananas[:,:,2] #rows x columns of color red

#Show the picture using opencv
"""cv2.imshow('',bananas) #Original Image
cv2.waitKey(0)
cv2.destroyAllWindows()"""

#Channel Colors - RGB Scales
#Color blue
cv2.imshow('',b) #Original Image
cv2.waitKey(0)
cv2.destroyAllWindows()

#Color green
cv2.imshow('',g) #Original Image
cv2.waitKey(0)
cv2.destroyAllWindows()

#Color red
cv2.imshow('',r) #Original Image
cv2.waitKey(0)
cv2.destroyAllWindows()


#Intensity Color (Average Channels - Gray's Scales)
img_gray = cv2.cvtColor(blueberries, cv2.COLOR_BGR2GRAY) #BGT to GRAY
cv2.imshow('', img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Binary Image upper 100
binary = np.uint8(255* (img_gray>100)) #Convert bool to image
cv2.imshow('', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Binary image under 70
binary2 = np.uint8(255*(img_gray<200)) #Show in white
cv2.imshow('',binary2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Binary image upper 190
binary3 = np.uint8(255*(img_gray>190)) #Show in black
cv2.imshow('',binary3)
cv2.waitKey(0)
cv2.destroyAllWindows()



