# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:59:23 2023

@author: JUAN PABLO GASCA
"""
#Import library's
import cv2
import numpy as np

#Read the picture
blueberries = cv2.imread(r"C:\Users\JUANPABLOGASCA\Desktop\Deep_Vision\recursos\Arandanos.webp")
#Show the picture
cv2.imshow('',blueberries)

#Kernel Creation
kernel_3x3 = np.ones((3,3))/(3*3) #Matrix conformed by 3*3 (average equal 1)
output3x3 = cv2.filter2D(blueberries,-1,kernel_3x3) #Final image with filter
#Picture filter by 3x3
cv2.imshow('Filter 3*3',output3x3)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Kernel Creation with filter 11x11
kernel_11x11 = np.ones((11,11))/(11*11) #Matrix conformed by 11*11 (average equal 1)
output11x11 = cv2.filter2D(blueberries,-1,kernel_11x11) #Final image with filter
#Picture filter by 11x11
cv2.imshow('Filter 11*11',output11x11)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Kernel Creation with filter 31x31
kernel_31x31 = np.ones((31,31))/(31*31) #Matrix conformed by 31*31 (average equal 1)
output31x31 = cv2.filter2D(blueberries,-1,kernel_31x31) #Final image with filter
#Picture filter by 31x31
cv2.imshow('Filter 31*31',output31x31)
cv2.waitKey(0)
cv2.destroyAllWindows()



