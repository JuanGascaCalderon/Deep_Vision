# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 16:04:23 2023

@author: JUAN PABLO GASCA
"""
#Import library's
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

#Show the picture
#img1 = cv2.imread(r"C:\Users\JUANPABLOGASCA\Desktop\Deep_Vision\recursos\datasetLunares\dysplasticNevi\train\dysplasticNevi1.jpg")

def getFeatures(img):    
    
    #Convert to gray
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Otso Method
    threshold,_ = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
    mask = np.uint8(1*(gray<threshold))
    
    B = (1/255)*np.sum(img[:,:,0]*mask) / np.sum(mask)  
    G = (1/255)*np.sum(img[:,:,1]*mask) / np.sum(mask)      
    R = (1/255)*np.sum(img[:,:,2]*mask) / np.sum(mask) 
    
    return [B,G,R]

#Generation of features dataset
paths = [r"C:\Users\JUANPABLOGASCA\Desktop\Deep_Vision\recursos\datasetLunares\dysplasticNevi\train", r"C:\Users\JUANPABLOGASCA\Desktop\Deep_Vision\recursos\datasetLunares\spitzNevus\train"]

labels = [] #Etiquets
features = []


for label, path in enumerate(paths):
    for filename in glob.glob(path+ "\\*.jpg"):
        img = cv2.imread(filename)
        features.append(getFeatures(img))
        labels.append(label)

#Numpy's array's
features = np.array(features)
labels = np.array(labels)

labels = 2*labels-1

#Data Visualization on the space features
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
for i, features_row in enumerate(features):
    if labels[i]==-1:
        ax.scatter(features_row[0], features_row[1],features_row[2],marker='*',c='k')
    else:
        ax.scatter(features_row[0], features_row[1],features_row[2],marker='*',c='r')

ax.set_xlabel('B')
ax.set_ylabel('G')
ax.set_zlabel('R')

#Functions errors about the hiperplane constants
subFeatures = features[:,1::]
loss = []

#Loss Function
for w1 in np.linspace(-6,6,100):
    for w2 in np.linspace(-6,6,100):
        totalError = 0
        for i, features_row in enumerate(subFeatures):
            sample_error = (w1*features_row[0]+w2*features_row[1]-labels[i])**2
            totalError += sample_error
        loss.append([w1,w2,totalError])
    
loss = np.array(loss)

#See the loss in a plot
fig = plt.figure()
ax1 = fig.add_subplot(111, projection = '3d')

ax1.plot_trisurf(loss[:,0],loss[:,1],loss[:,2],cmap=cm.jet, linewidth=0)
ax1.set_xlabel('w1')
ax1.set_ylabel('w2')
ax1.set_zlabel('loss')         


#Hiplerplane that separate the two class in an optime way
A = np.zeros((4,4)) #Constants
b = np.zeros((4,1))

for i, features_row in enumerate(features):
    x = np.append([1], features_row)
    x = x.reshape((4, 1))
    y = labels[i]
    A = A+x*x.T
    b = b+x*y

inverseA = np.linalg.inv(A)

W = np.dot(inverseA,b)
X = np.arange(0,1,0.1)
Y = np.arange(0,1,0.1)

X,Y = np.meshgrid(X,Y)

Z = -(W[1]*X+W[2]*Y+W[0])/W[3]

ax.plot_surface(X,Y,Z, cmap = cm.Blues)
plt.show()









    
    
    
    
    
    
    