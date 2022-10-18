#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 13:09:38 2022

@author: anil
"""

import cv2 as cv
import numpy as np
from math import dist
import  os ,glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

# setting the path and the labels list for classification of targets on the basis in human understandable form
train_dir = os.path.join('Garbage_Dataset/Garbage classification/Garbage classification/')
labels = ['cardboard', 'glass', 'paper', 'trash']

# checking how many images from which class
for label in labels:
    directory = os.path.join(train_dir, label)
    print("Images of label \"" + label + "\":\t", len(os.listdir(directory)))

##############################################################################################
#################### GET CONTOURS ############################################################
def getCounters(img,cThr=[100,100],showCanny=False): # The function that gets contours of a image
    
    imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray,(33,33),1)
    imgCanny = cv.Canny(imgBlur,cThr[0],cThr[1])
    kernel = np.ones((3,3))
    imgDial = cv.dilate(imgCanny,kernel,iterations=3)
    imgThre = cv.erode(imgDial,kernel,iterations=1)
    
    if showCanny:
        cv.imshow('Canny',imgThre)
    
    contours, hiarchy = cv.findContours(imgThre,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    
    return  contours


##############################################################################################
#################### BIGGEST CONTOURS AREA ###################################################
def get_biggest_contour_area(img): # The function that draws a rectangle around a image 
    
    #blank = np.zeros(img.shape, dtype='uint8')
    
    contours = getCounters(img,[3,99],False)
    
    i = 0
    for cnt in contours:  
        cv.drawContours(img, contours,i, (0, 0, 255), 2)
        i = i + 1      
                
    # find the biggest countour (biggest_contour) by the area
    biggest_contour = max(contours, key = cv.contourArea)           
    rect = cv.minAreaRect(biggest_contour)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(img,[box],0,(255,0,0),2)
    
    a,b,c,d = box  
    color = (33,33,33)
    
    # Drawing line for image hight
    cv.line(img, a-7, b-7,color , 3, cv.LINE_AA)
    cv.circle(img, a-7, 5, color,-1)
    cv.circle(img, b-7, 5, color,-1)
    mid_p_h = np.array(((b[0]-a[0])/2+a[0],(b[1]-a[1])/2+a[1]),dtype=int)
    cv.circle(img, mid_p_h-7, 5, (255,255,255),-1)
    
    # Drawing line for image widht
    cv.line(img, (b[0]+7,b[1]-7), (c[0]+7,c[1]-7),color , 3, cv.LINE_AA)
    cv.circle(img, (b[0]+7,b[1]-7), 5, color,-1)
    cv.circle(img, (c[0]+7,c[1]-7), 5, color,-1)
    mid_p_w = np.array(((c[0]-b[0])/2+b[0],(c[1]-b[1])/2+b[1]),dtype=int)
    cv.circle(img, (mid_p_w[0]+7,mid_p_w[1]-7), 5, (255,255,255),-1)    
    
    Width = int(dist(a, b))
    Hight = int(dist(b, c))
    
    
    #cv.imshow('Blank', blank)
    
    # width label
    cv.putText(img,"Width : "+str(Width)+" pixel", (mid_p_w[0]-11,mid_p_w[1]-11), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
    # Higth label
    cv.putText(img,"Hight : "+str(Hight)+" pixel", (mid_p_h[0]-11,mid_p_h[1]-11), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
    
    #print(a,b,c,d)
    #print("Hight : ",Hight)
    #print("Width : ",Width)

    return Width, Hight
    

"""
img = cv.imread("glass17.jpg")
raw_img = img.copy()
img_width,img_hight = get_biggest_contour_area(img)

print(img_width,img_hight)

# show the images
cv.imshow("Result", np.hstack([raw_img, img]))
"""
#########################################################################################
#########################################################################################
file_dir = os.path.join('Garbage_Dataset/Garbage classification/Garbage classification/')
labels = ['cardboard', 'glass', 'paper', 'trash']


img_list = os.path.join(file_dir, os.listdir(file_dir)[0])   
print(img_list)         
print(len(img_list))

#plt.figure(figsize=(30,14))
records = []
for i in range(4):
    directory = os.path.join(file_dir, labels[i])
    count_img = len(directory)
    for j in range(9):
        path = os.path.join(directory, os.listdir(directory)[j])
        #image = mpimg.imread(path)
        image_cv = cv.imread(path)
        raw_img = image_cv.copy()
        
        img_width,img_hight = get_biggest_contour_area(image_cv)
        
        
        img_name = path.split('/')[-1]
        if ((img_hight/img_width) <= 0.6 or (img_hight/img_width) >= 1.5) and ((img_hight/img_width)<=3.5):
            record = [img_name,img_hight,img_width,img_hight/img_width,'GLASS']
            cv.imshow("Predict : "+record[-1], np.hstack([raw_img, image_cv]))
            #print(img_name)
        else:
            record = [img_name,img_hight,img_width,img_hight/img_width,'OTHER']
            cv.imshow("Predict : "+record[-1], np.hstack([raw_img, image_cv]))
        
        records.append(record)
        
        #plt.subplot(4, 10, i*10 + j + 1)
        #plt.imshow(image)
        #cv.imshow(label[i], image_cv)
        #if j == 0:
         #   plt.ylabel(labels[i], fontsize=20)
    print(j)

#plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
#plt.tight_layout()
#plt.show()

columns = ['File Name','Image Hight','Image Width','Aspect Ratio','Predict']
df = pd.DataFrame(data=records,columns=columns)

while True:
    
    if cv.waitKey(0)==27:
        cv.destroyAllWindows()
        break

