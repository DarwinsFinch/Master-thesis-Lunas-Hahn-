# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 13:08:34 2022

@author: lunas
"""

###create input file for AE

import cv2
from  matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import json
from scipy.spatial import distance
import imutils
import math

#%%
#get single cells with mask and aligned to train the AE unlabeled

cellcounter = 0
list_of_bad_cells = []

#iterate through dic and load images
for filename in os.listdir("/media/lunas/Samsung USB/data/contours/"):

    contours = filename
    size = len(contours)
    image_name = contours[:size-5]
    f = open("/media/lunas/Samsung USB/data/contours/" + contours)
    data = json.load(f)

    count = 0

    for i in data:
        x = i["geometry"]['coordinates'][0]
        if type(x[0][0]) == list:
            df = pd.DataFrame.from_records(x[0])
        else:  
            df = pd.DataFrame.from_records(x)
            
        #get label
        label = i["properties"]['classification']["name"]   
        
        #aligne image
        hdist = distance.cdist(df, df, 'euclidean')
        bestpair = np.unravel_index(hdist.argmax(), hdist.shape)
        x1 = int(df.iloc[bestpair[1]].values[0])
        x2 = int(df.iloc[bestpair[0]].values[0])
        y1 = int(df.iloc[bestpair[1]].values[1])
        y2 = int(df.iloc[bestpair[0]].values[1])
        
        
        dx = x2 - x1
        dy = y2 - y1
        angle = math.atan2(dy, dx)
        degrees = math.degrees(angle)
        
        x1 = int(df.agg([min, max])[0]["min"])
        x2 = int(df.agg([min, max])[0]["max"])
        y1 = int(df.agg([min, max])[1]["min"])
        y2 = int(df.agg([min, max])[1]["max"])

        if all([v-4 >= 0 for v in [x1,y1]]) and all([v+4 <= 512 for v in [x2,y2]]):
            x1 = x1 - 4 
            y1 = y1 - 4
            x2 = x2 + 4
            y2 = y2 + 4


        image = cv2.imread("/media/lunas/Samsung USB/data/images/" + image_name)
        #cv2.line(image,(int(df.iloc[bestpair[0]].values[0]),int(df.iloc[bestpair[0]].values[1])),(int(df.iloc[bestpair[1]].values[0]),int(df.iloc[bestpair[1]].values[1])),(255,0,0),1)
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [np.int32(x).reshape((-1, 1, 2))], contourIdx = -1, color = (255, 255, 255), thickness=-1)
        masked = cv2.bitwise_and(image, image, mask=mask)
        
        #crop image
        crop_image = masked[y1:y2, x1:x2] 
       
        #QC
        if crop_image.shape[0] == 0 or crop_image.shape[1] == 0:
            bad_cell = [image_name, count]
            list_of_bad_cells.append(bad_cell)
            pass
        
        #save image
        else:
            my_path = '/media/lunas/Elements/data_ma/Single_cells_masked_aligned_unlabeled/' + image_name + "_" + str(count) + "_" + label + ".jpg"
            #cv2.imwrite(my_path + image_name + "_" + str(count) + "_"  + ".jpg", crop_image)	
            count = count + 1
            cellcounter = cellcounter + 1
            rotated = imutils.rotate_bound(crop_image, degrees*-1)
            cv2.imwrite(my_path, rotated)
            print(cellcounter) 
            plt.imshow(rotated)
#%%
#get single cells with mask and aligned to train the AE unlabeled, 

cellcounter = 0
list_of_bad_cells = []

for filename in os.listdir("/media/lunas/Samsung USB/data/contours/"):

    contours = filename
    size = len(contours)
    image_name = contours[:size-5]
    f = open("/media/lunas/Samsung USB/data/contours/" + contours)
    data = json.load(f)

    count = 0

    for i in data:
        x = i["geometry"]['coordinates'][0]
        if type(x[0][0]) == list:
            df = pd.DataFrame.from_records(x[0])
        else:  
            df = pd.DataFrame.from_records(x)
            
        label = i["properties"]['classification']["name"]
        
        hdist = distance.cdist(df, df, 'euclidean')
        bestpair = np.unravel_index(hdist.argmax(), hdist.shape)
        x1 = int(df.iloc[bestpair[1]].values[0])
        x2 = int(df.iloc[bestpair[0]].values[0])
        y1 = int(df.iloc[bestpair[1]].values[1])
        y2 = int(df.iloc[bestpair[0]].values[1])
        
        
        dx = x2 - x1
        dy = y2 - y1
        angle = math.atan2(dy, dx)
        degrees = math.degrees(angle)
        
        x1 = int(df.agg([min, max])[0]["min"])
        x2 = int(df.agg([min, max])[0]["max"])
        y1 = int(df.agg([min, max])[1]["min"])
        y2 = int(df.agg([min, max])[1]["max"])

        if all([v-4 >= 0 for v in [x1,y1]]) and all([v+4 <= 512 for v in [x2,y2]]):
            x1 = x1 - 4 
            y1 = y1 - 4
            x2 = x2 + 4
            y2 = y2 + 4


        image = cv2.imread("/media/lunas/Samsung USB/data/images/" + image_name)
        #cv2.line(image,(int(df.iloc[bestpair[0]].values[0]),int(df.iloc[bestpair[0]].values[1])),(int(df.iloc[bestpair[1]].values[0]),int(df.iloc[bestpair[1]].values[1])),(255,0,0),1)
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [np.int32(x).reshape((-1, 1, 2))], contourIdx = -1, color = (255, 255, 255), thickness=-1)
        masked = cv2.bitwise_and(image, image, mask=mask)
        crop_image = masked[y1:y2, x1:x2] 
       
        if crop_image.shape[0] == 0 or crop_image.shape[1] == 0:
            bad_cell = [image_name, count]
            list_of_bad_cells.append(bad_cell)
            pass
        else:
            my_path = '/media/lunas/Elements/data_ma/Single_cells_masked_aligned_unlabeled/' + image_name + "_" + str(count) + "_" + label + ".jpg"
            #cv2.imwrite(my_path + image_name + "_" + str(count) + "_"  + ".jpg", crop_image)	
            count = count + 1
            cellcounter = cellcounter + 1
            rotated = imutils.rotate_bound(crop_image, degrees*-1)
            cv2.imwrite(my_path, rotated)
            print(cellcounter) 
            plt.imshow(rotated)            