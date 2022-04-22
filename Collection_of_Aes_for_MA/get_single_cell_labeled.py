#!/usr/bin/env python3
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from scipy import stats
import os
import json
import pandas as pd
import cv2
import matplotlib as plt
import matplotlib.pyplot as plt
import skimage.io
from skimage.color import rgb2hed, hed2rgb
from skimage.util import crop
from skimage import data, io
from matplotlib import pyplot as plt
from scipy.spatial import distance
import imutils
import math
import csv
#%%
#get single cells with mask and aligned to train the AE unlabeled

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
#%%
#get single cells with mask and aligned to train the AE labeled

cellcounter = 0
list_of_bad_cells = []
Final_list_for_AE = []
count = 0
for filename in os.listdir("/media/lunas/Samsung USB/labeled/annotations/"):
    
    contours = filename
    size = len(contours)
    image_name = contours[:size-5]
    f = open("/media/lunas/Samsung USB/labeled/annotations/" + contours)
    #f = open("D:/26.01.22/Data/Images_Bozek/contours/" + contours)
    data = json.load(f)
    for i in data:
        x = i["geometry"]['coordinates'][0]
        #print(image_name)   
        try:
            if type(x[0][0]) == list:
                df = pd.DataFrame.from_records(x[0])
            else:  
                df = pd.DataFrame.from_records(x)

            if i["properties"] == {'isLocked': False, 'measurements': []} or len(x) < 1:
                #print(i)
                pass
            else: 
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

               
                image = cv2.imread( "/media/lunas/Samsung USB/labeled/images/" + image_name)
                
                mask = np.zeros(image.shape[:2], dtype="uint8")
                cv2.drawContours(mask, [np.int32(x).reshape((-1, 1, 2))], contourIdx = -1, color = (255, 255, 255), thickness=-1)
                masked = cv2.bitwise_and(image, image, mask=mask)
                crop_image = masked[y1:y2, x1:x2] 

                if crop_image.shape[0] == 0 or crop_image.shape[1] == 0:
                    bad_cell = [image_name, count]
                    list_of_bad_cells.append(bad_cell)
                    pass
                else:

                    path = '/media/lunas/Elements/data_ma/Single_cells_masked_aligned_labeled/' + image_name + "_" + str(count) + "_" + label + ".jpg"	
                    count = count + 1
                    cellcounter = cellcounter + 1
                    print(cellcounter)
                    liste = [path, label]
                    Final_list_for_AE.append(liste)
                    rotated = imutils.rotate_bound(crop_image, degrees*-1)
                    cv2.imwrite(path, rotated)

        except IndexError:
            pass

with open('/media/lunas/Elements/data_ma/Single_cells_masked_aligned_labeled/labels_of_cell_for_AE.csv', "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(Final_list_for_AE)
       
#%%
#get single cells with mask and aligned to train the AE labeled same format for cell names like features to compare ae and features
list_of_data = []
cellcounter = 0
list_of_bad_cells = []
Final_list_for_AE = []
count = 0
for filename in os.listdir("/media/lunas/Samsung USB/labeled/annotations/"):
    count = 0
    contours = filename
    size = len(contours)
    image_name = contours[:size-5]
    f = open("/media/lunas/Samsung USB/labeled/annotations/" + contours)
    #f = open("D:/26.01.22/Data/Images_Bozek/contours/" + contours)
    data = json.load(f)
    for i in data:
        count = count + 1
        x = i["geometry"]['coordinates'][0]
        #print(image_name)   
        try:
            if type(x[0][0]) == list:
                df = pd.DataFrame.from_records(x[0])
            else:  
                df = pd.DataFrame.from_records(x)

            if i["properties"] == {'isLocked': False, 'measurements': []} or len(x) < 1:
                #print(i)
                pass
            else: 
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

               
                image = cv2.imread( "/media/lunas/Samsung USB/labeled/images/" + image_name)
                
                mask = np.zeros(image.shape[:2], dtype="uint8")
                cv2.drawContours(mask, [np.int32(x).reshape((-1, 1, 2))], contourIdx = -1, color = (255, 255, 255), thickness=-1)
                masked = cv2.bitwise_and(image, image, mask=mask)
                crop_image = masked[y1:y2, x1:x2] 

                if crop_image.shape[0] == 0 or crop_image.shape[1] == 0:
                    bad_cell = [image_name, count]
                    list_of_bad_cells.append(bad_cell)
                    pass
                else:

                    path = '/media/lunas/Elements/data_ma/Single_cells_masked_aligned_labeled_compare_to_features/' + image_name + "_" + str(count) + "_" + label + ".jpg"	
                    #count = count + 1
                    cellcounter = cellcounter + 1
                    print(cellcounter)
                    liste = [path, label]
                    Final_list_for_AE.append(liste)
                    rotated = imutils.rotate_bound(crop_image, degrees*-1)
                    cv2.imwrite(path, rotated)
                    
                    #to comapre to features 
                    label = i["properties"]["classification"]['name']
                    entry = [filename, count, x, label, path]
                    list_of_data.append(entry)

        except IndexError:
            pass

with open('/media/lunas/Elements/data_ma/Single_cells_masked_aligned_labeled_compare_to_features/labels_of_cell_for_AE_compare_to_features.csv', "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(Final_list_for_AE)
       
with open("/media/lunas/Elements/data_ma/Single_cells_masked_aligned_labeled_compare_to_features/labled_data_out_to_compare_ae_and_features.csv", "w") as f:
    wr = csv.writer(f, delimiter=";")
    wr.writerows(list_of_data)    
#%%
#get single cells labeled and aligned
#get single cells with mask to train the AE
cellcounter = 0
list_of_bad_cells = []
Final_list_for_AE = []
count = 0
for filename in os.listdir("/media/lunas/Samsung USB/labeled/annotations/"):
#for filename in os.listdir("D:/26.01.22/Data/Images_Bozek/contours/"):
    contours = filename
    size = len(contours)
    image_name = contours[:size-5]
    f = open("/media/lunas/Samsung USB/labeled/annotations/" + contours)
    #f = open("D:/26.01.22/Data/Images_Bozek/contours/" + contours)
    data = json.load(f)
    for i in data:
        x = i["geometry"]['coordinates'][0]
        #print(image_name)   
        try:
            if type(x[0][0]) == list:
                df = pd.DataFrame.from_records(x[0])
            else:  
                df = pd.DataFrame.from_records(x)
               
            #print(df)
            #print(x)
            if i["properties"] == {'isLocked': False, 'measurements': []} or len(x) < 1:
                #print(i)
                pass
            else: 
                label = i["properties"]['classification']["name"]
                
                #print(df)
                hdist = distance.cdist(df, df, 'euclidean')
                bestpair = np.unravel_index(hdist.argmax(), hdist.shape)
                x1 = int(df.iloc[bestpair[1]].values[0])
                x2 = int(df.iloc[bestpair[0]].values[0])
                y1 = int(df.iloc[bestpair[1]].values[1])
                y2 = int(df.iloc[bestpair[0]].values[1])
                
                
                dx = x2 - x1
                dy = y2 - y1
                print(dx, dy)
                angle = math.atan2(dy, dx)
                degrees = math.degrees(angle)
                print(angle)
                print(degrees)
                #print(bestpair)
                #print([df.iloc[bestpair[0]].values,df.iloc[bestpair[1]].values])
                x1 = int(df.agg([min, max])[0]["min"])
                x2 = int(df.agg([min, max])[0]["max"])
                y1 = int(df.agg([min, max])[1]["min"])
                y2 = int(df.agg([min, max])[1]["max"])
                count = count + 1
                #print(x1, x2, y1, y2)
                #print("x:", x2-x1)
                #print("y:", y2-y1)
                if count > 5:
                    break
                else:
                    pass
               
                #print(x1, x2, y1, y2)
                #image = cv2.imread( "D:/26.01.22/Data/Images_Bozek/images/" + image_name) 
                #print(df.iloc[bestpair[0]].values[0])
                image = cv2.imread( "/media/lunas/Samsung USB/labeled/images/" + image_name)
                cv2.line(image,(int(df.iloc[bestpair[0]].values[0]),int(df.iloc[bestpair[0]].values[1])),(int(df.iloc[bestpair[1]].values[0]),int(df.iloc[bestpair[1]].values[1])),(255,0,0),1)
                #rotated = imutils.rotate_bound(image, degrees)
                #plt.imshow(rotated)
                crop_image = image[y1:y2, x1:x2] 
                if crop_image.shape[0] == 0 or crop_image.shape[1] == 0:
                    bad_cell = [image_name, count]
                    list_of_bad_cells.append(bad_cell)
                    pass
                else:
                    #path = '/media/lunas/Samsung USB/data/Single_cell_images_labeled_for_AE/' + image_name + "_" + str(count) + "_" + label + ".jpg"
                    path = '/media/lunas/Elements/data_ma/single_cells_labeled/' + image_name + "_" + str(count) + "_" + label + ".jpg"
                    #cv2.imwrite('D:/26.01.22/Data/Images_Bozek/single_cell_images/' + image_name + "_" + str(count) + ".jpg", crop_image)	
                    #cv2.imwrite(path, crop_image)	
                    count = count + 1
                    cellcounter = cellcounter + 1
                    print(cellcounter)
                    liste = [path, label]
                    #Final_list_for_AE.append(liste)
                    #rotated = imutils.rotate(crop_image, degrees)
                    rotated = imutils.rotate_bound(crop_image, degrees*-1)
                    plt.imshow(rotated)
                    #plt.imshow(crop_image)
                
    #print(Final_list_for_AE)
        except IndexError:
            pass

#with open("/media/lunas/Samsung USB/data/Single_cell_images_labeled_for_AE/test123.csv", "w", newline="") as f:
#with open('/media/lunas/Elements/data_ma/single_cells_labeled/labels_of_cell_for_AE.csv', "w", newline="") as f:
#    writer = csv.writer(f)
#    writer.writerows(Final_list_for_AE)

#%%
#get single cells with mask to train the AE
cellcounter = 0
list_of_bad_cells = []
Final_list_for_AE = []
for filename in os.listdir("/media/lunas/Samsung USB/labeled/annotations/"):
#for filename in os.listdir("D:/26.01.22/Data/Images_Bozek/contours/"):
    contours = filename
    size = len(contours)
    image_name = contours[:size-5]
    f = open("/media/lunas/Samsung USB/labeled/annotations/" + contours)
    #f = open("D:/26.01.22/Data/Images_Bozek/contours/" + contours)
    data = json.load(f)

    count = 0
    for i in data:
        x = i["geometry"]['coordinates'][0]
        #print(image_name)
        try:
            if type(x[0][0]) == list:
                df = pd.DataFrame.from_records(x[0])
            else:  
                df = pd.DataFrame.from_records(x)
            #if len(df) <= 4:
            #print(df)
            #print(x)
            if i["properties"] == {'isLocked': False, 'measurements': []} or len(x) < 1:
                #print(i)
                pass
            else: 
                label = i["properties"]['classification']["name"]
                
                
                x1 = int(df.agg([min, max])[0]["min"])
                x2 = int(df.agg([min, max])[0]["max"])
                y1 = int(df.agg([min, max])[1]["min"])
                y2 = int(df.agg([min, max])[1]["max"])
        
        #        if all([v-4 >= 0 for v in [x1,y1]]) and all([v+4 <= 512 for v in [x2,y2]]):
        #            x1 = x1 - 1 
        #            y1 = y1 - 1
        #            x2 = x2 + 1
        #            y2 = y2 + 1
        
        
        #        else:
        #            pass
                
                #print(x1, x2, y1, y2)
                #image = cv2.imread( "D:/26.01.22/Data/Images_Bozek/images/" + image_name)  
                image = cv2.imread( "/media/lunas/Samsung USB/labeled/images/" + image_name)  
                crop_image = image[y1:y2, x1:x2] 
                if crop_image.shape[0] == 0 or crop_image.shape[1] == 0:
                    bad_cell = [image_name, count]
                    list_of_bad_cells.append(bad_cell)
                    pass
                else:
                    #path = '/media/lunas/Samsung USB/data/Single_cell_images_labeled_for_AE/' + image_name + "_" + str(count) + "_" + label + ".jpg"
                    path = '/media/lunas/Elements/data_ma/single_cells_labeled/' + image_name + "_" + str(count) + "_" + label + ".jpg"
                    #cv2.imwrite('D:/26.01.22/Data/Images_Bozek/single_cell_images/' + image_name + "_" + str(count) + ".jpg", crop_image)	
                    cv2.imwrite(path, crop_image)	
                    count = count + 1
                    cellcounter = cellcounter + 1
                    print(cellcounter)
                    liste = [path, label]
                    Final_list_for_AE.append(liste)
                    #if cellcounter > 10:
                    #    break
                
    #print(Final_list_for_AE)
        except IndexError:
            pass
#%%
#with open("/media/lunas/Samsung USB/data/Single_cell_images_labeled_for_AE/test123.csv", "w", newline="") as f:
with open('/media/lunas/Elements/data_ma/single_cells_labeled/labels_of_cell_for_AE.csv', "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(Final_list_for_AE)

#%%
Final_list_for_AE = []
#for filename in os.listdir("/media/lunas/Samsung USB/data/Single_cell_images_labeled_for_AE"):
for filename in os.listdir('/media/lunas/Elements/data_ma/single_cells_labeled/'):
    first = filename
    second = first[:len(first)-4]
    second = second[::-1].split("_")[0][::-1]
    line = [first, second]
    Final_list_for_AE.append(line)
print(len(Final_list_for_AE))
#%%
#with open("/media/lunas/Samsung USB/data/Data_and_labels_for_AE.csv", "w", newline="") as f:
with open('/media/lunas/Elements/data_ma/single_cells_labeled/', "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(Final_list_for_AE)