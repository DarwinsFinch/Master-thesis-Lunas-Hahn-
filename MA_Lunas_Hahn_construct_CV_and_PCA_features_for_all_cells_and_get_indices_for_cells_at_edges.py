#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 18:42:19 2022
@author: lunas
"""

#This script calculates the CV and PCA features for all cells
#It also creates a list of indices for cells that are located at the edges to be removed
#import packages
import cv2
from  matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from scipy import stats
import math
from skimage import data
from skimage.color import rgb2hed, hed2rgb

#%%
#define function to calculate curvature
def getCurvature(contour,stride=1):
 curvature=[]
 assert stride<len(contour),"stride must be shorther than length of contour"
 
 for i in range(len(contour)):
    
    before=i-stride+len(contour) if i-stride<0 else i-stride
    after=i+stride-len(contour) if i+stride>=len(contour) else i+stride
    
    f1x,f1y=(contour[after]-contour[before])/stride
    f2x,f2y=(contour[after]-2*contour[i]+contour[before])/stride**2
    denominator=(f1x**2+f1y**2)**3+1e-11
    
    curvature_at_i=np.sqrt(4*(f2y*f1x-f2x*f1y)**2/denominator) if denominator > 1e-12 else -1

    curvature.append(curvature_at_i)
    
 return curvature

#%%
#load table with cell name and contour information
My_Data = [] 
with open ("/media/lunas/Elements/combined_datasets_all_cells/All_cells_labled_data_out_interpolated.csv") as f:
  for i in f:
    x = i.strip("\n").split(";")

    my_contours = []
    count = 0
    for i in x[4].replace("[", "").replace("]", "").split():
      if (count % 2) == 0:
        x1 = float(i.replace(",", "").replace("'",""))
        #x1 = x1.astype(int)
        count = count + 1 
      else:
        y1 = float(i.replace(",", "").replace("'", ""))
        #y1 = y1.astype(int)
        entry = [x1, y1]
        my_contours.append(entry)
        count = count + 1  

    x[4] = my_contours
    My_Data.append(x)

#My_Data
#%%
#get list of interpolated contours
contours1 = []
for i in My_Data:
    contours1.append(np.array(i[4]))
print(contours1[0])    
#%%
#remove cells at edges. This creates a list of indices for the cells that are located at the edges and should be removed later in the analysis
import csv 
Potentially_remove_these_cells = []
count = 0
for i in My_Data:
    min_count = 0
    max_count = 0
    count = count + 1 
    for e in i[2].strip("[]").split():
        num = float(e.strip(",").strip("[]").strip("''"))
        if num == 0:
            min_count = min_count + 1
        elif num == 512:
            max_count = max_count + 1
        if min_count or max_count >= 2:
            #print(num)
            entry = str(i[0] + "_" + i[1] + "_" + i[3])
            #print(entry)
            List_of_cells_at_the_edge = [entry, (count - 1)]
            Potentially_remove_these_cells.append(List_of_cells_at_the_edge)
            break
    #print(i[2].strip("[]").split()[51].strip(",").strip("[]").strip("''"))
print(len(Potentially_remove_these_cells))

with open("/home/lunas/Documents/Uni/Masterarbeit/Data/Cells_at_the_edges_from_interpolate_contours_from_labeled_all_cells.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(Potentially_remove_these_cells)



#%%
#Calculate CV morphology features based on interpolated contours for each cell
cry = 0
fun = 0
feature_df = []
for i in contours1:
  List_of_features = []
  cnt = i
  cnt = cnt.astype(int)
  M = cv2.moments(cnt)
  #filter out cells where interpolation did not work
  if len(cnt) > 10:
      area = cv2.contourArea(cnt)
      List_of_features.append(area)
      perimeter = cv2.arcLength(cnt,True)
      List_of_features.append(perimeter)
      (x,y),radius = cv2.minEnclosingCircle(cnt)
      List_of_features.append(radius)
      x,y,w,h = cv2.boundingRect(cnt)
      aspect_ratio = float(w)/h
      List_of_features.append(aspect_ratio)
      rect_area = w*h
      extent = float(area)/rect_area
      List_of_features.append(extent)
      hull = cv2.convexHull(cnt)
      hull_area = cv2.contourArea(hull)
      solidity = float(area)/hull_area
      List_of_features.append(solidity)
      equi_diameter = np.sqrt(4*area/np.pi)
      List_of_features.append(equi_diameter)
      ellipse = cv2.fitEllipse(cnt)
      center, axes, orientation = ellipse
      majoraxis_length = max(axes)
      minoraxis_length = min(axes)
      eccentricity = np.sqrt(1-(minoraxis_length/majoraxis_length)**2)
      List_of_features.append(eccentricity)
      mean_cuvature = np.mean(getCurvature(cnt))
      List_of_features.append(mean_cuvature)
      std_cuvature = np.std(getCurvature(cnt))
      List_of_features.append(std_cuvature)      
      cx = int(M['m10']/M['m00'])
      cy = int(M['m01']/M['m00'])
      list_of_lengths = []
      for e in cnt:
          x = e[0]
          y = e[1]
          calcx = x - cx
          #print(calcx)
          calcx1 = calcx**2
          #print(calcx1)
          calcy = y - cy
          calcy1 = calcy**2
          length = calcx1 + calcy1
          if length > 0:
              length = math.sqrt(length)
          else:
              length = np.NaN
              fun = fun + 1
          list_of_lengths.append(length)
      roughness = np.var(list_of_lengths)
      List_of_features.append(roughness)
      cry = cry + 1
      feature_df.append(List_of_features)
      print(cry)
  else:
      area = math.nan
      List_of_features.append(area)
      perimeter = math.nan
      List_of_features.append(perimeter)
      radius = math.nan
      List_of_features.append(radius)
      aspect_ratio = math.nan
      List_of_features.append(aspect_ratio)
      rect_area = math.nan
      extent = math.nan
      List_of_features.append(extent)
      hull = math.nan
      hull_area = math.nan
      solidity = math.nan
      List_of_features.append(solidity)
      equi_diameter = math.nan
      List_of_features.append(equi_diameter)
      ellipse = math.nan
      #center, axes, orientation = ellipse
      majoraxis_length = math.nan
      minoraxis_length = math.nan
      eccentricity = math.nan
      List_of_features.append(eccentricity)
      mean_cuvature = math.nan
      List_of_features.append(mean_cuvature)
      std_cuvature = math.nan
      List_of_features.append(std_cuvature)
      roughness =math.nan
      List_of_features.append(roughness)
      cry = cry + 1
      feature_df.append(List_of_features)
      

    
    
#%%
#safe CV morphology features
df_contour_features = pd.DataFrame.from_records(feature_df)
df_contour_features.to_csv("//home/lunas/Documents/Uni/Masterarbeit/Data/pd_contourfeatures_from_interpolate_contours_from_labeled_all_cells.csv")

#%%
#Calculate CV pixel features based on interpolated contours for each cell
count = 0
Pixel_features = []
for i in My_Data:
    name = i[0]
    cnt = contours1[count]
    cnt = cnt.astype(int)
    pic = name[:len(name)-5]
    fetch = '/media/lunas/Elements/combined_datasets_all_cells/images/' + pic
    image= cv2.imread(fetch)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    mask = np.zeros(image.shape,np.uint8)
    cv2.drawContours(mask,[cnt],0,255,-1)
    pixelpoints = cv2.findNonZero(mask)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(image,mask = mask)
    mean_val = cv2.mean(image ,mask = mask)[0]
    #print(mean_val, min_val, max_val, min_loc, max_loc)
    line = [mean_val, min_val, max_val]
    Pixel_features.append(line)
    count = count + 1

#%%
#Calculate CV pixel features based on interpolated contours for each cell for the H&E channels
count = 0
Pixel_features_h_channel = []
Pixel_features_e_channel = []

for i in My_Data:
    name = i[0]
    cnt = contours1[count]
    cnt = cnt.astype(int)
    pic = name[:len(name)-5]
    fetch = '/media/lunas/Elements/combined_datasets_all_cells/images/' + pic
    image= cv2.imread(fetch)
    tile_hed = rgb2hed(image)
    
    # hematoxylin channel
    h_channel = tile_hed[:, :, 0]
    # eosin channel
    e_channel = tile_hed[:, :, 1]
    
    #get pixel features from h_channel
    mask = np.zeros(h_channel.shape,np.uint8)
    cv2.drawContours(mask,[cnt],0,255,-1)
    pixelpoints = cv2.findNonZero(mask)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(h_channel,mask = mask)
    mean_val = cv2.mean(h_channel ,mask = mask)[0]
    #print(mean_val, min_val, max_val)
    line = [mean_val, min_val, max_val]
    Pixel_features_h_channel.append(line)

    
    #get pixel features from e_channel
    mask = np.zeros(e_channel.shape,np.uint8)
    cv2.drawContours(mask,[cnt],0,255,-1)
    pixelpoints = cv2.findNonZero(mask)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(e_channel,mask = mask)
    mean_val = cv2.mean(e_channel ,mask = mask)[0]
    #print(mean_val, min_val, max_val)
    line = [mean_val, min_val, max_val]
    Pixel_features_e_channel.append(line)
    count = count + 1
    
    
    
    
    
#%%
#safe CV pixel features
pd_pixelfeatures = pd.DataFrame.from_records(Pixel_features)
pd_pixelfeatures.to_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/pd_pixelfeatures_from_interpolate_contours_from_labeled_all_cells.csv")

#%%
#safe CV morphology features for hed channel
pd_pixelfeatures_h_channel = pd.DataFrame.from_records(Pixel_features_h_channel)
pd_pixelfeatures_e_channel = pd.DataFrame.from_records(Pixel_features_e_channel)
pd_pixelfeatures = pd.concat([pd_pixelfeatures_h_channel, pd_pixelfeatures_e_channel], axis=1)
pd_pixelfeatures.to_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/pd_pixelfeatures_from_HED_channel_from_interpolate_contours_from_labeled_all_cells.csv")

#%%
#flatten interpolated contours and use the flattend contours to create the PCA features
#flatten 
import copy
Flattend_shapes = []
My_Data_4 = []
for x in My_Data:
    count = 0
    my_contours_aligned = []
    for i in x[4]:
        x1 = i[0]
        y1 = i[1]
        entry = [x1, y1]
        my_contours_aligned.append(entry)
        count = count + 1  

    x[4] = np.asarray(my_contours_aligned) 
    Flattend_shapes.append(np.asarray(my_contours_aligned).flatten())#get flattened array for pca
    My_Data_4.append(x)

PD_Flattend_shapes = pd.DataFrame(list(map(np.ravel, Flattend_shapes)))

#%%
#PCA
import cv2
from  matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from  matplotlib import pyplot as plt
 
#%%
x = StandardScaler().fit_transform(PD_Flattend_shapes)
pca = PCA(n_components=100)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents)


#%%
#safe PCA features
principalDf.to_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/100_principal_components_from_interpolation_contours_from_labeled_all_cells.csv")





