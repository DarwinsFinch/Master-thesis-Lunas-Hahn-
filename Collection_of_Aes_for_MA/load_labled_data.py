#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 21:20:26 2022

@author: lunas
"""

import os
import json
import numpy as np
from scipy import interpolate
from scipy.interpolate import splprep, splev

#%%

list_of_data = []

for filename in os.listdir("/home/lunas/Documents/Uni/Masterarbeit/Data/Images_Bozek/labeled/annotations"):
  if filename.endswith(".json"): 
    #print(os.path.join(filename))
    f = open("/home/lunas/Documents/Uni/Masterarbeit/Data/Images_Bozek/labeled/annotations/" + filename)
    data = json.load(f)
    count = 0 
    for i in data:
      count = count + 1
      x = i["geometry"]['coordinates'][0]
      if i["properties"] == {'isLocked': False, 'measurements': []} or len(x) < 1:
          print(i)
      #if filename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no0_tile_no24.png.json" and count == 98:
      #    print(i["properties"])
      #elif filename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no33_tile_no48.png.json" and count == 69:
      #    print(i["properties"])
      else:  
          #print(filename, count)
          label = i["properties"]["classification"]['name']
          entry = [filename, count, x, label]
          list_of_data.append(entry)
      #if i["properties"]['classification']['name'] == True:
      #    pass
      #else:
      #    print(filename, count)
      
print(len(list_of_data))

#%%
print(list_of_data[0])
print(i)
#%%
import csv

with open("/home/lunas/Documents/Uni/Masterarbeit/Data/labled_data_out.csv", "w") as f:
    wr = csv.writer(f, delimiter=";")
    wr.writerows(list_of_data)
#%%
#do this in bash
#cat labled_data_out.csv | cut -d";" -f1,2,4 > data_only_lables_labled.csv
#tr ";" "," < data_only_lables_labled.csv > data_only_lables_labled_1.csv

My_Data = [] 
with open ("/home/lunas/Documents/Uni/Masterarbeit/Data/labled_data_out.csv") as f:
  for i in f:
    x = i.strip("\n").split(";")

    my_contours = []
    count = 0
    for i in x[2].replace("[", "").replace("]", "").split():
      if (count % 2) == 0:
        x1 = i.replace(",", "")
        count = count + 1 
      else:
        y1 = i.replace(",", "")
        entry = [x1, y1]
        my_contours.append(entry)
        count = count + 1  

    x[2] = my_contours
    My_Data.append(x)
  
#%%
print(len(My_Data))
#%%     
for i in My_Data:
    if len(i) == 4 and len(i[0]):
        pass
    else:
        print(i)
#%%   
loss = 0
count = 0
for i in My_Data:

  x = []
  y = []
  if len(i[2]) > 1:
      for e in i[2]:
        x.append(float(e[0]))
        y.append(float(e[1]))
      p = np.asarray(x)
      q = np.asarray(y)
      #print(i)
      tck,u = interpolate.splprep([p,q], s = 0) #s=0 for little smoothing.  The user can use s to control the trade-off between closeness and smoothness of fit
      xi, yi = splev(np.linspace(0, 1, 50), tck) #choose to interpolate 50 points since this is the average 
      xy = list(map(list, zip(xi, yi)))
      My_Data[count].append(xy)
      count = count + 1 
  else:
      My_Data.pop(count - loss)
      loss = loss + 1
      print(i)
      
print(loss, len(My_Data)) 
#%% 
#%%
print(My_Data[0][2])
#%%
 
count = 0      
for i in My_Data:
    count = count + 1
    x = []
    y = []
    
    if count < 2:
        for e in i[2]:
            print(e[0])
            pass        

#%%  
print(My_Data[0])        
 

#%%
import numpy as np
import pandas as pd
import math

loss = 0
#calculate rescaling factor R
def R_rescale(contours):
    count = 0
    sum = 0
    for (x, y) in contours:
        count = count + 1
        sum = sum + (x**2 + y**2)
        R = math.sqrt(sum/count)
        return R

#center around 0/0
Final = []
for counter, value in enumerate(My_Data):
    if len(value) == 5:
        #print(value)
        df = pd.DataFrame(value[4], columns = ['X','Y'])
        df = df - df.mean()
        centered = df.to_numpy()
        R = R_rescale(centered)

        centered_rescaled = []
        for (x, y) in centered:
            x = x / R
            y = y / R
            z = (x, y)
            centered_rescaled.append(z)

  #else:
   # break

        Final.append(np.asarray(centered_rescaled))
    else:
        My_Data.pop(counter - loss)
        loss = loss + 1
        print(value)
        
print(len(My_Data), len(Final), loss)
#%%
import numpy as np
from scipy.stats import ortho_group
from procrustes import orthogonal
from procrustes import generalized

array_aligned, new_distance_gpa = generalized(Final)
len(array_aligned)
#%%
import cv2
from  matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from scipy import stats
import math



#%%

#%%
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


#%%
My_Data = [] 
with open ("/home/lunas/Documents/Uni/Masterarbeit/Data/labled_data_out.csv") as f:
  for i in f:
    x = i.strip("\n").split(";")

    my_contours = []
    count = 0
    for i in x[2].replace("[", "").replace("]", "").split():
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

    x[2] = my_contours
    My_Data.append(x)
print(My_Data)





#%%


contours = []
for i in My_Data:
    contours.append(np.array(i[2]))
print(contours[0])    
    
#%%
cry = 0
fun = 0
feature_df = []
for i in contours:
  List_of_features = []
  cnt = i
  cnt = cnt.astype(int)
  M = cv2.moments(cnt) #moments als features?
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
      #print(area, hull_area)
      #print(len(cnt))
      #print(len(My_Data[cry][2]))
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
      #M = cv2.moments(cnt)      
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
          #calc = calc + ((x - cx)**2 + (y - cy)**2)
      #calc = math.sqrt(calc)
      roughness = np.var(list_of_lengths)
      List_of_features.append(roughness)
      cry = cry + 1
      feature_df.append(List_of_features)
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
      #print(area, hull_area)
      #print(len(cnt))
      #print(len(My_Data[cry][2]))
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
      
      print(cry)

    
    
#%%
df_contour_features = pd.DataFrame.from_records(feature_df)
print(df_contour_features.shape)
#%%
df_contour_features.to_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/pd_contourfeatures_from_og_contours_labled.csv")
#%%
print(df_contour_features.head())
#%%
print(df_contour_features.head())





#%%
#pixel
count = 0
Pixel_features = []
for i in My_Data:
    name = i[0]
    cnt = contours[count]
    cnt = cnt.astype(int)
    pic = name[:len(name)-5]
    fetch = '/home/lunas/Documents/Uni/Masterarbeit/Data/Images_Bozek/labeled/images/' + pic
    image= cv2.imread(fetch)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY )
    mask = np.zeros(image.shape,np.uint8)
    cv2.drawContours(mask,[cnt],0,255,-1)
    pixelpoints = cv2.findNonZero(mask)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(image,mask = mask)
    mean_val = cv2.mean(image ,mask = mask)[0]
    #print(mean_val, min_val, max_val, min_loc, max_loc)
    line = [mean_val, min_val, max_val]
    Pixel_features.append(line)
    count = count + 1
    print(count)
#%%

#%%
pd_pixelfeatures = pd.DataFrame.from_records(Pixel_features)
print(pd_pixelfeatures.shape)
#%%
pd_pixelfeatures.to_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/pd_pixelfeatures_from_og_contours_labled.csv")




#print(cnt)

#print(Load_Procrustes[0])





#%%
import cv2
from  matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#%%



lables = pd.read_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/data_only_lables_labled_1.csv", header = None)#, index_col=0)  
pixel = pd.read_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/pd_pixelfeatures_from_og_contours_labled.csv")
contour = pd.read_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/pd_contourfeatures_from_og_contours_labled.csv")


#pca = pca.iloc[: , 1:]
pixel = pixel.iloc[: , 1:]
contour = contour.iloc[: , 1:]

#%%
print(lables.head(5))
#%%
print(pixel.head(5))
#%%
print(contour.head(5))
#%%
#print(pca.head(5))
#%%
print(len(lables), len(pixel), len(contour))
#%%

X = pd.concat([lables, contour, pixel], axis=1, ignore_index=True)
print(X.shape)

#%%




colnames =  ["image", "cell","class","area", "perimeter", "radius", "aspect_ratio", "extent", "solidity", "equi_diameter","eccentricity","mean_cuvature","std_cuvature","roughness", "mean_val", "min_val", "max_val"]
#%%
X.columns = colnames
#%%
X.to_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/Feature_gathering_from_og_contours_labled.csv", index=False)




###feature analysis
#%%
import cv2
from  matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from scipy import stats
#%%
features = pd.read_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/Feature_gathering_from_og_contours_labled.csv")
#%%

#%%

#%%
def correl(*args, **kwargs):
    # determine pearson correlation between each feature
    corr_r = (args[0].corr(args[1], 'pearson'))
    # annotation of correlation values (number of fractions)
    corr_text = f"{corr_r:2.2f}"
    # set axes properties
    ax = plt.gca()
    ax.set_axis_off()
    # size and settings of colored squares for each entry
    marker_size = 15000
    ax.scatter([.5], [.5], s=marker_size, c=[corr_r],  marker="s",
               cmap="RdBu", vmin=-1, vmax=1, transform=ax.transAxes)
    font_size = 45
    # use white annotation if  marker color is dark
    if abs(corr_r) >= 0.7:
        clr = "white"
    # use black annotation if  marker color is bright    
    else:        
        clr = "black"
    # set position of annotation (centering)   
    plt.annotate(corr_text, [0.5, 0.5,],  xycoords="axes fraction",
                ha='center', va='center', fontsize=font_size, color=clr)
    return ax

#%%
g = sns.PairGrid(features.iloc[:, 3:16], diag_sharey=False) # call seaborn's PairGrid function
g.map_lower(plt.scatter,edgecolors="w")     # show scatter plot matrix on lower triangle
g.map_diag(sns.distplot)                    # show distributions on main diagonal
g.map_upper(correl)                         # show correlation matrix on lower triangle (using function above)
plt.savefig("/home/lunas/Documents/Uni/Masterarbeit/Data/plots/Features_Pairgrid_from_og_contours_labled.png")
plt.show()
