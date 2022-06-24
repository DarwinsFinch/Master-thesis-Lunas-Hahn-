
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:16:20 2022
@author: lunas
"""
#%%

#import packages
import os
import json
import numpy as np
from scipy import interpolate
from scipy.interpolate import splprep, splev

#%%
#Read in the information for the single cells from the slides and their contour points. 
#Create a table with cell name, number, contour points and class
list_of_data = []

for filename in os.listdir("/media/lunas/Elements/combined_datasets_all_cells/contours/"):
  if filename.endswith(".json"): 
    f = open("/media/lunas/Elements/combined_datasets_all_cells/contours/" + filename)
    data = json.load(f)
    count = 0 
    for i in data:
      count = count + 1
      x = i["geometry"]['coordinates'][0]
      if i["properties"] == {'isLocked': False, 'measurements': []} or len(x) < 1:
          print(i)
      else:  
          label = i["properties"]["classification"]['name']
          entry = [filename, count, x, label]
          list_of_data.append(entry)      
print(len(list_of_data))


#%%
#safe the table as csv
import csv

with open("/media/lunas/Elements/combined_datasets_all_cells/data_out_all_cells.csv", "w") as f:
    wr = csv.writer(f, delimiter=";")
    wr.writerows(list_of_data)
#%%
#clean the table to use it for interpolation

#do this in bash
#cat labled_data_out.csv | cut -d";" -f1,2,4 > data_only_lables_labled.csv
#tr ";" "," < data_only_lables_labled.csv > data_only_lables_labled_1.csv

My_Data = [] 
with open ("/media/lunas/Elements/combined_datasets_all_cells/data_out_all_cells.csv") as f:
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
#interpolate the data points to get 50 points for each cell 
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
        

#%%  
#safe table with interpolated points as csv
print(My_Data[0])        
 
import csv

with open("/media/lunas/Elements/combined_datasets_all_cells/All_cells_labled_data_out_interpolated.csv", "w") as f:
    wr = csv.writer(f, delimiter=";")
    wr.writerows(My_Data)