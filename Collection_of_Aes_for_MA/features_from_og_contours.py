# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 12:59:36 2022

@author: lunas
"""
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
with open ("/home/lunas/Documents/Uni/Masterarbeit/Data/data_aligned_cell_shapes.csv") as f:
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

#My_Data
#%%
print(My_Data[0])
#%%
less_than_10_contours = []
for i in My_Data:
    if len(i[2]) <= 5:
        liste = [i[0], i[1]]
        less_than_10_contours.append(liste)
        print(i[0])

#print(type(My_Data[0][2]))

#print(np.array(My_Data[0][2]))
#%%
print(len(less_than_10_contours))
#%%



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
df_contour_features.to_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/pd_contourfeatures_from_og_contours.csv")
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
    fetch = '/home/lunas/Documents/Uni/Masterarbeit/Data/Images_Bozek/images/' + pic
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
pd_pixelfeatures.to_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/pd_pixelfeatures_from_og_contours.csv")




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



lables = pd.read_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/data_only_lables_1.csv", header = None)#, index_col=0)  
pixel = pd.read_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/pd_pixelfeatures_from_og_contours.csv")
contour = pd.read_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/pd_contourfeatures_from_og_contours.csv")


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




colnames =  ["image", "cell","area", "perimeter", "radius", "aspect_ratio", "extent", "solidity", "equi_diameter","eccentricity","mean_cuvature","std_cuvature","roughness", "mean_val", "min_val", "max_val"]
#%%
X.columns = colnames
#%%
X.to_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/Feature_gathering_from_og_contours.csv", index=False)




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
features = pd.read_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/Feature_gathering_from_og_contours.csv")
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
g = sns.PairGrid(features.iloc[:, 2:16], diag_sharey=False) # call seaborn's PairGrid function
g.map_lower(plt.scatter,edgecolors="w")     # show scatter plot matrix on lower triangle
g.map_diag(sns.distplot)                    # show distributions on main diagonal
g.map_upper(correl)                         # show correlation matrix on lower triangle (using function above)
plt.savefig("/home/lunas/Documents/Uni/Masterarbeit/Data/plots/Features_Pairgrid_from_og_contours.png")
plt.show()




