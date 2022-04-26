#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 18:42:19 2022

@author: lunas
"""

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
print(My_Data[0])

print(My_Data[0][4])

#print(np.array(My_Data[0][2]))

#%%
contours1 = []
for i in My_Data:
    contours1.append(np.array(i[4]))
print(contours1[0])    

#%%
cry = 0
fun = 0
feature_df = []
for i in contours1:
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
df_contour_features.to_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/pd_contourfeatures_from_interpolate_contours_from_labeled_all_cells.csv")
#%%
print(df_contour_features.head())
#%%
print(My_Data[0])

print(contours1[0]) 



#%%
#pixel
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
    print(count)
#%%
#pixel using hed channel
#rgb2hed
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
    print(count)
    
    
    
    
    
#%%
pd_pixelfeatures = pd.DataFrame.from_records(Pixel_features)
print(pd_pixelfeatures.head())
#%%
pd_pixelfeatures.to_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/pd_pixelfeatures_from_interpolate_contours_from_labeled_all_cells.csv")

#%%
#for hed channel
pd_pixelfeatures_h_channel = pd.DataFrame.from_records(Pixel_features_h_channel)
pd_pixelfeatures_e_channel = pd.DataFrame.from_records(Pixel_features_e_channel)
pd_pixelfeatures = pd.concat([pd_pixelfeatures_h_channel, pd_pixelfeatures_e_channel], axis=1)
print(pd_pixelfeatures_h_channel.shape, pd_pixelfeatures_e_channel.shape)
print(pd_pixelfeatures.head())
#%%
pd_pixelfeatures.to_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/pd_pixelfeatures_from_HED_channel_from_interpolate_contours_from_labeled_all_cells.csv")




#print(cnt)

#print(Load_Procrustes[0])
#%%
print(My_Data[0][4][0][0])
#PD_Flattend_shapes = pd.DataFrame(list(map(np.ravel, Flattend_shapes)))
#print(PD_Flattend_shapes)
#%%
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

print(PD_Flattend_shapes.head())
#%%
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0).fit(PD_Flattend_shapes)
print(kmeans.labels_)
#%%
from sklearn.manifold import TSNE
import time
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(PD_Flattend_shapes)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
#%%
PD_Flattend_shapes['tsne-2d-one'] = tsne_results[:,0]
PD_Flattend_shapes['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    #hue="y",
    palette=sns.color_palette("hls", 10),
    data=PD_Flattend_shapes,
    legend="full",
    alpha=0.3
)
#%%
import cv2
from  matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from  matplotlib import pyplot as plt
#%%

#Load_PD_Flattend_shapes = pd.read_csv('/home/lunas/Documents/Uni/Masterarbeit/Data/PD_Flattend_shapes.csv', index_col=0) 
#%%
x = StandardScaler().fit_transform(PD_Flattend_shapes)
#%%
pca = PCA(n_components=100)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents)


#%%
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))
#%%
principalDf.to_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/100_principal_components_from_interpolation_contours_from_labeled_all_cells.csv")
#%%
print(pca.explained_variance_ratio_)
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(range(0,100), pca.explained_variance_ratio_)
plt.ylabel('Explained Variance')
plt.xlabel('Principal Components')
plt.title('Explained Variance Ratio')
plt.show()


#%%
import cv2
from  matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#%%



lables = pd.read_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/data_only_lables_labled_1.csv", header = None)#, index_col=0)  
pixel = pd.read_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/pd_pixelfeatures_from_interpolate_contours_from_labeled.csv")
contour = pd.read_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/pd_contourfeatures_from_interpolate_contours_from_labeled.csv")
pca = pd.read_csv('/home/lunas/Documents/Uni/Masterarbeit/Data/100_principal_components_from_interpolation_contours_from_labeled.csv')




#%%

pca = pca.iloc[: , 1:]
pixel = pixel.iloc[: , 1:]
contour = contour.iloc[: , 1:]

#%%
print(lables.head(5))
#%%
print(pixel.head(5))
#%%
print(contour.head(5))
#%%
print(pca.head(5))
#%%
print(len(lables), len(pixel), len(contour), len(pca))
#%%

X = pd.concat([lables, contour, pixel, pca], axis=1, ignore_index=True)
print(X.shape)

#%%


colnames =  ["image", "cell", "class", "area", "perimeter", "radius", "aspect_ratio", "extent", "solidity", "equi_diameter","eccentricity","mean_cuvature","std_cuvature","roughness", "mean_val", "min_val", "max_val"]
count = 0
for i in pca.columns:
    count = count + 1
    name = "PC_" + str(count)
    colnames.append(name)
print(colnames)
#%%
X.columns = colnames
#%%
X.to_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/Feature_gathering_from_interpolate_contours_from_labeled.csv", index=False)




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
features = pd.read_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/Feature_gathering_from_interpolate_contours_from_labeled.csv")
#%%
features.drop('roughness', inplace=True, axis=1)
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
g = sns.PairGrid(features.iloc[:, 3:19], diag_sharey=False) # call seaborn's PairGrid function
g.map_lower(plt.scatter(), edgecolors="w")     # show scatter plot matrix on lower triangle
g.map_diag(sns.distplot)                    # show distributions on main diagonal
g.map_upper(correl)                         # show correlation matrix on lower triangle (using function above)
#plt.savefig("/home/lunas/Documents/Uni/Masterarbeit/Data/plots/Features_Pairgrid_from_interpolate_contours_from_labeled.png")
plt.show()

#%%
plt.scatter(features['PC_1'], features['min_val'], c=features['class_as_factor'])
#plt.figure(dpi=1200)
#pd.plotting.scatter_matrix(features.iloc[:, 3:18], alpha=0.2, c=features["class_as_factor"], figsize=[22, 20])
#plt.savefig("/home/lunas/Documents/Uni/Masterarbeit/Data/plots/Features_scatter_matrix_from_interpolate_contours_from_labeled_colored.png")
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

from sklearn import datasets


X = features.iloc[:, 3:19]
y = features["class_as_factor"]
df = X

pd.plotting.scatter_matrix(df, c=y, figsize = [8,8],
                      s=80, marker = 'D');
#df['y'] = y

sns.pairplot(df)#,hue='y')

#%%
counts = features['class'].value_counts().to_dict()
for i in counts.items():
    print(i)
farbe = features['class'].factorize()[0]
print(features.shape)
#print(farbe.shape, features.shape)
features["class_as_factor"] = farbe
#%%
for (columnName, columnData) in features.iloc[:, 3:25].iteritems():
   print('Colunm Name : ', columnName)
   print('pearson : ', abs(features[columnName].corr(features["class_as_factor"])))
   print('spearman : ', abs(features[columnName].corr(features["class_as_factor"], method = "spearman")))
#%%
features["class_tumor"] = features["class"]
print(features["class_tumor"])  

mapping = {'Tumor': 1,
           'Stroma':0, 
           'normal small lymphocyte':0,
           'large leucocyte':0,
           'Unknown':0,
           'apoptotic bodies':0,
           'Epithelial cell':0, 
           'Plasma cell':0,
           'Eosinophil':0,
           'Erythrocyte':0,
           'Vessel':0,
           'Immune cells':0,
           'Mitose':0,
           'macrophage':0,
           'Positive':0,}

#features.replace({'Tumor': mapping})
#print(features["class_tumor"]) 

features.class_tumor = [mapping[item] for item in features.class_tumor]
#print(features["class_tumor"])
print(features.groupby('class_tumor').count())
for (columnName, columnData) in features.iloc[:, 3:23].iteritems():
   print('Colunm Name : ', columnName)
   print('pearson : ', abs(features[columnName].corr(features["class_tumor"])))
   print('spearman : ', abs(features[columnName].corr(features["class_tumor"], method = "spearman")))
#%%
import statsmodels.api as sm


import numpy as np
from numpy.random import uniform, normal, poisson, binomial
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def logistic(x):
    return 1 / (1 + np.exp(-x))

#%%
#glm logistic
exog, endog = sm.add_constant(features.iloc[:, 3:20]), features["class_tumor"]
mod = sm.GLM(endog, exog,
             family=sm.families.Binomial(link=sm.families.links.logit()))
res = mod.fit()
display(res.summary())
#%%

#%%
#use k-means for 15 clusters (n clusters = n labels)
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.cluster import v_measure_score

X = features.iloc[:, 3:20] 

for i in range(2, 15):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
#print(len(kmeans.labels_))
    print("number of clusters:", i)
    print("v-score:", v_measure_score(kmeans.labels_, farbe))

#%%
#same for bivariate with tumor
X = features.iloc[:, 3:20] 

for i in range(2, 15):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
#print(len(kmeans.labels_))
    print("number of clusters:", i)
    print("v-score:", v_measure_score(kmeans.labels_, features["class_tumor"]))


#kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
#print(len(kmeans.labels_))
#print(v_measure_score(kmeans.labels_, features["class_tumor"]))
#%%
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 20)
X = features.iloc[:, 3:20] 

for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
 
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / X.shape[0])
    inertias.append(kmeanModel.inertia_)
 
    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / X.shape[0]
    mapping2[k] = kmeanModel.inertia_
for key, val in mapping1.items():
    print(f'{key} : {val}')
#%%
plt.plot(*zip(*sorted(mapping1.items())))
plt.show()
#%%
#pca
data = []
pca = PCA(n_components=3)
pca_result = pca.fit_transform(features.iloc[:, 3:20] )

#data.append([pca_result[:,0],pca_result[:,1] ,pca_result[:,2]]) 
#ata.append(pca_result[:,1])
#ata.append(pca_result[:,2])

#df = pd.DataFrame(data, columns=['pca1', 'pca2', 'pca3'])

#df['pca-two'] = pca_result[:,1] 
#df['pca-three'] = pca_result[:,2]
print(len(pca_result[:,0]))
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
print(pca.components_)
#df["class"] = features["class_tumor"]
#plt.scatter(pca_result[:,0], pca_result[:,1])

fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
scatter = ax.scatter(pca_result[:,0], pca_result[:,1], c=features["class_tumor"])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('scatter plot of forst two PCs from PCA on features')
legend1 = ax.legend(*scatter.legend_elements(num=1),
                    loc="lower right", title="Classes")
plt.show()


#plt.scatter(pca_result[:,0], pca_result[:,1], c=farbe)

#%%
#tsne
from sklearn.manifold import TSNE
import time

time_start = time.time()

tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=300)
tsne_results = tsne.fit_transform(features.iloc[:, 3:20])

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
#%%
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
scatter = ax.scatter(tsne_results[:,0], tsne_results[:,1], c=features["class_tumor"])
ax.set_xlabel('Dim1')
ax.set_ylabel('Dim2')
ax.set_title('scatter plot of first two Dims from tsne on features')
legend1 = ax.legend(*scatter.legend_elements(num=1),
                    loc="upper right", title="Classes")
plt.show()

#%%
#umap
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

# Dimension reduction and clustering libraries
import umap
#import hdbscan
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
#%%
clusterable_embedding = umap.UMAP(
    n_neighbors=100,
    min_dist=0.0,
    n_components=5,
    random_state=42,
).fit_transform(features.iloc[:, 3:20])

#%%
fig, ax = plt.subplots()

scatter = ax.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1],
            c=farbe)#, s=0.1, cmap='Spectral');
ax.set_xlabel('Dim1')
ax.set_ylabel('Dim2')
ax.set_title('scatter plot of first two Dims from umap on features')
legend1 = ax.legend(*scatter.legend_elements(num=15),
                    loc="lower right", title="Classes")

ax.add_artist(legend1)

plt.show()
#%%
nan_values = features.isna()

nan_columns = nan_values.any()


columns_with_nan = features.columns[nan_columns].tolist()

print(columns_with_nan)
print(features["roughness"].isna()==True)
#evaluate
#from sklearn.metrics.cluster import v_measure_score
#print(v_measure_score(kmeans.labels_, farbe))
#%%
is_NaN = features.isnull()

row_has_NaN = is_NaN.any(axis=1)

rows_with_NaN = features["roughness"][row_has_NaN]


print(rows_with_NaN)



