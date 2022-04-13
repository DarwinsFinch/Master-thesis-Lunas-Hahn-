#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 14:08:09 2022

@author: lunas
"""

#create dataset from og features and pca from interpolate for labeled data 

import cv2
from  matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sb 
#%%



lables = pd.read_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/data_only_lables_labled_1.csv", header = None)#, index_col=0)  
pixel = pd.read_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/pd_pixelfeatures_from_og_contours_labled.csv")
contour = pd.read_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/pd_contourfeatures_from_og_contours_labled.csv")
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
nan_values = pixel.isna()

nan_columns = nan_values.any()


columns_with_nan = pixel.columns[nan_columns].tolist()

print(columns_with_nan)
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
print(X.iloc[:, 0:3])
#%%
nan_values = X.isna()

nan_columns = nan_values.any()


columns_with_nan = X.columns[nan_columns].tolist()

print(columns_with_nan)
#%%
for i in range(3,16):
    print(i)
    count_nan = X.iloc[:, i:i+1].isna().sum()
    print ('Count of NaN: ' + str(count_nan))
#%%
X.to_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/Feature_gathering_from_og_contours_with_interpolate_pca_from_labeled.csv", index=False)




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
features = pd.read_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/Feature_gathering_from_og_contours_with_interpolate_pca_from_labeled.csv")

#%%
#reduce only to cells I have pictures of
features["keep"] = features["class"]
for filename in os.listdir("/media/lunas/Samsung USB/data/Single_cell_images/"):
#for filename in os.listdir("D:/26.01.22/Data/Images_Bozek/contours/"):
    contours = filename
    size = len(contours)
    cell_number = contours.split("_")[0]
    image = contours.lstrip("1234567890_") + ".json"
    
#print(type(image))
#print(type(cell_number))
#print(type(features.iloc[1,0]))
#print(type(features.iloc[1,1]))
    features.loc[(features["image"] == str(image)) & (features["cell"] == np.int64(cell_number)), "keep"] = 0
#%%
#print(len(features))
print(features["keep"].value_counts())

features = features.loc[features['keep'] == 0]

#print(len(features))
#features = features[features["keep"] == 0]
print(len(features))
#%%
import os
import json
list_of_bad_cells  = []

for i in range(0, len(features)):
    imagename_json = features["image"].iloc[i]
    cell_number = int(features["cell"].iloc[i])
    cell_class = str(features["class"].iloc[i])
    #print(cell_number)
    size = len(imagename_json)
    imagename = imagename_json[:size-5]
    #print(imagename)
    image= cv2.imread( "/home/lunas/Documents/Uni/Masterarbeit/Data/Images_Bozek/images/" + imagename)
    #print(outlier["pc1"].iloc[1])
    
    f = open("/home/lunas/Documents/Uni/Masterarbeit/Data/Images_Bozek/contours/" + imagename_json)
    data = json.load(f)
    try:    
        #print(data[cell_number]["geometry"]['coordinates'])
        x = data[cell_number]["geometry"]['coordinates'][0]
        if type(x[0][0]) == list:
            df = pd.DataFrame.from_records(x[0])
        else:  
            df = pd.DataFrame.from_records(x)
        
        x1 = int(df.agg([min, max])[0]["min"])
        x2 = int(df.agg([min, max])[0]["max"])
        y1 = int(df.agg([min, max])[1]["min"])
        y2 = int(df.agg([min, max])[1]["max"])
        #print(x1, x2, y1, y2)
        
        
        
        crop_img = image[y1:y2, x1:x2]
        #plt.imshow(crop_img)
        if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
            bad_cell = [imagename, cell_number]
            list_of_bad_cells.append(bad_cell)
            features = features.drop(labels = i , axis = 0)
            pass
        else:
            pass
        #image = cv2.rectangle(image, (x1,y2), (x2,y1), color = (255, 0, 0), thickness = 1)
        
            #cv2.imwrite('/media/lunas/Samsung USB/data/Single_cell_images/' + str(cell_number) + "_" + imagename, crop_img)	
    except IndexError:
        print(imagename, cell_number)
        #features = features.drop(labels = i , axis = 0)
        pass
    

#%%
#drop all celltypes that are not common in dataset
print(len(features))

features.drop(features[features["class"] == "Mitose"].index, inplace=True)
print(len(features))
features.drop(features[features["class"] == "Immune cells"].index, inplace=True)
print(len(features))
features.drop(features[features["class"] == "macrophage"].index, inplace=True)
print(len(features))
features.drop(features[features["class"] == "Vessel"].index, inplace=True)
print(len(features))
features.drop(features[features["class"] == "Positive"].index, inplace=True)
print(len(features))
features.drop(features[features["class"] == "Eosinophil"].index, inplace=True)
print(len(features))
features.drop(features[features["class"] == "Plasma cell"].index, inplace=True)
print(len(features))
features.drop(features[features["class"] == "Erythrocyte"].index, inplace=True)
print(len(features))
features.drop(features[features["class"] == "apoptotic bodies"].index, inplace=True)
print(len(features))
features.drop(features[features["class"] == "Unknown"].index, inplace=True)
print(len(features))

#%%
scaler = StandardScaler()
print(features.iloc[:, 3:17].columns)
cell_features_scaled = scaler.fit_transform(features.iloc[:, 3:17])
cell_features_scaled = np.nan_to_num(cell_features_scaled)
#%%
print(cell_features_scaled)
print(np.argwhere(np.isnan(cell_features_scaled)))
#%%
#use pca as features
print(features.iloc[:, 17:23].columns)
six_pcas_as_features = features.iloc[:, 17:23]

#%%
plt.figure(figsize=(12,10))
cor = pd.DataFrame(cell_features_scaled, columns=["area", "perimeter", "radius", "aspect_ratio", "extent", "solidity", "equi_diameter","eccentricity","mean_cuvature","std_cuvature","roughness", "mean_val", "min_val", "max_val"]).corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

#%%
#features.drop('mean_val', inplace=True, axis=1)
#features.drop('min_val', inplace=True, axis=1)
#features.drop('max_val', inplace=True, axis=1)
#cell_features_scaled = scaler.fit_transform(features.iloc[:, 3:14])
#cell_features_scaled = np.nan_to_num(cell_features_scaled)
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
g = sns.PairGrid(six_pcas_as_features , diag_sharey=False) # call seaborn's PairGrid function
g.map_lower(plt.scatter(), edgecolors="w")     # show scatter plot matrix on lower triangle
g.map_diag(sns.distplot)                    # show distributions on main diagonal
g.map_upper(correl)                         # show correlation matrix on lower triangle (using function above)
#plt.savefig("/home/lunas/Documents/Uni/Masterarbeit/Data/plots/Features_Pairgrid_from_interpolate_contours_from_labeled.png")
plt.show()

#%%
g = sns.PairGrid(cell_features_scaled, diag_sharey=False) # call seaborn's PairGrid function
g.map_lower(plt.scatter(), edgecolors="w")     # show scatter plot matrix on lower triangle
g.map_diag(sns.distplot)                    # show distributions on main diagonal
g.map_upper(correl)                         # show correlation matrix on lower triangle (using function above)
#plt.savefig("/home/lunas/Documents/Uni/Masterarbeit/Data/plots/Features_Pairgrid_from_interpolate_contours_from_labeled.png")
plt.show()
#%%
print(len(features.loc[features["class_tumor"] == 1]))
print(len(features.loc[features["class_nsl"] == 1]))
only_tumor_and_snl_for_labeles = features.loc[(features["class_tumor"] == 1) | (features["class_nsl"] == 1)]
only_tumor_and_snl = scaler.fit_transform(features.loc[(features["class_tumor"] == 1) | (features["class_nsl"] == 1)].iloc[:, 3:17])
only_tumor_and_snl = np.nan_to_num(only_tumor_and_snl)
print(len(only_tumor_and_snl))



only_tumor_and_snl_for_labeles["class_tumor"] = only_tumor_and_snl_for_labeles["class"]
print(only_tumor_and_snl_for_labeles["class_tumor"])  

mapping = {'Tumor': 1,
           'normal small lymphocyte':0}

#features.replace({'Tumor': mapping})
#print(features["class_tumor"]) 

only_tumor_and_snl_for_labeles.class_tumor = [mapping[item] for item in only_tumor_and_snl_for_labeles.class_tumor]
print(only_tumor_and_snl_for_labeles["class_tumor"])


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
features["class_nsl"] = features["class"]
print(features["class_nsl"])  

mapping = {'Tumor': 0,
           'Stroma':0, 
           'normal small lymphocyte':1,
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

features.class_nsl = [mapping[item] for item in features.class_nsl]
#print(features["class_tumor"])
print(features.groupby('class_nsl').count())
for (columnName, columnData) in features.iloc[:, 3:23].iteritems():
   print('Colunm Name : ', columnName)
   print('pearson : ', abs(features[columnName].corr(features["class_nsl"])))
   print('spearman : ', abs(features[columnName].corr(features["class_nsl"], method = "spearman")))
#%%
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
features["color_tumor"] = features["class"]
print(features["color_tumor"])  

mapping = {'Tumor': 'red',
           'Stroma':'green', 
           'normal small lymphocyte':'green',
           'large leucocyte':'green',
           'Unknown':'green',
           'apoptotic bodies':'green',
           'Epithelial cell':'green', 
           'Plasma cell':'green',
           'Eosinophil':'green',
           'Erythrocyte':'green',
           'Vessel':'green',
           'Immune cells':'green',
           'Mitose':'green',
           'macrophage':'green',
           'Positive':'green',}

#features.replace({'Tumor': mapping})
#print(features["class_tumor"]) 

features.color_tumor = [mapping[item] for item in features.color_tumor]
#%%
features["class_non_tumor"] = features["class"]
print(features["class_non_tumor"])  

mapping = {'Tumor': 'Tumor',
           'Stroma':'Non_Tumor', 
           'normal small lymphocyte':'Non_Tumor',
           'large leucocyte':'Non_Tumor',
           'Unknown':'Non_Tumor',
           'apoptotic bodies':'Non_Tumor',
           'Epithelial cell':'Non_Tumor', 
           'Plasma cell':'Non_Tumor',
           'Eosinophil':'Non_Tumor',
           'Erythrocyte':'Non_Tumor',
           'Vessel':'Non_Tumor',
           'Immune cells':'Non_Tumor',
           'Mitose':'Non_Tumor',
           'macrophage':'Non_Tumor',
           'Positive':'Non_Tumor',}

#features.replace({'Tumor': mapping})
#print(features["class_tumor"]) 

features.class_non_tumor = [mapping[item] for item in features.class_non_tumor]

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

#%%
#use k-means for 15 clusters (n clusters = n labels)
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.cluster import v_measure_score

#X = cell_features_scaled
X = cell_features_scaled

for i in range(2, 15):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
#print(len(kmeans.labels_))
    print("number of clusters:", i)
    print("v-score:", v_measure_score(kmeans.labels_, farbe))

#%%
#same for bivariate with tumor
#X = cell_features_scaled
X = six_pcas_as_features

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
X = cell_features_scaled

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
#"area", "perimeter", "radius", "aspect_ratio", "extent", "solidity", "equi_diameter","eccentricity","mean_cuvature","std_cuvature","roughness", "mean_val", "min_val", "max_val"
#print(cell_features_scaled[0])
#features["area_scaled"] = cell_features_scaled[:,0]
#print(cell_features_scaled[:,1])
#%%
#pca
data = []
pca = PCA(n_components=3)
#pca_result = pca.fit_transform(cell_features_scaled)
#pca_result = pca.fit_transform(np.nan_to_num(scaler.fit_transform(features[(features["class_tumor"] == 1)].iloc[:, 3:17])))
pca_result = pca.fit_transform(cell_features_scaled)
#pca_result = pca.fit_transform(cell_features_scaled[0]) #area only
#data.append([pca_result[:,0],pca_result[:,1] ,pca_result[:,2]]) 
#ata.append(pca_result[:,1])
#ata.append(pca_result[:,2])

#df = pd.DataFrame(data, columns=['pca1', 'pca2', 'pca3'])


features["1pca_from_features_tumor_vs_non_tumor"] = pca_result[:,0]
features["2pca_from_features_tumor_vs_non_tumor"] = pca_result[:,1]
#df['pca-two'] = pca_result[:,1] 
#df['pca-three'] = pca_result[:,2]
print(len(pca_result[:,0]))
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
print(pca.components_)
#df["class"] = features["class_tumor"]
#plt.scatter(pca_result[:,0], pca_result[:,1])

fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
#sns.scatterplot(data=features, x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue="area", style="class_non_tumor")
#sns.scatterplot(data=features, x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue="perimeter", style="class_non_tumor")
#sns.scatterplot(data=features, x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue="radius", style="class_non_tumor")
#sns.scatterplot(data=features, x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue="aspect_ratio", style="class_non_tumor")
#sns.scatterplot(data=features, x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue="extent", style="class_non_tumor")
#sns.scatterplot(data=features[features["class"] != "Tumor"], x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue="solidity", style="class_non_tumor")
#sns.scatterplot(data=features, x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue="equi_diameter", style="class_non_tumor")
#sns.scatterplot(data=features, x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue="eccentricity", style="class_non_tumor")
#sns.scatterplot(data=features, x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue="mean_cuvature", style="class_non_tumor")
#sns.scatterplot(data=features, x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue="std_cuvature", style="class_non_tumor")
#sns.scatterplot(data=features, x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue="roughness", style="class_non_tumor")
#sns.scatterplot(data=features, x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue="mean_val", style="class_non_tumor")
#sns.scatterplot(data=features, x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue="min_val", style="class_non_tumor")
#sns.scatterplot(data=features, x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue="max_val", style="class_non_tumor")


#tum = plt.scatter(features[features["class"] == "Tumor"]["1pca_from_features_tumor_vs_non_tumor"], features[features["class"] == "Tumor"]["2pca_from_features_tumor_vs_non_tumor"], color="red", label='Tumor', facecolors='none')
#non_tum = plt.scatter(features[features["class"] != "Tumor"]["1pca_from_features_tumor_vs_non_tumor"], features[features["class"] != "Tumor"]["2pca_from_features_tumor_vs_non_tumor"], color="green", label='Non_Tumor', facecolors='none')
#plt.legend()

#print outliers
tum = plt.scatter(features[features["class"] == "Tumor"]["1pca_from_features_tumor_vs_non_tumor"], features[features["class"] == "Tumor"]["2pca_from_features_tumor_vs_non_tumor"], color="red", label='Tumor', facecolors='none')
non_tum = plt.scatter(features[features["class"] != "Tumor"]["1pca_from_features_tumor_vs_non_tumor"], features[features["class"] != "Tumor"]["2pca_from_features_tumor_vs_non_tumor"], color="green", label='Non_Tumor', facecolors='none')
outlier = plt.scatter(features[(features["1pca_from_features_tumor_vs_non_tumor"] > 10) | (features["2pca_from_features_tumor_vs_non_tumor"] > 5) | (features["2pca_from_features_tumor_vs_non_tumor"] < -4.5)]["1pca_from_features_tumor_vs_non_tumor"], features[(features["1pca_from_features_tumor_vs_non_tumor"] > 10) | (features["2pca_from_features_tumor_vs_non_tumor"] > 5) | (features["2pca_from_features_tumor_vs_non_tumor"] < -4.5)]["2pca_from_features_tumor_vs_non_tumor"], color="blue", label='Outlier', s=70)#, facecolors='none')
plt.legend()

#scatter = ax.scatter(pca_result[:,0], pca_result[:,1], c=features["color_tumor"])
#scatter = ax.scatter(pca_result[:,0], pca_result[:,1], c=features[(features["class_tumor"] == 1)]["class_tumor"])

#plot every cell type
#'Tumor'
#           'Stroma':'green', 
#           'normal small lymphocyte':'green',
#           'large leucocyte':'green',
#           'Unknown':'green',
#           'apoptotic bodies':'green',
#           'Epithelial cell':'green', 
#           'Plasma cell':'green',
#           'Eosinophil':'green',
#           'Erythrocyte':'green',
#           'Vessel':'green',
#           'Immune cells':'green',
#           'Mitose':'green',
#           'macrophage':'green',
#           'Positive':'green',}

#tum = plt.scatter(features[features["class"] == "Tumor"]["1pca_from_features_tumor_vs_non_tumor"], features[features["class"] == "Tumor"]["2pca_from_features_tumor_vs_non_tumor"], color="red", label='Tumor', facecolors='none')
#tum = plt.scatter(features[features["class"] == "Stroma"]["1pca_from_features_tumor_vs_non_tumor"], features[features["class"] == "Stroma"]["2pca_from_features_tumor_vs_non_tumor"], color="red", label='Tumor', facecolors='none')
#tum = plt.scatter(features[features["class"] == "normal small lymphocyte"]["1pca_from_features_tumor_vs_non_tumor"], features[features["class"] == "normal small lymphocyte"]["2pca_from_features_tumor_vs_non_tumor"], color="red", label='Tumor', facecolors='none')
#tum = plt.scatter(features[features["class"] == "large leucocyte"]["1pca_from_features_tumor_vs_non_tumor"], features[features["class"] == "large leucocyte"]["2pca_from_features_tumor_vs_non_tumor"], color="red", label='Tumor', facecolors='none')
#tum = plt.scatter(features[features["class"] == "Unknown"]["1pca_from_features_tumor_vs_non_tumor"], features[features["class"] == "Unknown"]["2pca_from_features_tumor_vs_non_tumor"], color="red", label='Tumor', facecolors='none')
#tum = plt.scatter(features[features["class"] == "apoptotic bodies"]["1pca_from_features_tumor_vs_non_tumor"], features[features["class"] == "apoptotic bodies"]["2pca_from_features_tumor_vs_non_tumor"], color="red", label='Tumor', facecolors='none')
#tum = plt.scatter(features[features["class"] == "Epithelial cell"]["1pca_from_features_tumor_vs_non_tumor"], features[features["class"] == "Epithelial cell"]["2pca_from_features_tumor_vs_non_tumor"], color="red", label='Tumor', facecolors='none')
#tum = plt.scatter(features[features["class"] == "Plasma cell"]["1pca_from_features_tumor_vs_non_tumor"], features[features["class"] == "Plasma cell"]["2pca_from_features_tumor_vs_non_tumor"], color="red", label='Tumor', facecolors='none')
#tum = plt.scatter(features[features["class"] == "Eosinophil"]["1pca_from_features_tumor_vs_non_tumor"], features[features["class"] == "Eosinophil"]["2pca_from_features_tumor_vs_non_tumor"], color="red", label='Tumor', facecolors='none')
#tum = plt.scatter(features[features["class"] == "Erythrocyte"]["1pca_from_features_tumor_vs_non_tumor"], features[features["class"] == "Erythrocyte"]["2pca_from_features_tumor_vs_non_tumor"], color="red", label='Tumor', facecolors='none')
#tum = plt.scatter(features[features["class"] == "Vessel"]["1pca_from_features_tumor_vs_non_tumor"], features[features["class"] == "Vessel"]["2pca_from_features_tumor_vs_non_tumor"], color="red", label='Tumor', facecolors='none')
#tum = plt.scatter(features[features["class"] == "Immune cells"]["1pca_from_features_tumor_vs_non_tumor"], features[features["class"] == "Immune cells"]["2pca_from_features_tumor_vs_non_tumor"], color="red", label='Tumor', facecolors='none')
#tum = plt.scatter(features[features["class"] == "Mitose"]["1pca_from_features_tumor_vs_non_tumor"], features[features["class"] == "Mitose"]["2pca_from_features_tumor_vs_non_tumor"], color="red", label='Tumor', facecolors='none')
#tum = plt.scatter(features[features["class"] == "macrophage"]["1pca_from_features_tumor_vs_non_tumor"], features[features["class"] == "macrophage"]["2pca_from_features_tumor_vs_non_tumor"], color="red", label='Tumor', facecolors='none')


#tum = plt.scatter(features[features["class"] == "Tumor"]["1pca_from_features_tumor_vs_non_tumor"], features[features["class"] == "Tumor"]["2pca_from_features_tumor_vs_non_tumor"], c="red", label='Tumor', facecolors='none')
#non_tum = plt.scatter(features[features["class"] != "Tumor"]["1pca_from_features_tumor_vs_non_tumor"], features[features["class"] != "Tumor"]["2pca_from_features_tumor_vs_non_tumor"], c="green", label='Non_Tumor', facecolors='none')
#plt.legend()
plt.ylim(-6, 8)
plt.xlim(-6, 25)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
#ax.set_yscale('log')
#ax.set_xscale('log')
#ax.set_title('scatter plot of first two PCs from PCA on features')
ax.set_title('scatter plot of first two PCs from PCA on features')
#plt.savefig("/home/lunas/Documents/Uni/Masterarbeit/Data/plots/pca_from_labeled_tumor_vs_non_tumor_with_outlier.svg", format="svg")
#plt.savefig("/home/lunas/Documents/Uni/Masterarbeit/Data/plots/pca_from_labeled_tumor_vs_non_tumor_with_outlier_log_scale.svg", format="svg")
plt.show()
#plt.scatter(pca_result[:,0], pca_result[:,1], c=farbe)
#%%
outlier = features[(features["1pca_from_features_tumor_vs_non_tumor"] > 10) | (features["2pca_from_features_tumor_vs_non_tumor"] > 5) | (features["2pca_from_features_tumor_vs_non_tumor"] < -4.5)]["class"]
print(outlier)
#%%
features["tumor_class"] = features["class"].replace(to_replace=r"^(.(?<!Tumor))*?$", value='Non_Tumor',regex=True)
#%%
#2d density plot
features["1pca_from_features_tumor_vs_non_tumor"] = pca_result[:,0]
features["2pca_from_features_tumor_vs_non_tumor"] = pca_result[:,1]

fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('scatter plot of first two PCs from PCA on features')
ax.set_xlim(-5, 7)
ax.set_ylim(-5, 6)
#sns.kdeplot(x=pca_result[:,0], y=pca_result[:,1], hue=features["class_tumor"],c=features["color_tumor"], fill=True, label=["tumor", "non-tumor"])
sns.kdeplot(x=features[features["class"] == "Tumor"]["1pca_from_features_tumor_vs_non_tumor"], y=features[features["class"] == "Tumor"]["2pca_from_features_tumor_vs_non_tumor"], fill=True, alpha=.3, cbar = True, color="red", label='Tumor', cbar_kws= {'label': 'Tumor','ticks': [0, 100]}).set(title='KDE_of_PCA_for_tumor')
sns.kdeplot(x=features[features["class"] != "Tumor"]["1pca_from_features_tumor_vs_non_tumor"], y=features[features["class"] != "Tumor"]["2pca_from_features_tumor_vs_non_tumor"], fill=True, alpha=.3, cbar = True, color="green", label='Non_Tumor', cbar_kws= {'label': 'Non_Tumor','ticks': [0, 100]}).set(title='KDE_of_PCA_for_non_tumor')
 
#plt.legend()
plt.show()

#%%
print(pca_result.shape)
pca_pd =  pd.DataFrame(data=pca_result, columns=["pc1", "pc2", "pc3"])

pca_pd["image"] = features["image"]
pca_pd["cell"] = features["cell"]
pca_pd["class_tumor"] = features["class_tumor"]
pca_pd["class_as_factor"] = features["class_as_factor"]
pca_pd["class"] = features["class"]
print(pca_pd.head())

print(len(pca_pd))
print(len(pca_pd[(pca_pd["pc1"] <= 12) | (pca_pd["pc2"] <= -4.5)]))
#%%
print(pca_pd[(pca_pd["pc1"] >= 12) | (pca_pd["pc2"] <= -4.5 ) | (pca_pd["pc2"] >= 6 )].head().to_string())
outlier = pca_pd[(pca_pd["pc1"] >= 12) | (pca_pd["pc2"] <= -4.5) | (pca_pd["pc2"] >= 6 )]
print(len(outlier))
#%%
for i in range(0, len(outlier)):
    print(i)
    imagename_json = outlier["image"].iloc[i]
    cell_number = outlier["cell"].iloc[i]
    features.loc[(features["image"] ==  imagename_json) & (features["cell"] == cell_number), "class_tumor"] = 2

#%%
for i in range(0, len(outlier)):
    print(outlier["image"].iloc[i], outlier["class"].iloc[i], outlier["pc1"].iloc[i], outlier["pc2"].iloc[i])
#%%
#from tabulate import tabulate
import pandas as pd
print(pca_pd.iloc[100])
import os
import json
#print(outlier["image"].iloc[0])
imagename_json = pca_pd["image"].iloc[100]
cell_number = int(pca_pd["cell"].iloc[100])
print(cell_number)
size = len(imagename_json)
imagename = imagename_json[:size-5]
print(imagename)
image= cv2.imread( "/home/lunas/Documents/Uni/Masterarbeit/Data/Images_Bozek/images/" + imagename)
print(outlier["pc1"].iloc[1])

f = open("/home/lunas/Documents/Uni/Masterarbeit/Data/Images_Bozek/contours/" + imagename_json)
data = json.load(f)
x = data[cell_number]["geometry"]['coordinates'][0]
#print(x)

df = pd.DataFrame.from_records(x)
#x1 = int(df.agg([min, max])[0]["min"])
#x2 = int(df.agg([min, max])[0]["max"])
#y1 = int(df.agg([min, max])[1]["min"])
#y2 = int(df.agg([min, max])[1]["max"])
#print(x1, x2, y1, y2)


x1 = int(df.agg([min, max])[0]["min"] )
x2 = int(df.agg([min, max])[0]["max"] )
y1 = int(df.agg([min, max])[1]["min"] )
y2 = int(df.agg([min, max])[1]["max"] )
print(x1, x2, y1, y2)



crop_img = image[y1:y2, x1:x2]
plt.imshow(crop_img)

image = cv2.rectangle(image, (x1,y2), (x2,y1), color = (255, 0, 0), thickness = 1)
#print(tabulate(pca_pd, headers='keys', tablefmt='psql'))
#%%
#get picture 
import os
import json
#print(outlier["image"].iloc[0])
imagename_json = outlier["image"].iloc[2]
cell_number = int(outlier["cell"].iloc[2])
print(cell_number)
size = len(imagename_json)
imagename = imagename_json[:size-5]
print(imagename)
image= cv2.imread( "/home/lunas/Documents/Uni/Masterarbeit/Data/Images_Bozek/images/" + imagename)
print(outlier["pc1"].iloc[1])

f = open("/home/lunas/Documents/Uni/Masterarbeit/Data/Images_Bozek/contours/" + imagename_json)
data = json.load(f)
x = data[cell_number]["geometry"]['coordinates'][0]
#print(x)

df = pd.DataFrame.from_records(x)
#x1 = int(df.agg([min, max])[0]["min"])
#x2 = int(df.agg([min, max])[0]["max"])
#y1 = int(df.agg([min, max])[1]["min"])
#y2 = int(df.agg([min, max])[1]["max"])
#print(x1, x2, y1, y2)


x1 = int(df.agg([min, max])[0]["min"] )
x2 = int(df.agg([min, max])[0]["max"] )
y1 = int(df.agg([min, max])[1]["min"] )
y2 = int(df.agg([min, max])[1]["max"] )
print(x1, x2, y1, y2)



crop_img = image[y1:y2, x1:x2]
plt.imshow(crop_img)

image = cv2.rectangle(image, (x1,y2), (x2,y1), color = (255, 0, 0), thickness = 1)
#plt.imshow(image)
#plt.imshow(crop_img)
#%%
plt.imshow(image)
#%%
#%%
plt.imshow(crop_img)

#%%
import os
import json

for i in range(0, len(outlier)):
    imagename_json = outlier["image"].iloc[i]
    cell_number = int(outlier["cell"].iloc[i])
    cell_class = str(outlier["class"].iloc[i])
    print(cell_number)
    size = len(imagename_json)
    imagename = imagename_json[:size-5]
    print(imagename)
    image= cv2.imread( "/home/lunas/Documents/Uni/Masterarbeit/Data/Images_Bozek/images/" + imagename)
    print(outlier["pc1"].iloc[1])
    
    f = open("/home/lunas/Documents/Uni/Masterarbeit/Data/Images_Bozek/contours/" + imagename_json)
    data = json.load(f)
    x = data[cell_number]["geometry"]['coordinates'][0]
    
    df = pd.DataFrame.from_records(x)
    
    x1 = int(df.agg([min, max])[0]["min"])
    x2 = int(df.agg([min, max])[0]["max"])
    y1 = int(df.agg([min, max])[1]["min"])
    y2 = int(df.agg([min, max])[1]["max"])
    print(x1, x2, y1, y2)
    
    
    
    crop_img = image[y1:y2, x1:x2]
    plt.imshow(crop_img)
    
    image = cv2.rectangle(image, (x1,y2), (x2,y1), color = (255, 0, 0), thickness = 1)
    
    cv2.imwrite('/home/lunas/Documents/Uni/Masterarbeit/Data/plots/cell_images_for_pca_from_labeled/' + str(i) + "_" + imagename + ".jpg", image)	
    cv2.imwrite('/home/lunas/Documents/Uni/Masterarbeit/Data/plots/cell_images_for_pca_from_labeled/' + str(i) + "_" + cell_class + ".jpg", crop_img) 
#%%


#%%
#%%
#tsne
from sklearn.manifold import TSNE
import time

time_start = time.time()

tsne = TSNE(n_components=2, verbose=1, perplexity=100, n_iter=300)
#tsne_results = tsne.fit_transform(cell_features_scaled)
tsne_results = tsne.fit_transform(cell_features_scaled)
#tsne_results = tsne.fit_transform(np.nan_to_num(features.iloc[:, 3:17]))
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
#%%
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
#scatter = ax.scatter(tsne_results[:,0], tsne_results[:,1], c=features["class_tumor"])
features["1tsne_dim_from_features_tumor_vs_non_tumor"] = tsne_results[:,0]
features["2tsne_dim_from_features_tumor_vs_non_tumor"] = tsne_results[:,1]



#color for each feature
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
#sns.scatterplot(data=features, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue="area", style="class_non_tumor")
#sns.scatterplot(data=features, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue="perimeter", style="class_non_tumor")
#sns.scatterplot(data=features, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue="radius", style="class_non_tumor")
#sns.scatterplot(data=features, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue="aspect_ratio", style="class_non_tumor")
#sns.scatterplot(data=features, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue="extent", style="class_non_tumor")
#sns.scatterplot(data=features, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue="solidity", style="class_non_tumor")
#sns.scatterplot(data=features, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue="equi_diameter", style="class_non_tumor")
#sns.scatterplot(data=features, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue="eccentricity", style="class_non_tumor")
#sns.scatterplot(data=features, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue="mean_cuvature", style="class_non_tumor")
#sns.scatterplot(data=features, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue="std_cuvature", style="class_non_tumor")
#sns.scatterplot(data=features, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue="roughness", style="class_non_tumor")
#sns.scatterplot(data=features, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue="mean_val", style="class_non_tumor")
#sns.scatterplot(data=features, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue="min_val", style="class_non_tumor")
#sns.scatterplot(data=features, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue="max_val", style="class_non_tumor")



#tsne from all classes
#tum = plt.scatter(features[features["class"] == "Tumor"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "Tumor"]["2pca_from_features_tumor_vs_non_tumor"], color="red", label='Tumor', facecolors='none')
#tum = plt.scatter(features[features["class"] == "Stroma"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "Stroma"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="red", label='Stroma', facecolors='none')
#tum = plt.scatter(features[features["class"] == "normal small lymphocyte"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "normal small lymphocyte"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="red", label='normal small lymphocyte', facecolors='none')
#tum = plt.scatter(features[features["class"] == "large leucocyte"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "large leucocyte"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="red", label='large leucocyte', facecolors='none')
#tum = plt.scatter(features[features["class"] == "Unknown"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "Unknown"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="red", label='Unknown', facecolors='none')
#tum = plt.scatter(features[features["class"] == "apoptotic bodies"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "apoptotic bodies"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="red", label='apoptotic bodies', facecolors='none')
#tum = plt.scatter(features[features["class"] == "Epithelial cell"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "Epithelial cell"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="red", label='Epithelial cell', facecolors='none')
#tum = plt.scatter(features[features["class"] == "Plasma cell"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "Plasma cell"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="red", label='Plasma cell', facecolors='none')
#tum = plt.scatter(features[features["class"] == "Eosinophil"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "Eosinophil"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="red", label='Eosinophil', facecolors='none')
#tum = plt.scatter(features[features["class"] == "Erythrocyte"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "Erythrocyte"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="red", label='Erythrocyte', facecolors='none')
#tum = plt.scatter(features[features["class"] == "Vessel"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "Vessel"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="red", label='Vessel', facecolors='none')
#tum = plt.scatter(features[features["class"] == "Immune cells"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "Immune cells"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="red", label='Immune cell', facecolors='none')
#tum = plt.scatter(features[features["class"] == "Mitose"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "Mitose"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="red", label='Mitose', facecolors='none')
#tum = plt.scatter(features[features["class"] == "macrophage"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "macrophage"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="red", label='Macrophage', facecolors='none')




tum = plt.scatter(features[features["class"] == "Tumor"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "Tumor"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="red", label='Tumor', facecolors='none')
non_tum = plt.scatter(features[features["class"] != "Tumor"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["class"] != "Tumor"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="green", label='Non_Tumor', facecolors='none')
outlier = plt.scatter(features[(features["1pca_from_features_tumor_vs_non_tumor"] > 10) | (features["2pca_from_features_tumor_vs_non_tumor"] > 5) | (features["2pca_from_features_tumor_vs_non_tumor"] < -4.5)]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[(features["1pca_from_features_tumor_vs_non_tumor"] > 10) | (features["2pca_from_features_tumor_vs_non_tumor"] > 5) | (features["2pca_from_features_tumor_vs_non_tumor"] < -4.5)]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="blue", label='Outlier', s=70)#, facecolors='none')
plt.legend()

#scatter = ax.scatter(tsne_results[:,0], tsne_results[:,1], c=only_tumor_and_snl_for_labeles["class_tumor"])
ax.set_xlabel('Dim1')
ax.set_ylabel('Dim2')
#ax.set_yscale('log')
#ax.set_xscale('log')
ax.set_title('scatter plot of first two Dims from tsne on features')
#legend1 = ax.legend(*scatter.legend_elements(num=2),
#                    loc="upper right", title="Classes")
#plt.legend(handles=scatter.legend_elements()[0], labels=["tumor", "non-tumor", "outlier"], loc="lower right")
#plt.legend(handles=scatter.legend_elements()[0], labels=["nsl", "non-nsl", "outlier"], loc="lower right")
#plt.ylim(-9, 9)
#plt.xlim(-20, 12)

#plt.savefig("/home/lunas/Documents/Uni/Masterarbeit/Data/plots/t-sne_from_labeled_tumor_vs_non_tumor_with_outlier.svg", format="svg")
plt.show()








#%%
#kde plot
#2d density plot
features["1tsne_dim_from_features_tumor_vs_non_tumor"] = tsne_results[:,0]
features["2tsne_dim_from_features_tumor_vs_non_tumor"] = tsne_results[:,1]
#sns.kdeplot(x=pca_result[:,0], y=pca_result[:,1], hue=features["class_tumor"], fill=True, label=["tumor", "non-tumor"])
#sns.kdeplot(x=features[features["class"] == "Tumor"]["1tsne_dim_from_features_tumor_vs_non_tumor"], y=features[features["class"] == "Tumor"]["2tsne_dim_from_features_tumor_vs_non_tumor"], hue=features["tumor_class"], fill=True, cbar = True).set(title='KDE_of_PCA_for_tumor')
#sns.kdeplot(x=features[features["class"] != "Tumor"]["1pca_from_features_tumor_vs_non_tumor"], y=features[features["class"] != "Tumor"]["2pca_from_features_tumor_vs_non_tumor"], hue=features["tumor_class"], fill=True, cbar = True).set(title='KDE_of_PCA_for_non_tumor')


fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('scatter plot of first two PCs from PCA on features')
#ax.set_xlim(-5, 7)
#ax.set_ylim(-5, 6)
#sns.kdeplot(x=pca_result[:,0], y=pca_result[:,1], hue=features["class_tumor"],c=features["color_tumor"], fill=True, label=["tumor", "non-tumor"])
sns.kdeplot(x=features[features["class"] == "Tumor"]["1tsne_dim_from_features_tumor_vs_non_tumor"], y=features[features["class"] == "Tumor"]["2tsne_dim_from_features_tumor_vs_non_tumor"], fill=True, alpha=.3, cbar = True, color="red", label='Tumor', cbar_kws= {'label': 'Tumor','ticks': [0, 100]}).set(title='KDE_of_Tsne_for_tumor')
sns.kdeplot(x=features[features["class"] != "Tumor"]["1tsne_dim_from_features_tumor_vs_non_tumor"], y=features[features["class"] != "Tumor"]["2tsne_dim_from_features_tumor_vs_non_tumor"], fill=True, alpha=.3, cbar = True, color="green", label='Non_Tumor', cbar_kws= {'label': 'Non_Tumor','ticks': [0, 100]}).set(title='KDE_of_Tsne_for_non_tumor')
 
#plt.legend()
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
).fit_transform(cell_features_scaled)

#input
#only_tumor_and_snl
#np.nan_to_num(features.iloc[:, 3:17])
#%%
data = np.column_stack((clusterable_embedding[:,0], clusterable_embedding[:,1]))
data = data.tolist()
#data = data[0:100]
print(len(data))
#%%
count = 0
thumbnails = []

for i in features.index:
    size = len(features["image"][i])
    imagename = features["image"][i][:size-5]
    cellnumber = features["cell"][i] 
    #print(imagename)
    path = "/media/lunas/Samsung USB/data/Single_cell_images/"+  str(cellnumber) + "_" + str(imagename) 
    print(path)
    count = count + 1
    #if count > 5:
    #    break
    image = cv2.imread(path)
    print(image.shape)
    #image = cv2.resize(image, (15, 15))
    #print(image.shape)
    #if image.any() == None:
    #    print(path)
    thumbnails.append(image)
plt.imshow(thumbnails[0])
#%%

#%%

#%%
#%%
#juans plotting of cell images
from matplotlib.figure import Figure
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
#fig, ax = plt.subplots()

ax.scatter(clusterable_embedding[:,0], clusterable_embedding[:,1])

for umap_embed, thumbnail in zip(data, thumbnails):
    im = OffsetImage(thumbnail, zoom=0.25)
    ab = AnnotationBbox(im, (umap_embed[0], umap_embed[1]), frameon=True, pad = 0.15)
    ax.add_artist(ab)



ax.set_xlabel('Dim1')
ax.set_ylabel('Dim2')
#ax.set_yscale('log')
#ax.set_xscale('log')
ax.set_title('scatter plot of first two Dims from umap on features')
plt.show()


#%%
features["1umap_dim_from_features_tumor_vs_non_tumor"] = clusterable_embedding[:,0]
features["2umap_dim_from_features_tumor_vs_non_tumor"] = clusterable_embedding[:,1]


fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
#scatter = ax.scatter(tsne_results[:,0], tsne_results[:,1], c=features["class_tumor"])

#farbverlauf
#sns.scatterplot(data=features, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue="area", style="class_non_tumor")
#sns.scatterplot(data=features, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue="perimeter", style="class_non_tumor")
#sns.scatterplot(data=features, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue="radius", style="class_non_tumor")
#sns.scatterplot(data=features, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue="aspect_ratio", style="class_non_tumor")
#sns.scatterplot(data=features, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue="extent", style="class_non_tumor")
#sns.scatterplot(data=features, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue="solidity", style="class_non_tumor")
#sns.scatterplot(data=features, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue="equi_diameter", style="class_non_tumor")
#sns.scatterplot(data=features, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue="eccentricity", style="class_non_tumor")
#sns.scatterplot(data=features, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue="mean_cuvature", style="class_non_tumor")
#sns.scatterplot(data=features, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue="std_cuvature", style="class_non_tumor")
#sns.scatterplot(data=features, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue="roughness", style="class_non_tumor")
#sns.scatterplot(data=features, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue="mean_val", style="class_non_tumor")
#sns.scatterplot(data=features, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue="min_val", style="class_non_tumor")
#sns.scatterplot(data=features, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue="max_val", style="class_non_tumor")



#umap from all classes
#tum = plt.scatter(features[features["class"] == "Tumor"]["1umap_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "Tumor"]["2umap_dim_from_features_tumor_vs_non_tumor"], color="red", label='Tumor', facecolors='none')
#tum = plt.scatter(features[features["class"] == "Stroma"]["1umap_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "Stroma"]["2umap_dim_from_features_tumor_vs_non_tumor"], color="red", label='Stroma', facecolors='none')
#tum = plt.scatter(features[features["class"] == "normal small lymphocyte"]["1umap_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "normal small lymphocyte"]["2umap_dim_from_features_tumor_vs_non_tumor"], color="red", label='normal small lymphocyte', facecolors='none')
#tum = plt.scatter(features[features["class"] == "large leucocyte"]["1umap_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "large leucocyte"]["2umap_dim_from_features_tumor_vs_non_tumor"], color="red", label='large leucocyte', facecolors='none')
#tum = plt.scatter(features[features["class"] == "Unknown"]["1umap_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "Unknown"]["2umap_dim_from_features_tumor_vs_non_tumor"], color="red", label='Unknown', facecolors='none')
#tum = plt.scatter(features[features["class"] == "apoptotic bodies"]["1umap_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "apoptotic bodies"]["2umap_dim_from_features_tumor_vs_non_tumor"], color="red", label='apoptotic bodies', facecolors='none')
#tum = plt.scatter(features[features["class"] == "Epithelial cell"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "Epithelial cell"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="red", label='Epithelial cell', facecolors='none')
#tum = plt.scatter(features[features["class"] == "Plasma cell"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "Plasma cell"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="red", label='Plasma cell', facecolors='none')
#tum = plt.scatter(features[features["class"] == "Eosinophil"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "Eosinophil"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="red", label='Eosinophil', facecolors='none')
#tum = plt.scatter(features[features["class"] == "Erythrocyte"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "Erythrocyte"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="red", label='Erythrocyte', facecolors='none')
#tum = plt.scatter(features[features["class"] == "Vessel"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "Vessel"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="red", label='Vessel', facecolors='none')
#tum = plt.scatter(features[features["class"] == "Immune cells"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "Immune cells"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="red", label='Immune cell', facecolors='none')
#tum = plt.scatter(features[features["class"] == "Mitose"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "Mitose"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="red", label='Mitose', facecolors='none')
#tum = plt.scatter(features[features["class"] == "macrophage"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "macrophage"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="red", label='Macrophage', facecolors='none')



#tumor
#tum = plt.scatter(features[features["class"] == "Tumor"]["1umap_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "Tumor"]["2umap_dim_from_features_tumor_vs_non_tumor"], color="red", label='Tumor', facecolors='none')
non_tum = plt.scatter(features[features["class"] != "Tumor"]["1umap_dim_from_features_tumor_vs_non_tumor"], features[features["class"] != "Tumor"]["2umap_dim_from_features_tumor_vs_non_tumor"], color="green", label='Non_Tumor', facecolors='none')
#outlier = plt.scatter(features[(features["1pca_from_features_tumor_vs_non_tumor"] > 10) | (features["2pca_from_features_tumor_vs_non_tumor"] > 5) | (features["2pca_from_features_tumor_vs_non_tumor"] < -4.5)]["1umap_dim_from_features_tumor_vs_non_tumor"], features[(features["1pca_from_features_tumor_vs_non_tumor"] > 10) | (features["2pca_from_features_tumor_vs_non_tumor"] > 5) | (features["2pca_from_features_tumor_vs_non_tumor"] < -4.5)]["2umap_dim_from_features_tumor_vs_non_tumor"], color="blue", label='Outlier', s=70)#, facecolors='none')

plt.legend()

#nsl
#tum = plt.scatter(features[features["class"] == "normal small lymphocyte"]["1umap_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "normal small lymphocyte"]["2umap_dim_from_features_tumor_vs_non_tumor"], color="red", label='normal_small_lymphocyte', facecolors='none')
#non_tum = plt.scatter(features[features["class"] != "normal small lymphocyte"]["1umap_dim_from_features_tumor_vs_non_tumor"], features[features["class"] != "normal small lymphocyte"]["2umap_dim_from_features_tumor_vs_non_tumor"], color="green", label='non_normal_small_lymphocyte', facecolors='none')
#plt.legend()
#tum = plt.scatter(features[features["class"] == ['normal small lymphocyte']["1umap_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "normal small lymphocyte"]["2umap_dim_from_features_tumor_vs_non_tumor"], color="red", label='Tumor')#, facecolors='none')
#non_tum = plt.scatter(features[features["class"] != ['normal small lymphocyte']["1umap_dim_from_features_tumor_vs_non_tumor"], features[features["class"] != "normal small lymphocyte"]["2umap_dim_from_features_tumor_vs_non_tumor"], color="green", label='Non_Tumor')#, facecolors='none')
#plt.legend()



#scatter = ax.scatter(tsne_results[:,0], tsne_results[:,1], c=only_tumor_and_snl_for_labeles["class_tumor"])
ax.set_xlabel('Dim1')
ax.set_ylabel('Dim2')
#plt.ylim(-1, 4.5)
#plt.xlim(0, 12)
#ax.set_yscale('log')
#ax.set_xscale('log')
ax.set_title('scatter plot of first two Dims from umap on features')

#%%
#kde plot
#2d density plot


fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.set_xlabel('Dim1')
ax.set_ylabel('DIm2')
ax.set_title('scatter plot of first two PCs from PCA on features')
#ax.set_xlim(-5, 7)
#ax.set_ylim(-5, 6)
#sns.kdeplot(x=pca_result[:,0], y=pca_result[:,1], hue=features["class_tumor"],c=features["color_tumor"], fill=True, label=["tumor", "non-tumor"])
#sns.kdeplot(x=features[features["class"] == "Tumor"]["1umap_dim_from_features_tumor_vs_non_tumor"], y=features[features["class"] == "Tumor"]["2umap_dim_from_features_tumor_vs_non_tumor"], fill=True, alpha=.3, cbar = True, color="red", label='Tumor', cbar_kws= {'label': 'Tumor','ticks': [0, 100]}).set(title='KDE_of_Umap_for_tumor')
sns.kdeplot(x=features[features["class"] != "Tumor"]["1umap_dim_from_features_tumor_vs_non_tumor"], y=features[features["class"] != "Tumor"]["2umap_dim_from_features_tumor_vs_non_tumor"], fill=True, alpha=.3, cbar = True, color="green", label='Non_Tumor', cbar_kws= {'label': 'Non_Tumor','ticks': [0, 100]}).set(title='KDE_of_Umap_for_non_tumor_and_tumor')
 
#plt.legend()
plt.show()
#%%
#get outliers and plot the images from tumor
upper_right = features[(features["1umap_dim_from_features_tumor_vs_non_tumor"] > 8) & (features["class"] == "Tumor")]
upper_middle = features[(features["1umap_dim_from_features_tumor_vs_non_tumor"] < 8) & (features["1umap_dim_from_features_tumor_vs_non_tumor"] > 4) & (features["2umap_dim_from_features_tumor_vs_non_tumor"] > 3) & (features["2umap_dim_from_features_tumor_vs_non_tumor"] < 4) & (features["class"] == "Tumor")]
middle = features[(features["1umap_dim_from_features_tumor_vs_non_tumor"] < 4.5) & (features["1umap_dim_from_features_tumor_vs_non_tumor"] > 3) & (features["2umap_dim_from_features_tumor_vs_non_tumor"] > 2) & (features["2umap_dim_from_features_tumor_vs_non_tumor"] < 3) & (features["class"] == "Tumor")]
print(len(upper_right), len(upper_middle), len(middle))

#%%
#upper right
columns = 4
rows = 5
fig = plt.figure()
count  = 1
for index, row in upper_right.iterrows():
    size = len(row["image"])
    #print(row["image"][:size-5], row["cell"])
    filename = str(row["cell"]) + "_" + row["image"][:size-5]
    print(filename)
#for filename in os.listdir("/home/lunas/Documents/Uni/Masterarbeit/Data/plots/cell_images_outlier_umap_tumor/tumor/"):
#    print(filename)
    Image1 = cv2.imread("/media/lunas/Samsung USB/data/Single_cell_images/" + filename)
    fig.add_subplot(rows, columns, count)
    plt.imshow(Image1)
    plt.axis('off')
    count = count + 1
plt.savefig("/media/lunas/Samsung USB/data/plots/Compare_density_clusters/tumor_upper_right.svg")
plt.show()
#%%
columns = 20
rows = 22
fig = plt.figure()
count  = 1
for index, row in upper_middle.iterrows():
    size = len(row["image"])
    #print(row["image"][:size-5], row["cell"])
    filename = str(row["cell"]) + "_" + row["image"][:size-5]
    print(filename)
#for filename in os.listdir("/home/lunas/Documents/Uni/Masterarbeit/Data/plots/cell_images_outlier_umap_tumor/tumor/"):
#    print(filename)
    Image1 = cv2.imread("/media/lunas/Samsung USB/data/Single_cell_images/" + filename)
    fig.add_subplot(rows, columns, count)
    plt.imshow(Image1)
    plt.axis('off')
    count = count + 1
 #   if count > 19:
 #       break
plt.savefig("/media/lunas/Samsung USB/data/plots/Compare_density_clusters/tumor_upper_middle.svg")
plt.show()
#%%
#middle
columns = 20
rows = 20
fig = plt.figure()
count  = 1
for index, row in middle.iterrows():
    size = len(row["image"])
    #print(row["image"][:size-5], row["cell"])
    filename = str(row["cell"]) + "_" + row["image"][:size-5]
    print(filename)
#for filename in os.listdir("/home/lunas/Documents/Uni/Masterarbeit/Data/plots/cell_images_outlier_umap_tumor/tumor/"):
#    print(filename)
    Image1 = cv2.imread("/media/lunas/Samsung USB/data/Single_cell_images/" + filename)
    fig.add_subplot(rows, columns, count)
    plt.imshow(Image1)
    plt.axis('off')
    count = count + 1
    if count > 399:
        break
plt.savefig("/media/lunas/Samsung USB/data/plots/Compare_density_clusters/tumor_middle.svg")
plt.show()
#%%
#get density plots from non tumor
upper_right = features[(features["1umap_dim_from_features_tumor_vs_non_tumor"] > 8) & (features["class"] != "Tumor")]
upper_left = features[(features["1umap_dim_from_features_tumor_vs_non_tumor"] < 3) & (features["1umap_dim_from_features_tumor_vs_non_tumor"] > 0) & (features["2umap_dim_from_features_tumor_vs_non_tumor"] > 3) & (features["2umap_dim_from_features_tumor_vs_non_tumor"] < 4) & (features["class"] != "Tumor")]
upper_middle = features[(features["1umap_dim_from_features_tumor_vs_non_tumor"] < 5.5) & (features["1umap_dim_from_features_tumor_vs_non_tumor"] > 4) & (features["2umap_dim_from_features_tumor_vs_non_tumor"] > 2.2) & (features["2umap_dim_from_features_tumor_vs_non_tumor"] < 3.2) & (features["class"] != "Tumor")]
lower_middle = features[(features["1umap_dim_from_features_tumor_vs_non_tumor"] < 6) & (features["1umap_dim_from_features_tumor_vs_non_tumor"] > 4.2) & (features["2umap_dim_from_features_tumor_vs_non_tumor"] > 0.5) & (features["2umap_dim_from_features_tumor_vs_non_tumor"] < 1.8) & (features["class"] != "Tumor")]
print(len(upper_right), len(upper_left), len(upper_middle), len(lower_middle))
#%%
#upper_right
columns = 20
rows = 20
fig = plt.figure()
count  = 1
for index, row in upper_right.iterrows():
    size = len(row["image"])
    #print(row["image"][:size-5], row["cell"])
    filename = str(row["cell"]) + "_" + row["image"][:size-5]
    print(filename)
#for filename in os.listdir("/home/lunas/Documents/Uni/Masterarbeit/Data/plots/cell_images_outlier_umap_tumor/tumor/"):
#    print(filename)
    Image1 = cv2.imread("/media/lunas/Samsung USB/data/Single_cell_images/" + filename)
    fig.add_subplot(rows, columns, count)
    plt.imshow(Image1)
    plt.axis('off')
    count = count + 1
    if count > 399:
        break
plt.savefig("/media/lunas/Samsung USB/data/plots/Compare_density_clusters/upper_right.svg")
plt.show()
#%%
#upper_left
columns = 9
rows = 9
fig = plt.figure()
count  = 1
for index, row in upper_left.iterrows():
    size = len(row["image"])
    #print(row["image"][:size-5], row["cell"])
    filename = str(row["cell"]) + "_" + row["image"][:size-5]
    print(filename)
#for filename in os.listdir("/home/lunas/Documents/Uni/Masterarbeit/Data/plots/cell_images_outlier_umap_tumor/tumor/"):
#    print(filename)
    Image1 = cv2.imread("/media/lunas/Samsung USB/data/Single_cell_images/" + filename)
    fig.add_subplot(rows, columns, count)
    plt.imshow(Image1)
    plt.axis('off')
    count = count + 1
    if count > 399:
        break
plt.savefig("/media/lunas/Samsung USB/data/plots/Compare_density_clusters/upper_left.svg")
plt.show()
#%%
#%%
#upper_middle
columns = 16
rows = 16
fig = plt.figure()
count  = 1
for index, row in upper_middle.iterrows():
    size = len(row["image"])
    #print(row["image"][:size-5], row["cell"])
    filename = str(row["cell"]) + "_" + row["image"][:size-5]
    print(filename)
#for filename in os.listdir("/home/lunas/Documents/Uni/Masterarbeit/Data/plots/cell_images_outlier_umap_tumor/tumor/"):
#    print(filename)
    Image1 = cv2.imread("/media/lunas/Samsung USB/data/Single_cell_images/" + filename)
    fig.add_subplot(rows, columns, count)
    plt.imshow(Image1)
    plt.axis('off')
    count = count + 1
    if count > 399:
        break
plt.savefig("/media/lunas/Samsung USB/data/plots/Compare_density_clusters/upper_middle.svg")
plt.show()
#%%
#%%
#lower_middle
columns = 20
rows = 20
fig = plt.figure()
count  = 1
for index, row in lower_middle.iterrows():
    size = len(row["image"])
    #print(row["image"][:size-5], row["cell"])
    filename = str(row["cell"]) + "_" + row["image"][:size-5]
    print(filename)
#for filename in os.listdir("/home/lunas/Documents/Uni/Masterarbeit/Data/plots/cell_images_outlier_umap_tumor/tumor/"):
#    print(filename)
    Image1 = cv2.imread("/media/lunas/Samsung USB/data/Single_cell_images/" + filename)
    fig.add_subplot(rows, columns, count)
    plt.imshow(Image1)
    plt.axis('off')
    count = count + 1
    if count > 399:
        break
plt.savefig("/media/lunas/Samsung USB/data/plots/Compare_density_clusters/lower_middle.svg")
plt.show()
#%%



#get cells for umap outlier
umap_pd =  pd.DataFrame(data=clusterable_embedding, columns=["dim1", "dim2", "dim3", "dim4", "dim5"])

umap_pd["image"] = features["image"]
umap_pd["cell"] = features["cell"]
umap_pd["class_tumor"] = features["class_tumor"]
umap_pd["class_as_factor"] = features["class_as_factor"]
umap_pd["class"] = features["class"]
print(umap_pd.head())

print(len(umap_pd))
print(len(umap_pd[(umap_pd["dim1"] >= 7) & (umap_pd["dim2"] >= 2)]))

#snl
outlier_from_umap = umap_pd[(umap_pd["dim1"] >= 7) & (umap_pd["dim2"] >= 2)]
rest_from_umap = umap_pd[(umap_pd["dim1"] <= 7) & (umap_pd["class"] == "normal small lymphocyte")]
print(len(rest_from_umap), len(outlier_from_umap), len(umap_pd))


#tumor
outlier_from_umap_tumor = umap_pd[(umap_pd["dim1"] >= 7) & (umap_pd["class"] == "Tumor")]
tumor_top_from_umap = umap_pd[(umap_pd["dim1"] <= 7) & (umap_pd["class"] == "Tumor") & (umap_pd["dim2"] <= 7)]
tumor_bottom_from_umap = umap_pd[(umap_pd["dim1"] <= 7) & (umap_pd["class"] == "Tumor") & (umap_pd["dim2"] <= 4)]
print(len(outlier_from_umap_tumor), len(tumor_top_from_umap), len(tumor_bottom_from_umap))

#unknown
outlier_from_umap_unknown = umap_pd[(umap_pd["dim1"] >= 7) & (umap_pd["class"] == "Unknown")]
rest_from_umap_unknown = umap_pd[(umap_pd["dim1"] <= 7) & (umap_pd["class"] == "Unknown")]
print(len(outlier_from_umap_unknown), len(rest_from_umap_unknown), len(umap_pd))
#%%
counts = umap_pd['class'].value_counts().to_dict()
for i in counts.items():
    print(i)
print("\n")
counts = outlier_from_umap['class'].value_counts().to_dict()
for i in counts.items():
    print(i)
#%%
#from tabulate import tabulate
import pandas as pd
print(umap_pd.iloc[100])
import os
import json
#print(outlier["image"].iloc[0])
imagename_json = umap_pd["image"].iloc[100]
cell_number = int(umap_pd["cell"].iloc[100])
print(cell_number)
size = len(imagename_json)
imagename = imagename_json[:size-5]
print(imagename)
image= cv2.imread( "/home/lunas/Documents/Uni/Masterarbeit/Data/Images_Bozek/images/" + imagename)
print(outlier_from_umap["dim1"].iloc[1])

f = open("/home/lunas/Documents/Uni/Masterarbeit/Data/Images_Bozek/contours/" + imagename_json)
data = json.load(f)
x = data[cell_number]["geometry"]['coordinates'][0]
#print(x)

df = pd.DataFrame.from_records(x)
#x1 = int(df.agg([min, max])[0]["min"])
#x2 = int(df.agg([min, max])[0]["max"])
#y1 = int(df.agg([min, max])[1]["min"])
#y2 = int(df.agg([min, max])[1]["max"])
#print(x1, x2, y1, y2)


x1 = int(df.agg([min, max])[0]["min"] )
x2 = int(df.agg([min, max])[0]["max"] )
y1 = int(df.agg([min, max])[1]["min"] )
y2 = int(df.agg([min, max])[1]["max"] )

x1 = x1 - 2
y1 = y1 - 2
x2 = x2 + 2
y2 = y2 + 2

print(image.shape)
print(x1, x2, y1, y2)



crop_img = image[y1:y2, x1:x2]
plt.imshow(crop_img)

image = cv2.rectangle(image, (x1,y2), (x2,y1), color = (255, 0, 0), thickness = 2)
plt.imshow(image)
#print(tabulate(pca_pd, headers='keys', tablefmt='psql'))
#%%
import os
import json

for i in range(0, len(outlier_from_umap)):
    imagename_json = outlier_from_umap["image"].iloc[i]
    cell_number = int(outlier_from_umap["cell"].iloc[i])
    cell_class = str(outlier_from_umap["class"].iloc[i])
    print(cell_number)
    size = len(imagename_json)
    imagename = imagename_json[:size-5]
    print(imagename)
    image= cv2.imread( "/home/lunas/Documents/Uni/Masterarbeit/Data/Images_Bozek/images/" + imagename)
    print(outlier_from_umap["dim1"].iloc[1])
    
    
    if cell_number == 277 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no33_tile_no61.png":
        pass
    elif cell_number == 286 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no33_tile_no48.png":
        pass
    else:
    
        f = open("/home/lunas/Documents/Uni/Masterarbeit/Data/Images_Bozek/contours/" + imagename_json)
        data = json.load(f)
        x = data[cell_number]["geometry"]['coordinates'][0]
        
        df = pd.DataFrame.from_records(x)
        
        x1 = int(df.agg([min, max])[0]["min"])
        x2 = int(df.agg([min, max])[0]["max"])
        y1 = int(df.agg([min, max])[1]["min"])
        y2 = int(df.agg([min, max])[1]["max"])
        print(x1, x2, y1, y2)
        
        if all([v-4 >= 0 for v in [x1,y1]]) and all([v+4 <= 512 for v in [x2,y2]]):
            x1 = x1 - 4 
            y1 = y1 - 4
            x2 = x2 + 4
            y2 = y2 + 4
       
        else:
            pass
        
        print(image.shape)
        crop_img = image[y1:y2, x1:x2]
        plt.imshow(crop_img)
        cv2.imwrite('/home/lunas/Documents/Uni/Masterarbeit/Data/plots/cell_images_outlier_umap_right/' + str(i) + "_" + cell_class + ".jpg", crop_img)
        
        image = cv2.rectangle(image, (x1,y2), (x2,y1), color = (255, 0, 0), thickness = 1)
        cv2.imwrite('/home/lunas/Documents/Uni/Masterarbeit/Data/plots/cell_images_outlier_umap_right/' + str(i) + "_" + imagename + ".jpg", image)	
        
#%%
image= cv2.imread( "/home/lunas/Documents/Uni/Masterarbeit/Data/Images_Bozek/images/Philipp Lohneis - C 16.1058 4C he - 2020-04-01 10.53.10.ndpi_tile_no25580.png")
f = open("/home/lunas/Documents/Uni/Masterarbeit/Data/Images_Bozek/contours/Philipp Lohneis - C 16.1058 4C he - 2020-04-01 10.53.10.ndpi_tile_no25580.png.json")
data = json.load(f)
x = data[cell_number]["geometry"]['coordinates'][0]
print(x)
#%%
#get rest from umap tumor
import os
import json

for i in range(0, len(tumor_top_from_umap)):
    imagename_json = tumor_top_from_umap["image"].iloc[i]
    cell_number = int(tumor_top_from_umap["cell"].iloc[i])
    cell_class = str(tumor_top_from_umap["class"].iloc[i])
    print(cell_number)
    size = len(imagename_json)
    imagename = imagename_json[:size-5]
    print(imagename)
    image= cv2.imread( "/home/lunas/Documents/Uni/Masterarbeit/Data/Images_Bozek/images/" + imagename)
    #print(rest_from_umap["dim1"].iloc[1])
    
    
    if cell_number == 164 and imagename == "Philipp Lohneis - C 16.1058 4C he - 2020-04-01 10.53.10.ndpi_tile_no25580.png":
        pass
    elif cell_number == 426 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no14_tile_no23.png":
        pass
    elif cell_number == 163 and imagename == "Philipp Lohneis - C 16.1058 4C he - 2020-04-01 10.53.10.ndpi_tile_no11207.png":
        pass
    elif cell_number == 85 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no0_tile_no13.png":
        pass
    elif cell_number == 196 and imagename == "Philipp Lohneis - C 16.1058 4C he - 2020-04-01 10.53.10.ndpi_tile_no25390.png":
        pass
    elif cell_number == 38 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no18_tile_no20.png":
        pass
    elif cell_number == 186 and imagename == "Philipp Lohneis - C 16.1058 4C he - 2020-04-01 10.53.10.ndpi_tile_no10398.png":
        pass
    elif cell_number == 163 and imagename == "Philipp Lohneis - C 16.1058 4C he - 2020-04-01 10.53.10.ndpi_tile_no10634.png":
        pass
    elif cell_number == 211 and imagename == "Philipp Lohneis - C 16.1058 4C he - 2020-04-01 10.53.10.ndpi_tile_no10194.png":
        pass
    elif cell_number == 85 and imagename == "Philipp Lohneis - C 16.1058 4C he - 2020-04-01 10.53.10.ndpi_tile_no3310.png":
        pass
    elif cell_number == 29 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no0_tile_no24.png":
        pass
    elif cell_number == 275 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no33_tile_no61.png":
        pass
    elif cell_number == 287 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no33_tile_no48.png":
        pass
    elif cell_number == 249 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no33_tile_no75.png":
        pass
    elif cell_number == 281 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no33_tile_no75.png":
        pass
    elif cell_number == 246 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no30_tile_no71.png":
        pass
    elif cell_number == 298 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no15_tile_no69.png":
        pass
    elif cell_number == 175 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no19_tile_no11.png":
        pass
    elif cell_number == 382 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no15_tile_no36.png":
        pass
    else:
    
        f = open("/home/lunas/Documents/Uni/Masterarbeit/Data/Images_Bozek/contours/" + imagename_json)
        data = json.load(f)
        x = data[cell_number]["geometry"]['coordinates'][0]
        
        df = pd.DataFrame.from_records(x)
        
        x1 = int(df.agg([min, max])[0]["min"])
        x2 = int(df.agg([min, max])[0]["max"])
        y1 = int(df.agg([min, max])[1]["min"])
        y2 = int(df.agg([min, max])[1]["max"])
        #print(x1, x2, y1, y2)
        
        if all([v-4 >= 0 for v in [x1,y1]]) and all([v+4 <= 512 for v in [x2,y2]]):
            x1 = x1 - 4 
            y1 = y1 - 4
            x2 = x2 + 4
            y2 = y2 + 4
       
        else:
            pass
        
        #print(image.shape)
        crop_img = image[y1:y2, x1:x2]
        plt.imshow(crop_img)
        cv2.imwrite('/home/lunas/Documents/Uni/Masterarbeit/Data/plots/cell_images_rest_umap_tumor/' + str(i) + "_" + cell_class + ".jpg", crop_img)
        
        image = cv2.rectangle(image, (x1,y2), (x2,y1), color = (255, 0, 0), thickness = 1)
        cv2.imwrite('/home/lunas/Documents/Uni/Masterarbeit/Data/plots/cell_images_rest_umap_tumor/' + str(i) + "_" + imagename + ".jpg", image)	



#%%
#get rest from umap unknown

import os
import json

for i in range(0, len(rest_from_umap_unknown)):
    imagename_json = rest_from_umap_unknown["image"].iloc[i]
    cell_number = int(rest_from_umap_unknown["cell"].iloc[i])
    cell_class = str(rest_from_umap_unknown["class"].iloc[i])
    print(cell_number)
    size = len(imagename_json)
    imagename = imagename_json[:size-5]
    print(imagename)
    image= cv2.imread( "/home/lunas/Documents/Uni/Masterarbeit/Data/Images_Bozek/images/" + imagename)
    #print(rest_from_umap["dim1"].iloc[1])
    
    
    if cell_number == 164 and imagename == "Philipp Lohneis - C 16.1058 4C he - 2020-04-01 10.53.10.ndpi_tile_no25580.png":
        pass
    elif cell_number == 426 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no14_tile_no23.png":
        pass
    elif cell_number == 163 and imagename == "Philipp Lohneis - C 16.1058 4C he - 2020-04-01 10.53.10.ndpi_tile_no11207.png":
        pass
    elif cell_number == 85 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no0_tile_no13.png":
        pass
    elif cell_number == 196 and imagename == "Philipp Lohneis - C 16.1058 4C he - 2020-04-01 10.53.10.ndpi_tile_no25390.png":
        pass
    elif cell_number == 38 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no18_tile_no20.png":
        pass
    elif cell_number == 186 and imagename == "Philipp Lohneis - C 16.1058 4C he - 2020-04-01 10.53.10.ndpi_tile_no10398.png":
        pass
    elif cell_number == 163 and imagename == "Philipp Lohneis - C 16.1058 4C he - 2020-04-01 10.53.10.ndpi_tile_no10634.png":
        pass
    elif cell_number == 211 and imagename == "Philipp Lohneis - C 16.1058 4C he - 2020-04-01 10.53.10.ndpi_tile_no10194.png":
        pass
    elif cell_number == 85 and imagename == "Philipp Lohneis - C 16.1058 4C he - 2020-04-01 10.53.10.ndpi_tile_no3310.png":
        pass
    elif cell_number == 29 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no0_tile_no24.png":
        pass
    elif cell_number == 275 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no33_tile_no61.png":
        pass
    elif cell_number == 287 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no33_tile_no48.png":
        pass
    elif cell_number == 249 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no33_tile_no75.png":
        pass
    elif cell_number == 281 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no33_tile_no75.png":
        pass
    elif cell_number == 246 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no30_tile_no71.png":
        pass
    elif cell_number == 298 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no15_tile_no69.png":
        pass
    elif cell_number == 175 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no19_tile_no11.png":
        pass
    elif cell_number == 382 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no15_tile_no36.png":
        pass
    else:
    
        f = open("/home/lunas/Documents/Uni/Masterarbeit/Data/Images_Bozek/contours/" + imagename_json)
        data = json.load(f)
        x = data[cell_number]["geometry"]['coordinates'][0]
        
        df = pd.DataFrame.from_records(x)
        
        x1 = int(df.agg([min, max])[0]["min"])
        x2 = int(df.agg([min, max])[0]["max"])
        y1 = int(df.agg([min, max])[1]["min"])
        y2 = int(df.agg([min, max])[1]["max"])
        #print(x1, x2, y1, y2)
        
        if all([v-4 >= 0 for v in [x1,y1]]) and all([v+4 <= 512 for v in [x2,y2]]):
            x1 = x1 - 4 
            y1 = y1 - 4
            x2 = x2 + 4
            y2 = y2 + 4
       
        else:
            pass
        
        #print(image.shape)
        crop_img = image[y1:y2, x1:x2]
        plt.imshow(crop_img)
        cv2.imwrite('/home/lunas/Documents/Uni/Masterarbeit/Data/plots/cell_images_rest_umap_unknown/' + str(i) + "_" + cell_class + ".jpg", crop_img)
        
        image = cv2.rectangle(image, (x1,y2), (x2,y1), color = (255, 0, 0), thickness = 1)
        cv2.imwrite('/home/lunas/Documents/Uni/Masterarbeit/Data/plots/cell_images_rest_umap_unknown/' + str(i) + "_" + imagename + ".jpg", image)	





#%%               
###
#get rest from umap

import os
import json

for i in range(0, len(rest_from_umap)):
    imagename_json = rest_from_umap["image"].iloc[i]
    cell_number = int(rest_from_umap["cell"].iloc[i])
    cell_class = str(rest_from_umap["class"].iloc[i])
    print(cell_number)
    size = len(imagename_json)
    imagename = imagename_json[:size-5]
    print(imagename)
    image= cv2.imread( "/home/lunas/Documents/Uni/Masterarbeit/Data/Images_Bozek/images/" + imagename)
    #print(rest_from_umap["dim1"].iloc[1])
    
    
    if cell_number == 164 and imagename == "Philipp Lohneis - C 16.1058 4C he - 2020-04-01 10.53.10.ndpi_tile_no25580.png":
        pass
    elif cell_number == 426 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no14_tile_no23.png":
        pass
    elif cell_number == 163 and imagename == "Philipp Lohneis - C 16.1058 4C he - 2020-04-01 10.53.10.ndpi_tile_no11207.png":
        pass
    elif cell_number == 85 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no0_tile_no13.png":
        pass
    elif cell_number == 196 and imagename == "Philipp Lohneis - C 16.1058 4C he - 2020-04-01 10.53.10.ndpi_tile_no25390.png":
        pass
    elif cell_number == 38 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no18_tile_no20.png":
        pass
    elif cell_number == 186 and imagename == "Philipp Lohneis - C 16.1058 4C he - 2020-04-01 10.53.10.ndpi_tile_no10398.png":
        pass
    elif cell_number == 163 and imagename == "Philipp Lohneis - C 16.1058 4C he - 2020-04-01 10.53.10.ndpi_tile_no10634.png":
        pass
    elif cell_number == 211 and imagename == "Philipp Lohneis - C 16.1058 4C he - 2020-04-01 10.53.10.ndpi_tile_no10194.png":
        pass
    elif cell_number == 85 and imagename == "Philipp Lohneis - C 16.1058 4C he - 2020-04-01 10.53.10.ndpi_tile_no3310.png":
        pass
    elif cell_number == 29 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no0_tile_no24.png":
        pass
    elif cell_number == 275 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no33_tile_no61.png":
        pass
    elif cell_number == 287 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no33_tile_no48.png":
        pass
    elif cell_number == 249 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no33_tile_no75.png":
        pass
    elif cell_number == 281 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no33_tile_no75.png":
        pass
    elif cell_number == 246 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no30_tile_no71.png":
        pass
    elif cell_number == 298 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no15_tile_no69.png":
        pass
    elif cell_number == 175 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no19_tile_no11.png":
        pass
    elif cell_number == 175 and imagename == "Lymph1 - 2021-01-15 13.17.54.ndpi_patch_no19_tile_no11.png":
        pass
    else:
    
        f = open("/home/lunas/Documents/Uni/Masterarbeit/Data/Images_Bozek/contours/" + imagename_json)
        data = json.load(f)
        x = data[cell_number]["geometry"]['coordinates'][0]
        
        df = pd.DataFrame.from_records(x)
        
        x1 = int(df.agg([min, max])[0]["min"])
        x2 = int(df.agg([min, max])[0]["max"])
        y1 = int(df.agg([min, max])[1]["min"])
        y2 = int(df.agg([min, max])[1]["max"])
        #print(x1, x2, y1, y2)
        
        if all([v-4 >= 0 for v in [x1,y1]]) and all([v+4 <= 512 for v in [x2,y2]]):
            x1 = x1 - 4 
            y1 = y1 - 4
            x2 = x2 + 4
            y2 = y2 + 4
       
        else:
            pass
        
        #print(image.shape)
        crop_img = image[y1:y2, x1:x2]
        plt.imshow(crop_img)
        cv2.imwrite('/home/lunas/Documents/Uni/Masterarbeit/Data/plots/cell_images_rest_nsl_umap/' + str(i) + "_" + cell_class + ".jpg", crop_img)
        
        image = cv2.rectangle(image, (x1,y2), (x2,y1), color = (255, 0, 0), thickness = 1)
        cv2.imwrite('/home/lunas/Documents/Uni/Masterarbeit/Data/plots/cell_images_rest_nsl_umap/' + str(i) + "_" + imagename + ".jpg", image)	
               


#%%
print(len(outlier_from_umap))
#%%
#get picture 
import os
import json
#print(outlier["image"].iloc[0])
imagename_json = outlier["image"].iloc[2]
cell_number = int(outlier["cell"].iloc[2])
print(cell_number)
size = len(imagename_json)
imagename = imagename_json[:size-5]
print(imagename)
image= cv2.imread( "/home/lunas/Documents/Uni/Masterarbeit/Data/Images_Bozek/images/" + imagename)
print(outlier["pc1"].iloc[1])

f = open("/home/lunas/Documents/Uni/Masterarbeit/Data/Images_Bozek/contours/" + imagename_json)
data = json.load(f)
x = data[cell_number]["geometry"]['coordinates'][0]
#print(x)

df = pd.DataFrame.from_records(x)
#x1 = int(df.agg([min, max])[0]["min"])
#x2 = int(df.agg([min, max])[0]["max"])
#y1 = int(df.agg([min, max])[1]["min"])
#y2 = int(df.agg([min, max])[1]["max"])
#print(x1, x2, y1, y2)


x1 = int(df.agg([min, max])[0]["min"] )
x2 = int(df.agg([min, max])[0]["max"] )
y1 = int(df.agg([min, max])[1]["min"] )
y2 = int(df.agg([min, max])[1]["max"] )
print(x1, x2, y1, y2)



crop_img = image[y1:y2, x1:x2]
plt.imshow(crop_img)

image = cv2.rectangle(image, (x1,y2), (x2,y1), color = (255, 0, 0), thickness = 1)
#plt.imshow(image)
#plt.imshow(crop_img)
#%%
plt.imshow(image)
#%%
#%%
plt.imshow(crop_img)

#%%

fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.set_xlabel('Dim1')
ax.set_ylabel('Dim2')
ax.set_title('scatter plot of first two Dims from Umap on features')
#ax.set_xlim(-5, 7)
#ax.set_ylim(-5, 6)
#sns.kdeplot(x=pca_result[:,0], y=pca_result[:,1], hue=features["class_tumor"],c=features["color_tumor"], fill=True, label=["tumor", "non-tumor"])
#sns.kdeplot(x=features[features["class"] == "Tumor"]["1umap_dim_from_features_tumor_vs_non_tumor"], y=features[features["class"] == "Tumor"]["2umap_dim_from_features_tumor_vs_non_tumor"], fill=True, cbar = True, color="red", label='Tumor', cbar_kws= {'label': 'Tumor','ticks': [0, 100]}).set(title='KDE_of_Umap_for_tumor')
sns.kdeplot(x=features[features["class"] != "Tumor"]["1umap_dim_from_features_tumor_vs_non_tumor"], y=features[features["class"] != "Tumor"]["2umap_dim_from_features_tumor_vs_non_tumor"], fill=True, cbar = True, color="green", label='Non_Tumor', cbar_kws= {'label': 'Non_Tumor','ticks': [0, 100]}).set(title='KDE_of_Umap_for_non_tumor')
 
#plt.legend()
plt.show()
#%%
print(clusterable_embedding.shape)
pca_pd =  pd.DataFrame(data=clusterable_embedding, columns=["pc1", "pc2", "pc3","pc4", "pc5" ])

pca_pd["image"] = only_tumor_and_snl_for_labeles["image"]
pca_pd["cell"] = only_tumor_and_snl_for_labeles["cell"]
pca_pd["class_tumor"] = only_tumor_and_snl_for_labeles["class_tumor"]
pca_pd["class_as_factor"] = only_tumor_and_snl_for_labeles["class_as_factor"]
pca_pd["class"] = only_tumor_and_snl_for_labeles["class"]
print(pca_pd.head())

#print(len(pca_pd))
print(len(pca_pd[(pca_pd["pc1"] >= 8) & (pca_pd["class_tumor"] == 0)]))
print(len(pca_pd[pca_pd["class_tumor"] == 0]))

counts = pca_pd["class_tumor"].value_counts().to_dict()
for i in counts.items():
    print(i)
#%%
print(pca_pd[(pca_pd["pc1"] >= 12.5) | (pca_pd["pc2"] <= -4.5 ) | (pca_pd["pc2"] >= 7 )].head().to_string())
outlier = pca_pd[(pca_pd["pc1"] >= 12.5) | (pca_pd["pc2"] <= -4.5) | (pca_pd["pc2"] >= 7 )]
print(len(outlier))
#%%
for i in range(0, len(outlier)):
    print(i)
    imagename_json = outlier["image"].iloc[i]
    cell_number = outlier["cell"].iloc[i]
    features.loc[(features["image"] ==  imagename_json) & (features["cell"] == cell_number), "class_tumor"] = 2


#%%
#%%
#%%
#random forest 

print(len(cell_features_scaled),len(features["class_nsl"]))
#%%
#random forest
# Using Skicit-learn to split data into training and testing sets
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
#import shap
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std

train_features, test_features, train_labels, test_labels = train_test_split(cell_features_scaled , features["class_tumor"], test_size = 0.2, random_state = 42, stratify=features["class_tumor"])
train_features, validate_features, train_labels, validate_labels = train_test_split(train_features , train_labels, test_size = 0.2, random_state = 42, stratify=train_labels)



print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
print('Validate Features Shape:', validate_features.shape)
print('Validate Labels Shape:', validate_labels.shape)




rf = RandomForestClassifier(n_estimators=100)
rf.fit(train_features, train_labels)

#%%
print(cell_features_scaled[:, :1])
#%%
sorted_idx = rf.feature_importances_.argsort()
print(rf.feature_importances_)
plt.barh(features.iloc[:, 3:17].columns[sorted_idx], rf.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")
#%%

perm_importance = permutation_importance(rf, test_features, test_labels)
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(features.iloc[:, 3:17].columns[sorted_idx], perm_importance.importances_mean[sorted_idx])

plt.xlabel("Permutation Importance")
#%%
# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(rf, train_features, train_labels, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

y_pred_test = rf.predict(test_features)
#print(y_pred_test)
print("accuracy score for train", accuracy_score(test_labels, y_pred_test))
#print(accuracy_score)
y_pred_test = rf.predict(train_features)
print("accuracy score for test",accuracy_score(train_labels, y_pred_test))
#print(accuracy_score)
#%%
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
#import shap
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std

train_features, test_features, train_labels, test_labels = train_test_split(cell_features_scaled, features["class_tumor"], test_size = 0.25, random_state = 42, stratify=features["class_tumor"])

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

for i in [2,4,8,16,32,64,128,256]:
    print(i)
    rf = RandomForestClassifier(n_estimators=i)
    rf.fit(train_features, train_labels)
    # evaluate the model
    #cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    #n_scores = cross_val_score(rf, train_features, train_labels, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    #print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

    y_pred_test = rf.predict(test_features)
    #print(y_pred_test)
    print("accuracy score for train", accuracy_score(test_labels, y_pred_test))
    #print(accuracy_score)
    y_pred_test = rf.predict(train_features)
    print("accuracy score for test",accuracy_score(train_labels, y_pred_test))
    #print(accuracy_score)
#%%
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestClassifier(random_state = 42)

# Number of trees in random forest
n_estimators = [2,4,8,16,32,64,128,256]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(1, 100, num = 10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
#%%
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(train_features, train_labels)
rf_random.fit(test_features, test_labels)
#%%
#print(rf_random.best_params_)
#no overfitting with these params
rf = RandomForestClassifier(random_state = 42, n_estimators=64, min_samples_split=10, min_samples_leaf=2, max_features="sqrt", max_depth=4, bootstrap=False)
#best params from the cv search for train
#rf = RandomForestClassifier(random_state = 42, n_estimators=64, min_samples_split=10, min_samples_leaf=2, max_features="sqrt", max_depth=56, bootstrap=False)
print(rf.get_params())
rf.fit(train_features, train_labels)


y_pred_test = rf.predict(train_features)
#print(y_pred_test)
print("accuracy score for train", accuracy_score(train_labels, y_pred_test))
#print(accuracy_score)
y_pred_test = rf.predict(test_features)
print("accuracy score for test",accuracy_score(test_labels, y_pred_test))
#print(accuracy_score)
y_pred_test = rf.predict(validate_features)
print("accuracy score for validate",accuracy_score(validate_labels, y_pred_test))
#%%
#feature importance on test,train,val
#perm_importance = permutation_importance(rf, train_features, train_labels)
perm_importance = permutation_importance(rf, test_features, test_labels)
#perm_importance = permutation_importance(rf, validate_features, validate_labels)
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(features.iloc[:, 3:17].columns[sorted_idx], perm_importance.importances_mean[sorted_idx])

plt.xlabel("Permutation Importance")
#%%
ll = []
for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13]:
    
    train_features, test_features, train_labels, test_labels = train_test_split(cell_features_scaled[:, i:i + 1] , features["class_tumor"], test_size = 0.2, random_state = 42, stratify=features["class_tumor"])
    #train_features, validate_features, train_labels, validate_labels = train_test_split(train_features , train_labels, test_size = 0.2, random_state = 42, stratify=train_labels)
    
    

    
    
    rf = RandomForestClassifier(random_state = 42, n_estimators=64, min_samples_split=10, min_samples_leaf=2, max_features="sqrt", max_depth=4, bootstrap=False)
    rf.fit(train_features, train_labels)

    print(features.iloc[:, 3 + i: 4 + i].columns)
    y_pred_test = rf.predict(train_features)
    #print(y_pred_test)
    print("accuracy score for train", accuracy_score(train_labels, y_pred_test))
    ll.append(accuracy_score(train_labels, y_pred_test))
    #print(accuracy_score)
    #y_pred_test = rf.predict(test_features)
    #print("accuracy score for test",accuracy_score(test_labels, y_pred_test))
    #print(accuracy_score)
    #y_pred_test = rf.predict(validate_features)
    #print("accuracy score for validate",accuracy_score(validate_labels, y_pred_test))
#%%
print(features.iloc[:, 3:4].columns)
fig, ax = plt.subplots()
x_pos = [0,20,40,60,80,100,120,140,160,180,200,220,240,260]#,56,60,64,68]
plt.bar(x_pos, ll, width = 4)
plt.xticks(x_pos, ["area", "perimeter", "radius", "aspect_ratio", "extent", "solidity", "equi_diameter","eccentricity","mean_cuvature","std_cuvature","roughness", "mean_val", "min_val", "max_val"], size = 10, rotation='vertical')
ax.set_xlabel('Features')
ax.set_ylabel('Score')
ax.set_title('accuracy score from the rf with only one feature')
plt.show() 
#plt.bar(["area", "perimeter", "radius", "aspect_ratio", "extent", "solidity", "equi_diameter","eccentricity","mean_cuvature","std_cuvature","roughness", "mean_val", "min_val", "max_val"], ll, width = 0.2)
#%%
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(train_features, train_labels)
base_accuracy = evaluate(base_model, test_features, test_labels) 
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, test_features, test_labels)

print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

#%%
#lasso
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0, penalty='l1',solver='liblinear').fit(train_features , train_labels)
print("mean accuracy for train", clf.score(train_features, train_labels))
print("mean accuracy for test", clf.score(test_features, test_labels))
print("mean accuracy for validate", clf.score(validate_features, validate_labels))
print(clf.coef_)
#%%
importance = clf.coef_
# summarize feature importance
for i,v in enumerate(importance[0]):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
x_pos = [0,20,40,60,80,100,120,140,160,180,200,220,240,260]
#plt.bar([x for x in range(len(importance[0]))], importance[0])
plt.bar(x_pos, importance[0], width = 4)
plt.xticks(x_pos, ["area", "perimeter", "radius", "aspect_ratio", "extent", "solidity", "equi_diameter","eccentricity","mean_cuvature","std_cuvature","roughness", "mean_val", "min_val", "max_val"], size = 4)
plt.show()
#%%
fig, ax = plt.subplots()
importance = clf.coef_
x_pos = [0,20,40,60,80,100,120,140,160,180,200,220,240,260]
barplot = plt.bar(x_pos, importance[0], width = 4)
plt.xticks(x_pos, ["area", "perimeter", "radius", "aspect_ratio", "extent", "solidity", "equi_diameter","eccentricity","mean_cuvature","std_cuvature","roughness", "mean_val", "min_val", "max_val"], size = 10, rotation='vertical')
ax.set_xlabel('Features')
ax.set_ylabel('Coefficient')
ax.set_title('Feature importance for L1 Logistic-regression_for_tumor_vs_non_tumor')

#ax.add_artist(legend1)
#plt.savefig("/home/lunas/Documents/Uni/Masterarbeit/Data/plots/umap_from_labeled_tumor_vs_non_tumor_with_outlier.svg", format="svg")
plt.show()

#%%
import matplotlib.pyplot as plt

coefs = pd.DataFrame(
   model.coef_,
   columns=['Coefficients'], index=pd.DataFrame(train_features).columns
)

coefs.plot(kind='barh', figsize=(9, 7))
plt.title('Ridge model')
plt.axvline(x=0, color='.5')
plt.subplots_adjust(left=.3)

#%%
print(model.coef_)
sorted_idx = model.coef_.argsort()

plt.barh(features.iloc[:, 3:16].columns[sorted_idx], model.coef_[sorted_idx])

#%%
print(model.feature_importances_)
#%%
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel

sel_ = SelectFromModel(LogisticRegression(C=1, penalty="l1", solver='liblinear'))
sel_.fit(train_features, np.ravel(train_labels,order="C"))
sel_.get_support()
X_train = pd.DataFrame(train_features)

#%%
selected_feat = X_train.columns[(sel_.get_support())]
print("total features: {}".format((X_train.shape[1])))
print("selected features: {}".format(len(selected_feat)))
print("features with coefficients shrank to zero: {}".format(
np.sum(sel_.estimator_.coef_ == 0)))
print(sel_.estimator_.coef_)

#%%
cell_features_scaled_pd = pd.DataFrame(data = cell_features_scaled, columns=["area", "perimeter", "radius", "aspect_ratio", "extent", "solidity", "equi_diameter","eccentricity","mean_cuvature","std_cuvature","roughness", "mean_val", "min_val", "max_val"])
print(cell_features_scaled_pd.head())

#%%
#use features as input
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
My_Data = [] 
with open ("/home/lunas/Documents/Uni/Masterarbeit/Data/data_aligned_cell_shapes_from_labeled.csv") as f:
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
scaler = StandardScaler()
print(PD_Flattend_shapes.columns)
contours_as_features_scaled = scaler.fit_transform(PD_Flattend_shapes)
print(contours_as_features_scaled)
#%%
#pca
data = []
pca = PCA(n_components=4)
pca_result = pca.fit_transform(contours_as_features_scaled)


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
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_title('scatter plot of first two PCs from PCA on features')
#legend1 = ax.legend(*scatter.legend_elements(num=2),
#                    loc="lower right", title="Classes")


plt.legend(handles=scatter.legend_elements()[0], labels=["tumor", "non-tumor", "outlier"], loc="lower right")
plt.legend(handles=scatter.legend_elements()[0], labels=["tumor", "non-tumor", "outlier"], loc="lower right")


#plt.savefig("/home/lunas/Documents/Uni/Masterarbeit/Data/plots/pca_from_labeled_tumor_vs_non_tumor_with_outlier.svg", format="svg")
#plt.savefig("/home/lunas/Documents/Uni/Masterarbeit/Data/plots/pca_from_labeled_tumor_vs_non_tumor_with_outlier_log_scale.svg", format="svg")
plt.show()

#%%
from sklearn.manifold import TSNE
import time

time_start = time.time()

tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300)
tsne_results = tsne.fit_transform(contours_as_features_scaled)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
#%%
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
scatter = ax.scatter(tsne_results[:,0], tsne_results[:,1], c=features["class_tumor"])
ax.set_xlabel('Dim1')
ax.set_ylabel('Dim2')
ax.set_title('scatter plot of first two Dims from tsne on features')
#legend1 = ax.legend(*scatter.legend_elements(num=2),
#                    loc="upper right", title="Classes")
plt.legend(handles=scatter.legend_elements()[0], labels=["tumor", "non-tumor", "outlier"], loc="upper right")



#plt.savefig("/home/lunas/Documents/Uni/Masterarbeit/Data/plots/t-sne_from_labeled_tumor_vs_non_tumor_with_outlier.svg", format="svg")
plt.show()


#%%