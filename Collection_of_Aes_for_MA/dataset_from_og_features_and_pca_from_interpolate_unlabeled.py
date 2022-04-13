#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 14:07:14 2022

@author: lunas
"""

import cv2
from  matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#create dataset from og features and pca from interpolate for unlabeled data 

lables = pd.read_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/data_only_lables_1.csv", header = None)#, index_col=0)  
pixel = pd.read_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/pd_pixelfeatures_from_og_contours.csv")
contour = pd.read_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/pd_contourfeatures_from_og_contours.csv")
pca = pd.read_csv('/home/lunas/Documents/Uni/Masterarbeit/Data/100_principal_components_from_interpolation_contours.csv')




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


colnames =  ["image", "cell","area", "perimeter", "radius", "aspect_ratio", "extent", "solidity", "equi_diameter","eccentricity","mean_cuvature","std_cuvature","roughness", "mean_val", "min_val", "max_val"]
count = 0
for i in pca.columns:
    count = count + 1
    name = "PC_" + str(count)
    colnames.append(name)
print(colnames)
#%%
X.columns = colnames
#%%
X.to_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/Feature_gathering_from_og_contours_and_pca_from_interpolate.csv", index=False)




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
#features = pd.read_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/Feature_gathering_from_og_contours_and_pca_from_interpolate.csv")
features = pd.read_csv("D:/26.01.22/Data/Feature_gathering_from_og_contours_and_pca_from_interpolate.csv")
#%%
#print(len(features))
#features.dropna(inplace=True)
#print(len(features))
for i in features.columns:
    print(i)
#%%
#print(features.iloc[:, 2:16].columns)
#%%
#scaler = StandardScaler()
#print(features.iloc[:, 2:16].columns)
#cell_features_scaled = scaler.fit_transform(features.iloc[:, 2:16])

##%
scaler = StandardScaler()
print(features.iloc[:, 2:16].columns)
cell_features_scaled = scaler.fit_transform(features.iloc[:, 2:16])
cell_features_scaled = np.nan_to_num(cell_features_scaled)
#%%
print(cell_features_scaled)
print(np.argwhere(np.isnan(cell_features_scaled)))
#%%
#use pca as features
print(features.iloc[:, 16:22].columns)
six_pcas_as_features = features.iloc[:, 16:22]



#%%
plt.figure(figsize=(12,10))
cor = pd.DataFrame(cell_features_scaled, columns=["area", "perimeter", "radius", "aspect_ratio", "extent", "solidity", "equi_diameter","eccentricity","mean_cuvature","std_cuvature","roughness", "mean_val", "min_val", "max_val"]).corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()
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
g = sns.PairGrid(features.iloc[:, 2:18], diag_sharey=False) # call seaborn's PairGrid function
g.map_lower(plt.scatter,edgecolors="w")     # show scatter plot matrix on lower triangle
g.map_diag(sns.distplot)                    # show distributions on main diagonal
g.map_upper(correl)                         # show correlation matrix on lower triangle (using function above)
#plt.savefig("/home/lunas/Documents/Uni/Masterarbeit/Data/plots/Features_Pairgrid_from_interpolate_contours.png")
plt.show()


#%%
#use k-means for 15 clusters (n clusters = n labels)
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
print(len(cell_features_scaled))
#%%
#pca
data = []
pca = PCA(n_components=3)
pca_result = pca.fit_transform(cell_features_scaled)
features["1pca_from_features_tumor_vs_non_tumor"] = pca_result[:,0]
features["2pca_from_features_tumor_vs_non_tumor"] = pca_result[:,1]
#data.append([pca_result[:,0],pca_result[:,1] ,pca_result[:,2]]) 
#ata.append(pca_result[:,1])
#ata.append(pca_result[:,2])

#df = pd.DataFrame(data, columns=['pca1', 'pca2', 'pca3'])

fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
#sns.scatterplot(data=features, x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue="area")
#sns.scatterplot(data=features, x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue="perimeter")
#sns.scatterplot(data=features, x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue="radius")
#sns.scatterplot(data=features, x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue="aspect_ratio")
#sns.scatterplot(data=features, x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue="extent")
#sns.scatterplot(data=features, x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue="solidity")
#sns.scatterplot(data=features, x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue="equi_diameter")
#sns.scatterplot(data=features, x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue="eccentricity")
#sns.scatterplot(data=features, x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue="mean_cuvature")
#sns.scatterplot(data=features, x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue="std_cuvature")
#sns.scatterplot(data=features, x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue="roughness")
#sns.scatterplot(data=features, x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue="mean_val")
#sns.scatterplot(data=features, x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue="min_val")
#sns.scatterplot(data=features, x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue="max_val")










#df['pca-two'] = pca_result[:,1] 
#df['pca-three'] = pca_result[:,2]
#print(len(pca_result[:,0]))
#print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
#print(pca.components_)
#df["class"] = features["class_tumor"]
#plt.scatter(pca_result[:,0], pca_result[:,1])

#fig=plt.figure()
#ax=fig.add_axes([0,0,1,1])
#scatter = ax.scatter(pca_result[:,0], pca_result[:,1])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
#ax.set_yscale('log')
#ax.set_xscale('log')
ax.set_title('scatter plot of forst two PCs from PCA on features')
legend1 = ax.legend(*scatter.legend_elements(num=1),
                    loc="lower right", title="Classes")
plt.show()


#plt.scatter(pca_result[:,0], pca_result[:,1], c=farbe)

#%%
print(pca_result.shape)
pca_pd =  pd.DataFrame(data=pca_result, columns=["pc1", "pc2", "pc3"])

pca_pd["image"] = features["image"]
pca_pd["cell"] = features["cell"]
print(pca_pd.head())

print(len(pca_pd[(pca_pd["pc1"] <= 12.5) & (pca_pd["pc2"] <= 10)]))
#%%
#tsne
from sklearn.manifold import TSNE
import time

time_start = time.time()

tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=300)
tsne_results = tsne.fit_transform(cell_features_scaled)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
#%%
#fig=plt.figure()
#ax=fig.add_axes([0,0,1,1])
#scatter = ax.scatter(tsne_results[:,0], tsne_results[:,1], c=features["class_tumor"])
features["1tsne_dim_from_features_tumor_vs_non_tumor"] = tsne_results[:,0]
features["2tsne_dim_from_features_tumor_vs_non_tumor"] = tsne_results[:,1]



#color for each feature
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
#sns.scatterplot(data=features, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue="area")
#sns.scatterplot(data=features, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue="perimeter")
#sns.scatterplot(data=features, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue="radius")
#sns.scatterplot(data=features, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue="aspect_ratio")
#sns.scatterplot(data=features, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue="extent")
#sns.scatterplot(data=features, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue="solidity")
#sns.scatterplot(data=features, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue="equi_diameter")
#sns.scatterplot(data=features, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue="eccentricity")
#sns.scatterplot(data=features, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue="mean_cuvature")
#sns.scatterplot(data=features, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue="std_cuvature")
#sns.scatterplot(data=features, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue="roughness")
#sns.scatterplot(data=features, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue="mean_val")
#sns.scatterplot(data=features, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue="min_val")
#sns.scatterplot(data=features, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue="max_val")





#fig=plt.figure()
#ax=fig.add_axes([0,0,1,1])
#scatter = ax.scatter(tsne_results[:,0], tsne_results[:,1])
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
    n_neighbors=10,
    min_dist=0.0,
    n_components=5,
    random_state=42,
).fit_transform(cell_features_scaled)

#%%
features["1umap_dim_from_features_tumor_vs_non_tumor"] = clusterable_embedding[:,0]
features["2umap_dim_from_features_tumor_vs_non_tumor"] = clusterable_embedding[:,1]


fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
#scatter = ax.scatter(tsne_results[:,0], tsne_results[:,1], c=features["class_tumor"])

#farbverlauf
#sns.scatterplot(data=features, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue="area")
#sns.scatterplot(data=features, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue="perimeter")
#sns.scatterplot(data=features, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue="radius")
#sns.scatterplot(data=features, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue="aspect_ratio")
#sns.scatterplot(data=features, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue="extent")
#sns.scatterplot(data=features, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue="solidity")
#sns.scatterplot(data=features, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue="equi_diameter")
#sns.scatterplot(data=features, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue="eccentricity")
#sns.scatterplot(data=features, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue="mean_cuvature")
#sns.scatterplot(data=features, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue="std_cuvature")
##sns.scatterplot(data=features, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue="roughness")
#sns.scatterplot(data=features, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue="mean_val")
#sns.scatterplot(data=features, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue="min_val")
sns.scatterplot(data=features, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue="max_val")






#fig, ax = plt.subplots()

#scatter = ax.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1])
ax.set_xlabel('Dim1')
ax.set_ylabel('Dim2')
ax.set_title('scatter plot of first two Dims from umap on features')
legend1 = ax.legend(*scatter.legend_elements(num=15),
                    loc="lower right", title="Classes")

ax.add_artist(legend1)

plt.show()


#%%
print(len(cell_features_scaled),len(features["class_tumor"]))
#%%
#random forest
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split# Split the data into training and testing sets
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
train_features, test_features, train_labels, test_labels = train_test_split(cell_features_scaled, features["class_tumor"], test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


X = cell_features_scaled
y = features["class_tumor"]
clf = RandomForestClassifier(max_depth=2, random_state=0)
#%%
clf.fit(X, y)



