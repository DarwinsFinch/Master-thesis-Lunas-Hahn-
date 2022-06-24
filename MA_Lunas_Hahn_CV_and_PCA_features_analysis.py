#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 11:59:54 2022

@author: lunas
"""



#this script is used to analyse the CV and PCA features for both the labeled cells and all cells

#import packages
import cv2
from  matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sb 
np.random.seed(42)
#%%

#from the combined dataset
lables = pd.read_csv("/media/lunas/Elements/combined_datasets_all_cells/All_cells_labled_data_out_interpolated_only_lables_labled_1.csv", header = None)#, index_col=0)  
pixel = pd.read_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/pd_pixelfeatures_from_interpolate_contours_from_labeled_all_cells.csv")
pixel_from_hed = pd.read_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/pd_pixelfeatures_from_HED_channel_from_interpolate_contours_from_labeled_all_cells.csv")
contour = pd.read_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/pd_contourfeatures_from_interpolate_contours_from_labeled_all_cells.csv")
pca = pd.read_csv('/home/lunas/Documents/Uni/Masterarbeit/Data/100_principal_components_from_interpolation_contours_from_labeled_all_cells.csv')


#%%
#remove first column
pca = pca.iloc[: , 1:]
pixel = pixel.iloc[: , 1:]
pixel_from_hed = pixel_from_hed.iloc[: , 1:]
contour = contour.iloc[: , 1:]

#%%
#Get one table for all features
X = pd.concat([lables, contour, pixel, pixel_from_hed, pca], axis=1, ignore_index=True)


#%%

#give column names to the table
colnames =  ["image", "cell", "class", "area", "perimeter", "radius", "aspect ratio", "extent", "solidity", "equivalent diameter","eccentricity","mean cuvature","std cuvature","roughness", "mean pixel value", "min pixel value", "max pixel value", "mean pixel value hematoxylin", "min pixel value hematoxylin", "max pixel value hematoxylin", "mean pixel value eosin", "min pixel value eosin", "max pixel value eosin"]
count = 0
for i in pca.columns:
    count = count + 1
    name = "PC_" + str(count)
    colnames.append(name)
print(colnames)

X.columns = colnames
print(X.iloc[:, 0:3])

#check if columns have missing values
nan_values = X.isna()

nan_columns = nan_values.any()


columns_with_nan = X.columns[nan_columns].tolist()

print(columns_with_nan)
for i in range(3,23):
    print(i)
    count_nan = X.iloc[:, i:i+1].isna().sum()
    print ('Count of NaN: ' + str(count_nan))
#%%
#safe df with all features
X.to_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/Feature_gathering_from_og_contours_with_interpolate_pca_from_all_cells.csv", index=False)




###feature analysis

#%%
import cv2
import os
from  matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from scipy import stats
#%%
#load all cells at the edges and remove them from the analysis
Edgy_cells = pd.read_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/Cells_at_the_edges_from_interpolate_contours_from_labeled_all_cells.csv", header = None) #for all cells

remove_these = Edgy_cells[1].tolist()

features = pd.read_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/Feature_gathering_from_og_contours_with_interpolate_pca_from_all_cells.csv") #all cells

features = features.drop(remove_these)

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
features.drop(features[features["class"] == "Positive"].index, inplace=True) #THIS DROPS ALL THE UNLABELED CELLS! TO USE THEM FOR THE ANALYSIS YOU HAVE TO KEEP THE POSITIVES
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
print(features["class"].value_counts())
#%%
#Get CV features and scale them 
scaler = StandardScaler()
print(features.iloc[:, 3:23].columns)
cell_features_scaled = scaler.fit_transform(features.iloc[:, 3:23])
cell_features_scaled = np.nan_to_num(cell_features_scaled)

#%%
#Get PCA features 
six_pcas_as_features = features.iloc[:, 23:43]
six_pcas_as_features = np.nan_to_num(six_pcas_as_features)
#%%
#use pca + cv-features as features
print(features.iloc[:, 3:43].columns)
cell_features_scaled = features.iloc[:, 3:43]
cell_features_scaled = np.nan_to_num(cell_features_scaled)
#%%
#calculate correlation matrix for features
plt.figure(figsize=(12,10))
cor = pd.DataFrame(cell_features_scaled, columns=["area", "perimeter", "radius", "aspect_ratio", "extent", "solidity", "equi_diameter","eccentricity","mean_cuvature","std_cuvature","roughness", "mean_val", "min_val", "max_val", "mean_val_hematoxylin", "min_val_hematoxylin", "max_val_hematoxylin", "mean_val_eosin", "min_val_eosin", "max_val_eosin"]).corr()
#cor = pd.DataFrame(cell_features_scaled, columns=["area", "perimeter", "radius", "aspect ratio", "extent", "solidity", "equivalent diameter","eccentricity","mean cuvature","std cuvature","roughness", "mean pixel value", "min pixel value", "max pixel value", "mean pixel value hematoxylin", "min pixel value hematoxylin", "max pixel value hematoxylin", "mean pixel value eosin", "min pixel value eosin", "max pixel value eosin"]).corr()
#cor = pd.DataFrame(six_pcas_as_features, columns=["PC_1", "PC_2", "PC_3", "PC_4", "PC_5", "PC_6", "PC_7", "PC_8", "PC_9", "PC_10", "PC_11", "PC_12", "PC_13", "PC_14", "PC_15", "PC_16", "PC_17", "PC_18", "PC_19", "PC_20"]).corr()
ax = plt.axes()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
#ax.set_title('Correlation of the Computer Vision features for the filtered dataset (64099 cells)')
ax.set_title('Correlation of the computer vision features for the filtered dataset')
#plt.show()
#plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/Correlation-of-the-Computer-Vision-features-for-the-filtered-dataset.png", bbox_inches='tight')
#plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/Correlation-of-the-Computer-Vision-features-for-the-filtered-dataset.svg", bbox_inches='tight')
plt.show()
#%%
#create beautiful correlation plot
# import libraries
import matplotlib.pyplot as plt
import seaborn as sns

#  define function for upper triangle of pair grid (correlation matrix)
def correl(*args, **kwargs):
    # determine pearson correlation between each feature
    corr_r = (args[0].corr(args[1], 'pearson'))
    # annotation of correlation values (number of fractions)
    corr_text = f"{corr_r:2.2f}"
    # set axes properties
    ax = plt.gca()
    ax.set_axis_off()
    #ax.axes.xaxis.set_visible(False)
    #ax.axes.yaxis.set_visible(False)
    # size and settings of colored squares for each entry
    marker_size = 15000
    ax.scatter([.5], [.5], s=marker_size, c=[corr_r],  marker="s",
               cmap="RdBu", vmin=-1, vmax=1, transform=ax.transAxes)
    font_size = 45
    #font_size = 90
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

# set up pair grid plot       
sns.set(font_scale=3) 
g = sns.PairGrid(features.iloc[:, 3:23], diag_sharey=False) # call seaborn's PairGrid function

#g = sns.PairGrid(features.iloc[:, 3:8], diag_sharey=False) # call seaborn's PairGrid function
g.map_lower(plt.scatter,edgecolors="w")     # show scatter plot matrix on lower triangle
g.map_diag(sns.displot)                    # show distributions on main diagonal
g.map_upper(correl)                         # show correlation matrix on lower triangle (using function above)
for ax in g.axes.flatten():
    # rotate x axis labels
    ax.set_xlabel(ax.get_xlabel(), rotation = 90)
    # rotate y axis labels
    ax.set_ylabel(ax.get_ylabel(), rotation = 0)
    # set y labels alignment
    ax.yaxis.get_label().set_horizontalalignment('right')
    #ax.axes.xaxis.set_visible(False)
    #ax.axes.yaxis.set_visible(False)
    
#plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/Correlation-of-the-Computer-Vision-features-for-the-filtered-dataset_husseinsplot_no_labels3.png", bbox_inches='tight')
#plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/Correlation-of-the-Computer-Vision-features-for-the-filtered-dataset_husseinsplot_no_labels3.svg", bbox_inches='tight')
plt.show()




#%%
#create a column with a label for tumor or non tumor
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


features.class_tumor = [mapping[item] for item in features.class_tumor]


#%%
#Use PCA
scaler = StandardScaler()
print(features[features["class"] != "Positive"].iloc[:, 3:23].columns)
#cell_features_scaled = scaler.fit_transform(features.iloc[:, 3:17])

#for cv features
#cell_features_scaled_only_labeled = scaler.fit_transform(features[features["class"] != "Positive"].iloc[:, 3:23])
#cell_features_scaled_only_labeled = np.nan_to_num(cell_features_scaled_only_labeled)

#for pca features
#six_pcas_as_features_only_labeled = scaler.fit_transform(features[features["class"] != "Positive"].iloc[:, 23:29])
#six_pcas_as_features_only_labeled = np.nan_to_num(six_pcas_as_features_only_labeled)

#for combined features
cell_features_scaled_only_labeled = scaler.fit_transform(features[features["class"] != "Positive"].iloc[:, 3:43])
cell_features_scaled_only_labeled = np.nan_to_num(cell_features_scaled_only_labeled)

#%%
#PCA
data = []
pca = PCA(n_components=3)
pca_result = pca.fit_transform(cell_features_scaled)
features["1pca_from_features_tumor_vs_non_tumor"] = pca_result[:,0]
features["2pca_from_features_tumor_vs_non_tumor"] = pca_result[:,1]
#pca_result = pca.fit_transform(np.nan_to_num(scaler.fit_transform(features[(features["class_tumor"] == 1)].iloc[:, 3:17])))
#pca_result = pca.fit_transform(cell_features_scaled)
pca_result = pca.fit_transform(cell_features_scaled_only_labeled)
#pca_result = pca.fit_transform(cell_features_scaled[0]) #area only
#data.append([pca_result[:,0],pca_result[:,1] ,pca_result[:,2]]) 
#ata.append(pca_result[:,1])
#ata.append(pca_result[:,2])

#df = pd.DataFrame(data, columns=['pca1', 'pca2', 'pca3'])

#features for all cells


#features for only labeled
#mask = features["class"] != "Positive"
features_only_labeled = features[features["class"] != "Positive"].copy()
print(features_only_labeled.columns)
print(len(features_only_labeled))
print(type(features_only_labeled))
print(type(pca_result[:,0]))
#features_only_labeled.drop(["1pca_from_features_tumor_vs_non_tumor", "2pca_from_features_tumor_vs_non_tumor"], axis=1)
#print(features_only_labeled.columns)
#features_only_labeled = features_only_labeled.assign(1pca_from_features_tumor_vs_non_tumor = pca_result[:,0])


features_only_labeled["1pca_from_features_tumor_vs_non_tumor"] = pca_result[:,0]
features_only_labeled["2pca_from_features_tumor_vs_non_tumor"] = pca_result[:,1]

#print(len(pca_result[:,0]))
#print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
#print(pca.components_)
#df["class"] = features["class_tumor"]
#plt.scatter(pca_result[:,0], pca_result[:,1])

#for pca
#pca_result = pca.fit_transform(six_pcas_as_features)
#features["1pca_from_features_tumor_vs_non_tumor"] = pca_result[:,0]
#features["2pca_from_features_tumor_vs_non_tumor"] = pca_result[:,1]
#features_only_labeled["1pca_from_features_tumor_vs_non_tumor"] = pca_result[:,0]
#features_only_labeled["2pca_from_features_tumor_vs_non_tumor"] = pca_result[:,1]

#%%
#create plots where the coloring is based on the value for a specific feature over all CV features
for i in ["area", "perimeter", "radius", "aspect ratio", "extent", "solidity", "equivalent diameter","eccentricity","mean cuvature","std cuvature","roughness", "mean pixel value", "min pixel value", "max pixel value", "mean pixel value hematoxylin", "min pixel value hematoxylin", "max pixel value hematoxylin", "mean pixel value eosin", "min pixel value eosin", "max pixel value eosin"]:
    print(i)
    fig=plt.figure()
    ax=fig.add_axes([0,0,1,1])
    sns.scatterplot(data=features, x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue=i, style="class")
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title('First two PCs from PCA on CV features colored by ' + i )
    ax.legend(loc='right', bbox_to_anchor=(1.4, 0.5), ncol=1)
    plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/PCA-of-the-Computer-Vision-features-for-the-filtered-dataset-colored-for-features_" + i +".svg", bbox_inches='tight')
    plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/PCA-of-the-Computer-Vision-features-for-the-filtered-dataset-colored-for-features_" + i +".png", bbox_inches='tight')
    plt.show()

#%%
#create plots where the coloring is based on the cell class
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])

#this
#coloring for each class
tum = plt.scatter(features[features["class"] == "Tumor"]["1pca_from_features_tumor_vs_non_tumor"], features[features["class"] == "Tumor"]["2pca_from_features_tumor_vs_non_tumor"], color="red", label='Tumor', facecolors='none')
nsl = plt.scatter(features[features["class"] == "normal small lymphocyte"]["1pca_from_features_tumor_vs_non_tumor"], features[features["class"] == "normal small lymphocyte"]["2pca_from_features_tumor_vs_non_tumor"], color="green", label='normal small lymphocyte', facecolors='none')
ll = plt.scatter(features[features["class"] == "large leucocyte"]["1pca_from_features_tumor_vs_non_tumor"], features[features["class"] == "large leucocyte"]["2pca_from_features_tumor_vs_non_tumor"], color="blue", label='large leucocyte', facecolors='none')
stroma = plt.scatter(features[features["class"] == "Stroma"]["1pca_from_features_tumor_vs_non_tumor"], features[features["class"] == "Stroma"]["2pca_from_features_tumor_vs_non_tumor"], color="yellow", label='Stroma', facecolors='none')
ec = plt.scatter(features[features["class"] == "Epithelial cell"]["1pca_from_features_tumor_vs_non_tumor"], features[features["class"] == "Epithelial cell"]["2pca_from_features_tumor_vs_non_tumor"], color="black", label='Epithelial cell', facecolors='none')
positive = plt.scatter(features[features["class"] == "Positive"]["1pca_from_features_tumor_vs_non_tumor"], features[features["class"] == "Positive"]["2pca_from_features_tumor_vs_non_tumor"], color="beige", label='Epithelial cell', facecolors='none', alpha = 0.1)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_title('First two PCs from PCA on CV features')


ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')


#plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/PCA-of-the-Computer-Vision-features-for-the-filtered-dataset-only-labeled.svg", bbox_inches='tight')
#plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/PCA-of-the-Computer-Vision-features-for-the-filtered-dataset-only-labeled.png", bbox_inches='tight')

plt.show()

#%%
#get outliers
#outlier right
outlier_right = features[(features["1pca_from_features_tumor_vs_non_tumor"] > 12)]
print(len(outlier_right))
print(outlier_right)
#%%
#get picture for some outliers
outlier_top_right = features[(features["1pca_from_features_tumor_vs_non_tumor"] > 11) & (features["2pca_from_features_tumor_vs_non_tumor"] > 4)]
outlier_top = features[(features["1pca_from_features_tumor_vs_non_tumor"] < 5) & (features["2pca_from_features_tumor_vs_non_tumor"] > 6.5)]
outlier_down_right = features[(features["1pca_from_features_tumor_vs_non_tumor"] > 13) & (features["2pca_from_features_tumor_vs_non_tumor"] < -2)]
outlier_down = features[(features["1pca_from_features_tumor_vs_non_tumor"] < 4) & (features["2pca_from_features_tumor_vs_non_tumor"] < -5.5)]
normals_left = features[(features["1pca_from_features_tumor_vs_non_tumor"] < -4.5) & (features["2pca_from_features_tumor_vs_non_tumor"] < 0.5) & (features["2pca_from_features_tumor_vs_non_tumor"] > 0)]
high_mean_val = features[(features["1pca_from_features_tumor_vs_non_tumor"] > 0) & (features["2pca_from_features_tumor_vs_non_tumor"] < 5) & (features["mean_val"] > 190)]
low_mean_val = features[(features["1pca_from_features_tumor_vs_non_tumor"] > 0) & (features["2pca_from_features_tumor_vs_non_tumor"] < 5) & (features["mean_val"] < 75)]

print(outlier_top_right.iloc[0]["image"])
print(len(normals_left))
print(normals_left.iloc[0]["image"])

columns = 1
rows = 1
count = 1
fig = plt.figure()
#count  = 1
#for index, row in upper_right.iterrows():
size = len(normals_left.iloc[2]["image"])
    #print(row["image"][:size-5], row["cell"])
filename = str(normals_left.iloc[2]["cell"]) + "_" + normals_left.iloc[2]["image"][:size-5]
#    print(filename)

Image1 = cv2.imread("/media/lunas/Samsung USB/data/Single_cell_images_2/" + filename)
fig.add_subplot(rows, columns, count)
plt.imshow(Image1)
#plt.axis('off')
#    count = count + 1
#plt.savefig("/media/lunas/Samsung USB/data/plots/Compare_density_clusters/tumor_upper_right.svg")
plt.show()

#%%
#2d density plot
features["1pca_from_features_tumor_vs_non_tumor"] = pca_result[:,0]
features["2pca_from_features_tumor_vs_non_tumor"] = pca_result[:,1]

fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
#ax.set_title('scatter plot of first two PCs from PCA on features')
ax.set_xlim(-5, 7)
ax.set_ylim(-6, 6)
#sns.kdeplot(x=pca_result[:,0], y=pca_result[:,1], hue=features["class_tumor"],c=features["color_tumor"], fill=True, label=["tumor", "non-tumor"])
sns.kdeplot(x=features[features["class"] == "Tumor"]["1pca_from_features_tumor_vs_non_tumor"], y=features[features["class"] == "Tumor"]["2pca_from_features_tumor_vs_non_tumor"], fill=True, alpha=.3, cbar = True, color="red", label='Tumor', cbar_kws= {'label': 'Tumor','ticks': [0, 100]}).set(title='Density plot of the first two principal components from the computer vision features of the tumor cells')
plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/Density_Plot_of_PCA-of-the-Computer-Vision-features-for-the-filtered-dataset-tumor_cells.svg", bbox_inches='tight')
plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/Density_Plot_of_PCA-of-the-Computer-Vision-features-for-the-filtered-dataset-tumor_cells.png", bbox_inches='tight')

#ax.set_xlim(-7, 3)
#ax.set_ylim(-5, 6)
#sns.kdeplot(x=features[features["class"] != "Tumor"]["1pca_from_features_tumor_vs_non_tumor"], y=features[features["class"] != "Tumor"]["2pca_from_features_tumor_vs_non_tumor"], fill=True, alpha=.3, cbar = True, color="green", label='Non_Tumor', cbar_kws= {'label': 'Non_Tumor','ticks': [0, 100]}).set(title='Density plot of the first two principal components from the computer vision features of the non-tumor cells')
#plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/Density_Plot_of_PCA-of-the-Computer-Vision-features-for-the-filtered-dataset-non-tumor_cells.svg", bbox_inches='tight')
#plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/Density_Plot_of_PCA-of-the-Computer-Vision-features-for-the-filtered-dataset-non-tumor_cells.png", bbox_inches='tight')

#plt.legend()
plt.show()


#t-SNE
from sklearn.manifold import TSNE
import time

time_start = time.time()

tsne = TSNE(n_components=2, verbose=1, perplexity=100, n_iter=300)
#tsne_results = tsne.fit_transform(six_pcas_as_features)
#this
tsne_results = tsne.fit_transform(cell_features_scaled_only_labeled)
#tsne_results = tsne.fit_transform(cell_features_scaled)
#for pca
#tsne_results = tsne.fit_transform(six_pcas_as_features)
#tsne_results = tsne.fit_transform(np.nan_to_num(features.iloc[:, 3:17]))
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
#%%
#only labeled
features_only_labeled = features[features["class"] != "Positive"].copy()
features_only_labeled["1tsne_dim_from_features_tumor_vs_non_tumor"] = tsne_results[:,0]
features_only_labeled["2tsne_dim_from_features_tumor_vs_non_tumor"] = tsne_results[:,1]


#%%
#create t-SNE plots colored for each feature 
for i in ["area", "perimeter", "radius", "aspect ratio", "extent", "solidity", "equivalent diameter","eccentricity","mean cuvature","std cuvature","roughness", "mean pixel value", "min pixel value", "max pixel value", "mean pixel value hematoxylin", "min pixel value hematoxylin", "max pixel value hematoxylin", "mean pixel value eosin", "min pixel value eosin", "max pixel value eosin"]:
    print(i)
    fig=plt.figure()
    ax=fig.add_axes([0,0,1,1])
    sns.scatterplot(data=features_only_labeled, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue=i, style="class")
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('First two dimensions from t-SNE on CV features colored by ' + i  )
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/TSNE-of-the-Computer-Vision-features-for-the-filtered-dataset-colored-for-features_" + i +".svg", bbox_inches='tight')
    plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/TSNE-of-the-Computer-Vision-features-for-the-filtered-dataset-colored-for-features_" + i +".png", bbox_inches='tight')
    plt.show()
    #plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/TSNE-of-the-Computer-Vision-features-for-the-filtered-dataset-colored-for-features_" + i +".svg")
#%%
#t-SNE plot colored for the cell classes 
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])

#this
#only labeled
#all cells
tum = plt.scatter(features_only_labeled[features_only_labeled["class"] == "Tumor"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features_only_labeled[features_only_labeled["class"] == "Tumor"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="red", label='Tumor', facecolors='none')
nsl = plt.scatter(features_only_labeled[features_only_labeled["class"] == "normal small lymphocyte"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features_only_labeled[features_only_labeled["class"] == "normal small lymphocyte"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="green", label='Normal small Lymphocyte', facecolors='none')
ll = plt.scatter(features_only_labeled[features_only_labeled["class"] == "large leucocyte"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features_only_labeled[features_only_labeled["class"] == "large leucocyte"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="blue", label='Large Leucocyte', facecolors='none')
stroma = plt.scatter(features_only_labeled[features_only_labeled["class"] == "Stroma"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features_only_labeled[features_only_labeled["class"] == "Stroma"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="yellow", label='Stroma', facecolors='none')
ec = plt.scatter(features_only_labeled[features_only_labeled["class"] == "Epithelial cell"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features_only_labeled[features_only_labeled["class"] == "Epithelial cell"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="black", label='Epithelial Cell', facecolors='none')
#positive = plt.scatter(features[features["class"] == "Positive"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "Positive"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="beige", label='Unlabeled', facecolors='none', alpha = 0.1)
#positive = plt.scatter(features[features["class"] == "Positive"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "Positive"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="red", label='Unlabeled', facecolors='none', alpha = 1)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.85))
ax.set_title('First two dimensions from t-SNE on CV features')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
#plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/TSNE-of-the-Computer-Vision-features-for-the-filtered-dataset-only-labeled.svg", bbox_inches='tight')
#plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/TSNE-of-the-Computer-Vision-features-for-the-filtered-dataset-only-labeled.png", bbox_inches='tight')


plt.show()

#%%
#get outliers

left_big_cluster_tsne = features_only_labeled[(features_only_labeled["1tsne_dim_from_features_tumor_vs_non_tumor"] < 2.5) & (features_only_labeled["class"] == "Tumor") & (features_only_labeled["2tsne_dim_from_features_tumor_vs_non_tumor"] > 1)]
right_bottom_small_cluster_tsne = features_only_labeled[(features_only_labeled["1tsne_dim_from_features_tumor_vs_non_tumor"] < 2.5) & (features_only_labeled["class"] == "Tumor") & (features_only_labeled["2tsne_dim_from_features_tumor_vs_non_tumor"] < -5) & (features_only_labeled["1tsne_dim_from_features_tumor_vs_non_tumor"] > -4)]


print(len(left_big_cluster_tsne), len(right_bottom_small_cluster_tsne))
#print(right_bottom_small_cluster_tsne.head)

#left_bottom_tsne_image_and_cell = left_bottom_tsne[["image", "cell"]].copy()
#left_bottom_tsne_image_and_cell.to_csv("/media/lunas/Elements/Results_for_Masterarbeit/left_bottom_tsne_image_and_cell.csv") 
#right_tsne_image_and_cell = right_tsne[["image", "cell"]].copy()
#right_tsne_image_and_cell.to_csv("/media/lunas/Elements/Results_for_Masterarbeit/right_tsne_image_and_cell.csv")
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
ax.set_xlabel('Dim1')
ax.set_ylabel('Dim2')
#ax.set_title('scatter plot of first two PCs from PCA on features')
#tumor
ax.set_xlim(-13, 11)
ax.set_ylim(-7, 11)
#sns.kdeplot(x=pca_result[:,0], y=pca_result[:,1], hue=features["class_tumor"],c=features["color_tumor"], fill=True, label=["tumor", "non-tumor"])
sns.kdeplot(x=features[features["class"] == "Tumor"]["1tsne_dim_from_features_tumor_vs_non_tumor"], y=features[features["class"] == "Tumor"]["2tsne_dim_from_features_tumor_vs_non_tumor"], fill=True, alpha=.3, cbar = True, color="red", label='Tumor', cbar_kws= {'label': 'Tumor','ticks': [0, 100]}).set(title='Denisty plot of the first two TSNE dimensions from the computer vision features of the tumor cells')
plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/Density_Plot_of_TSNE-of-the-Computer-Vision-features-for-the-filtered-dataset-tumor_cells.svg", bbox_inches='tight')
plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/Density_Plot_of_TSNE-of-the-Computer-Vision-features-for-the-filtered-dataset-tumor_cells.png", bbox_inches='tight')



#non tumor
#ax.set_xlim(-8, 10)
#ax.set_ylim(-12, 9)
#sns.kdeplot(x=features[features["class"] != "Tumor"]["1tsne_dim_from_features_tumor_vs_non_tumor"], y=features[features["class"] != "Tumor"]["2tsne_dim_from_features_tumor_vs_non_tumor"], fill=True, alpha=.3, cbar = True, color="green", label='Non_Tumor', cbar_kws= {'label': 'Non_Tumor','ticks': [0, 100]}).set(title='Denisty plot of the first two TSNE dimensions from the computer vision features of the non-tumor cells')
#plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/Density_Plot_of_TSNE-of-the-Computer-Vision-features-for-the-filtered-dataset-nontumor_cells.svg", bbox_inches='tight')
#plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/Density_Plot_of_TSNE-of-the-Computer-Vision-features-for-the-filtered-dataset-nontumor_cells.png", bbox_inches='tight')

#sns.kdeplot(x=features[features["class"] != "Tumor"]["1tsne_dim_from_features_tumor_vs_non_tumor"], y=features[features["class"] != "Tumor"]["2tsne_dim_from_features_tumor_vs_non_tumor"], fill=True, alpha=.3, cbar = True, color="green", label='Non_Tumor', cbar_kws= {'label': 'Non_Tumor','ticks': [0, 100]}).set(title='KDE_of_Tsne_for_non_tumor')
 
#sns.kdeplot(x=pca_result[:,0], y=pca_result[:,1], hue=features["class_tumor"],c=features["color_tumor"], fill=True, label=["tumor", "non-tumor"])
#sns.kdeplot(x=features[features["class"] == "Tumor"]["1pca_from_features_tumor_vs_non_tumor"], y=features[features["class"] == "Tumor"]["2pca_from_features_tumor_vs_non_tumor"], fill=True, alpha=.3, cbar = True, color="red", label='Tumor', cbar_kws= {'label': 'Tumor','ticks': [0, 100]}).set(title='Denisty plot of the first two principal components from the computer vision features of the tumor cells')
#plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/Density_Plot_of_PCA-of-the-Computer-Vision-features-for-the-filtered-dataset-tumor_cells.svg", bbox_inches='tight')
#plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/Density_Plot_of_PCA-of-the-Computer-Vision-features-for-the-filtered-dataset-tumor_cells.png", bbox_inches='tight')

#ax.set_xlim(-7, 3)
#ax.set_ylim(-5, 6)
#sns.kdeplot(x=features[features["class"] != "Tumor"]["1pca_from_features_tumor_vs_non_tumor"], y=features[features["class"] != "Tumor"]["2pca_from_features_tumor_vs_non_tumor"], fill=True, alpha=.3, cbar = True, color="green", label='Non_Tumor', cbar_kws= {'label': 'Non_Tumor','ticks': [0, 100]}).set(title='Denisty plot of the first two principal components from the computer vision features of the non-tumor cells')
#plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/Density_Plot_of_PCA-of-the-Computer-Vision-features-for-the-filtered-dataset-non-tumor_cells.svg", bbox_inches='tight')
#plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/Density_Plot_of_PCA-of-the-Computer-Vision-features-for-the-filtered-dataset-non-tumor_cells.png", bbox_inches='tight')



#plt.legend()
plt.show()
#%%
#UMAP
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
#).fit_transform(cell_features_scaled)
).fit_transform(cell_features_scaled_only_labeled)
#).fit_transform(six_pcas_as_features)
#input
#only_tumor_and_snl
#np.nan_to_num(features.iloc[:, 3:17])
#%%
data = np.column_stack((clusterable_embedding[:,0], clusterable_embedding[:,1]))
data = data.tolist()
#data = data[0:100]
print(len(data))

#%%
features_only_labeled = features[features["class"] != "Positive"].copy()

features_only_labeled["1umap_dim_from_features_tumor_vs_non_tumor"] = clusterable_embedding[:,0]
features_only_labeled["2umap_dim_from_features_tumor_vs_non_tumor"] = clusterable_embedding[:,1]

#features["1umap_dim_from_features_tumor_vs_non_tumor"] = clusterable_embedding[:,0]
#features["2umap_dim_from_features_tumor_vs_non_tumor"] = clusterable_embedding[:,1]

#%%
print(features.columns)
#%%
#UMAP plots colored for each CV feature 
for i in ["area", "perimeter", "radius", "aspect ratio", "extent", "solidity", "equivalent diameter","eccentricity","mean cuvature","std cuvature","roughness", "mean pixel value", "min pixel value", "max pixel value", "mean pixel value hematoxylin", "min pixel value hematoxylin", "max pixel value hematoxylin", "mean pixel value eosin", "min pixel value eosin", "max pixel value eosin"]:
    print(i)
    fig=plt.figure()
    ax=fig.add_axes([0,0,1,1])
    sns.scatterplot(data=features_only_labeled, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue=i, style="class")
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('First two dimensions from UMAP on CV features colored by ' + i)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/UMAP-of-the-Computer-Vision-features-for-the-filtered-dataset-colored-for-features_" + i +".svg", bbox_inches='tight')
    #plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/UMAP-of-the-Computer-Vision-features-for-the-filtered-dataset-colored-for-features_" + i +".png", bbox_inches='tight')
    plt.show()
    
    #plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/UMAP-of-the-Computer-Vision-features-for-the-filtered-dataset-only-labeled-colored-for-features_" + i +".svg")
#%%
#UMAP plot colored by cell class
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])

#this
#only labeled
tum = plt.scatter(features_only_labeled[features_only_labeled["class"] == "Tumor"]["1umap_dim_from_features_tumor_vs_non_tumor"], features_only_labeled[features_only_labeled["class"] == "Tumor"]["2umap_dim_from_features_tumor_vs_non_tumor"], color="red", label='Tumor', facecolors='none')
nsl = plt.scatter(features_only_labeled[features_only_labeled["class"] == "normal small lymphocyte"]["1umap_dim_from_features_tumor_vs_non_tumor"], features_only_labeled[features_only_labeled["class"] == "normal small lymphocyte"]["2umap_dim_from_features_tumor_vs_non_tumor"], color="green", label='normal small lymphocyte', facecolors='none')
ll = plt.scatter(features_only_labeled[features_only_labeled["class"] == "large leucocyte"]["1umap_dim_from_features_tumor_vs_non_tumor"], features_only_labeled[features_only_labeled["class"] == "large leucocyte"]["2umap_dim_from_features_tumor_vs_non_tumor"], color="blue", label='large leucocyte', facecolors='none')
stroma = plt.scatter(features_only_labeled[features_only_labeled["class"] == "Stroma"]["1umap_dim_from_features_tumor_vs_non_tumor"], features_only_labeled[features_only_labeled["class"] == "Stroma"]["2umap_dim_from_features_tumor_vs_non_tumor"], color="yellow", label='Stroma', facecolors='none')
ec = plt.scatter(features_only_labeled[features_only_labeled["class"] == "Epithelial cell"]["1umap_dim_from_features_tumor_vs_non_tumor"], features_only_labeled[features_only_labeled["class"] == "Epithelial cell"]["2umap_dim_from_features_tumor_vs_non_tumor"], color="black", label='Epithelial cell', facecolors='none')
#positive = plt.scatter(features[features["class"] == "Positive"]["1umap_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "Positive"]["2umap_dim_from_features_tumor_vs_non_tumor"], color="beige", label='Unlabeled', facecolors='none', alpha = 0.1)
#positive = plt.scatter(features[features["class"] == "Positive"]["1umap_dim_from_features_tumor_vs_non_tumor"], features[features["class"] == "Positive"]["2umap_dim_from_features_tumor_vs_non_tumor"], color="red", label='Unlabeled', facecolors='none', alpha = 1)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.85))
ax.set_title('First two dimensions from UMAP on CV features')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/UMAP-of-the-Computer-Vision-features-for-the-filtered-dataset-only-labeled.svg", bbox_inches='tight')
plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/UMAP-of-the-Computer-Vision-features-for-the-filtered-dataset-only-labeled.png", bbox_inches='tight')

plt.show()
#%%
#get outliers 
right_umap = features_only_labeled[(features_only_labeled["1umap_dim_from_features_tumor_vs_non_tumor"] > 7) & (features_only_labeled["class"] == "Tumor")]
right_top_umap = features_only_labeled[(features_only_labeled["1umap_dim_from_features_tumor_vs_non_tumor"] > 7) & (features_only_labeled["class"] == "Tumor") & (features_only_labeled["2umap_dim_from_features_tumor_vs_non_tumor"] > 4.5)]
left_umap = features_only_labeled[(features_only_labeled["1umap_dim_from_features_tumor_vs_non_tumor"] < 6) & (features_only_labeled["class"] == "Tumor")]

print(len(features_only_labeled), len(right_umap), len(right_top_umap), len(left_umap))


right_top_umap_image_and_cell = right_top_umap[["image", "cell"]].copy()
right_top_umap_image_and_cell.to_csv("/media/lunas/Elements/Results_for_Masterarbeit/right_top_umap_image_and_cell.csv") 
left_umap_image_and_cell = left_umap[["image", "cell"]].copy()
left_umap_image_and_cell.to_csv("/media/lunas/Elements/Results_for_Masterarbeit/left_umap_image_and_cell.csv")


#%%

#%%
#UMAP density plot


fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.set_xlabel('Dim1')
ax.set_ylabel('Dim2')
#ax.set_title('scatter plot of first two PCs from PCA on features')
#tumor
#x.set_xlim(1, 12)
#ax.set_ylim(1, 7)
#sns.kdeplot(x=pca_result[:,0], y=pca_result[:,1], hue=features["class_tumor"],c=features["color_tumor"], fill=True, label=["tumor", "non-tumor"])
#sns.kdeplot(x=features_only_labeled[features_only_labeled["class"] == "Tumor"]["1umap_dim_from_features_tumor_vs_non_tumor"], y=features_only_labeled[features_only_labeled["class"] == "Tumor"]["2umap_dim_from_features_tumor_vs_non_tumor"], fill=True, alpha=.3, cbar = True, color="red", label='Tumor', cbar_kws= {'label': 'Tumor','ticks': [0, 100]}).set(title='Denisty plot of the first two UMAP dimensions from the computer vision features of the tumor cells')
#plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/Density_Plot_of_UMAP-of-the-Computer-Vision-features-for-the-filtered-dataset-tumor_cells.svg", bbox_inches='tight')
#plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/Density_Plot_of_UMAP-of-the-Computer-Vision-features-for-the-filtered-dataset-tumor_cells.png", bbox_inches='tight')



#non tumor
ax.set_xlim(-9, 10)
ax.set_ylim(-13, 6)
sns.kdeplot(x=features_only_labeled[features_only_labeled["class"] != "Tumor"]["1tsne_dim_from_features_tumor_vs_non_tumor"], y=features_only_labeled[features_only_labeled["class"] != "Tumor"]["2tsne_dim_from_features_tumor_vs_non_tumor"], fill=True, alpha=.3, cbar = True, color="green", label='Non_Tumor', cbar_kws= {'label': 'Non_Tumor','ticks': [0, 100]}).set(title='Denisty plot of the first two UMAP dimensions from the computer vision features of the non-tumor cells')
plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/Density_Plot_of_UMAP-of-the-Computer-Vision-features-for-the-filtered-dataset-nontumor_cells.svg", bbox_inches='tight')
plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/Density_Plot_of_UMAP-of-the-Computer-Vision-features-for-the-filtered-dataset-nontumor_cells.png", bbox_inches='tight')

#%%
#get outliers and plot the images from tumor
upper_right = features[(features["1umap_dim_from_features_tumor_vs_non_tumor"] > 8) & (features["class"] == "Tumor")]
upper_middle = features[(features["1umap_dim_from_features_tumor_vs_non_tumor"] < 8) & (features["1umap_dim_from_features_tumor_vs_non_tumor"] > 4) & (features["2umap_dim_from_features_tumor_vs_non_tumor"] > 3) & (features["2umap_dim_from_features_tumor_vs_non_tumor"] < 4) & (features["class"] == "Tumor")]
middle = features[(features["1umap_dim_from_features_tumor_vs_non_tumor"] < 4.5) & (features["1umap_dim_from_features_tumor_vs_non_tumor"] > 3) & (features["2umap_dim_from_features_tumor_vs_non_tumor"] > 2) & (features["2umap_dim_from_features_tumor_vs_non_tumor"] < 3) & (features["class"] == "Tumor")]
print(len(upper_right), len(upper_middle), len(middle))

#%%
#RANDOM FOREST
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


#train_features, test_features, train_labels, test_labels = train_test_split(cell_features_scaled, features["class_tumor"], test_size = 0.25, random_state = 42, stratify=features["class_tumor"])

#this
#only labeled cells
train_features, test_features, train_labels, test_labels = train_test_split(cell_features_scaled_only_labeled, features[features["class"] != "Positive"]["class"], test_size = 0.2, random_state = 3, stratify=features[features["class"] != "Positive"]["class"])
#only labeled cells tumor vs non tumor
#train_features, test_features, train_labels, test_labels = train_test_split(cell_features_scaled_only_labeled, features[features["class"] != "Positive"]["class_tumor"], test_size = 0.2, random_state = 3, stratify=features[features["class"] != "Positive"]["class_tumor"])

#using PCA as predictors all cells
#train_features, test_features, train_labels, test_labels = train_test_split(six_pcas_as_features, features[features["class"] != "Positive"]["class"], test_size = 0.2, random_state = 3, stratify=features[features["class"] != "Positive"]["class"])
#using PCA as predictors all cells, tunmor vs non-tumor
#train_features, test_features, train_labels, test_labels = train_test_split(six_pcas_as_features, features[features["class"] != "Positive"]["class_tumor"], test_size = 0.2, random_state = 3, stratify=features[features["class"] != "Positive"]["class_tumor"])

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

    y_pred_test = rf.predict(train_features)
    #print(y_pred_test)
    print("accuracy score for train", accuracy_score(train_labels, y_pred_test))
    #print(accuracy_score)
    y_pred_test = rf.predict(test_features)
    print("accuracy score for test",accuracy_score(test_labels, y_pred_test))
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
print(rf_random.best_params_)
#%%
#no overfitting with these params
#rf = RandomForestClassifier(random_state = 42, n_estimators=256, min_samples_split=10, min_samples_leaf=2, max_features="sqrt", max_depth=4, bootstrap=False)
#no overfitting with these params
rf = RandomForestClassifier(random_state = 42, n_estimators=64, min_samples_split=10, min_samples_leaf=2, max_features="sqrt", max_depth=4, bootstrap=False)
#best params from the cv search for train
#rf = RandomForestClassifier(random_state = 42, n_estimators=64, min_samples_split=10, min_samples_leaf=2, max_features="sqrt", max_depth=56, bootstrap=False)
#best params from the cv search for train
#rf = RandomForestClassifier(random_state = 42, n_estimators=256, min_samples_split=2, min_samples_leaf=2, max_features="sqrt", max_depth=12, bootstrap=False)
#best params from the cv search for train tumor non tumor
#rf = RandomForestClassifier(random_state = 42, n_estimators=64, min_samples_split=10, min_samples_leaf=4, max_features="sqrt", max_depth=6, bootstrap=False)
print(rf.get_params())
rf.fit(train_features, train_labels)


y_pred_test = rf.predict(train_features)
#print(y_pred_test)
print("accuracy score for train", accuracy_score(train_labels, y_pred_test))
#print(accuracy_score)
y_pred_test = rf.predict(test_features)
print("accuracy score for test",accuracy_score(test_labels, y_pred_test))
#print(accuracy_score)
#y_pred_test = rf.predict(validate_features)
#print("accuracy score for validate",accuracy_score(validate_labels, y_pred_test))
#%%
#feature importance on test,train,val
#perm_importance = permutation_importance(rf, train_features, train_labels)
perm_importance = permutation_importance(rf, test_features, test_labels)
#perm_importance = permutation_importance(rf, validate_features, validate_labels)
sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(features.iloc[:, 3:23].columns[sorted_idx], perm_importance.importances_mean[sorted_idx])

plt.xlabel("Permutation Feature Importance")
plt.title("Permutation Feature Importance from the Random Forest")

plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/Random_Forest_Permutated_Feature_Importance.svg", bbox_inches='tight')
plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/Random_Forest_Permutated_Feature_Importance.png", bbox_inches='tight')
#%%
print(cell_features_scaled[:, 0:1])
print(cell_features_scaled.shape)
#%%
#using the single features to train the RF to see how predicitve they are on their own
ll = []
for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]:
    
    train_features, test_features, train_labels, test_labels = train_test_split(cell_features_scaled[:, i:i + 1] , features["class_tumor"], test_size = 0.2, random_state = 42, stratify=features["class_tumor"])
    #train_features, validate_features, train_labels, validate_labels = train_test_split(train_features , train_labels, test_size = 0.2, random_state = 42, stratify=train_labels)
    
    

    
    
    rf = RandomForestClassifier(random_state = 42, n_estimators=64, min_samples_split=10, min_samples_leaf=2, max_features="sqrt", max_depth=4, bootstrap=False)
    rf.fit(train_features, train_labels)

    print(features.iloc[:, 3 + i: 4 + i].columns)
    y_pred_test = rf.predict(train_features)
    #print(y_pred_test)
    print("accuracy score for train", accuracy_score(train_labels, y_pred_test))
    #ll.append(accuracy_score(train_labels, y_pred_test))
    #print(accuracy_score)
    y_pred_test = rf.predict(test_features)
    print("accuracy score for test",accuracy_score(test_labels, y_pred_test))
    ll.append(accuracy_score(test_labels, y_pred_test))
    #print(accuracy_score)
    #y_pred_test = rf.predict(validate_features)
    #print("accuracy score for validate",accuracy_score(validate_labels, y_pred_test))
#%%
#plot how predictive single features are
print(features.iloc[:, 3:4].columns)
fig, ax = plt.subplots()
x_pos = [0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380]#,56,60,64,68]
plt.bar(x_pos, ll, width = 4)
plt.xticks(x_pos, ["area", "perimeter", "radius", "aspect_ratio", "extent", "solidity", "equi_diameter","eccentricity","mean_cuvature","std_cuvature","roughness", "mean_val", "min_val", "max_val", "mean_val_hematoxylin", "min_val_hematoxylin", "max_val_hematoxylin", "mean_val_eosin", "min_val_eosin", "max_val_eosin"], size = 10, rotation='vertical')
ax.set_xlabel('Features')
ax.set_ylabel('Score')
ax.set_title('accuracy score from the rf with only one feature')
plt.show() 
#plt.bar(["area", "perimeter", "radius", "aspect_ratio", "extent", "solidity", "equi_diameter","eccentricity","mean_cuvature","std_cuvature","roughness", "mean_val", "min_val", "max_val"], ll, width = 0.2)

#%%
#lasso regression
from sklearn.linear_model import LogisticRegression
#for tumor vs non tumor
clf = LogisticRegression(random_state=0, penalty='l1',solver='liblinear', class_weight="balanced").fit(train_features , train_labels)
#for multiclass
#clf = LogisticRegression(multi_class='multinomial', solver='lbfgs').fit(train_features , train_labels)
print("mean accuracy for train", clf.score(train_features, train_labels))
print("mean accuracy for test", clf.score(test_features, test_labels))
#print(clf.coef_)
#%%
#calculate feature importance
importance = clf.coef_
# summarize feature importance
for i,v in enumerate(importance[0]):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
x_pos = [0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380]
#plt.bar([x for x in range(len(importance[0]))], importance[0])
#plt.barh()
plt.bar(x_pos, importance[0], width = 4)
plt.xticks(x_pos, ["area", "perimeter", "radius", "aspect_ratio", "extent", "solidity", "equi_diameter","eccentricity","mean_cuvature","std_cuvature","roughness", "mean_val", "min_val", "max_val", "mean_val_hematoxylin", "min_val_hematoxylin", "max_val_hematoxylin", "mean_val_eosin", "min_val_eosin", "max_val_eosin"], size = 4)
plt.show()
#%%
#plot feature importance
print(perm_importance.importances_mean)
print(importance[0].argsort())
#sorted_idx = perm_importance.importances_mean.argsort()
sorted_idx2 = importance[0].argsort()

plt.barh(features.iloc[:, 3:23].columns[sorted_idx2], importance[0][sorted_idx2])

plt.xlabel("Coefficient value")
plt.title("Feature Coefficients from the Lasso Regression")

plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/Coefficient_from_Lasso_Regression.svg", bbox_inches='tight')
plt.savefig("/media/lunas/Elements/Results_for_Masterarbeit/Coefficient_from_Lasso_Regression.png", bbox_inches='tight')
#%%
