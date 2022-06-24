#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 08:49:23 2022

@author: lunas
"""

"""
vae with labeled data 
"""
#this script builds, trains and evaluates an AE.
#It also creates a feature vector from the AE latent and uses it for PCA, t-SNE, UMAP, RF and lasso regression.
#Further, it combines the AE and CV features for the same evaluations
#%%
#labeled
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
import torch
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
#import helper
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch import optim
import random
import torch.nn.functional as F
import torch.utils.data as data_utils
from skimage.metrics import structural_similarity as ssim
import skimage.io
from skimage.color import rgb2hed, hed2rgb
from skimage.util import crop
from skimage import data, io
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
import umap
#import hdbscan
import sklearn.cluster as cluster
#import pytorch_ssim
startTime = datetime.now()                                                                                                                                                                                                                                      
#import pytorch_ssim                                                                                                                                                                                                                                                           
#random.seed(42)
#torch.manual_seed(42)
#np.random.seed(42)
#torch.cuda.manual_seed(42)
#torch.cuda.manual_seed_all(42)

torch.cuda.is_available()

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 

print(dev)
#/media/lunas/Samsung USB/data/Single_cell_images_for_AE/try_out/labels.csv 

#%%
import torch
from skimage.io import imread
from torch.utils import data
import csv
import pandas as pd
from skimage import io, transform
from skimage.transform import rescale, resize, downscale_local_mean
from PIL import Image
#pip install pytorch-msssim
#https://pypi.org/project/pytorch-msssim/

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

transform = transforms.Compose([#transforms.Grayscale(num_output_channels=1),
                                transforms.Resize([32, 32]),
                                transforms.ToTensor()])
#transform = transforms.Compose([transforms.ToTensor()])



class MS_SSIM_Loss(MS_SSIM):
    def forward(self, img1, img2):
        return 100*( 1 - super(MS_SSIM_Loss, self).forward(img1, img2) )

class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 100*( 1 - super(SSIM_Loss, self).forward(img1, img2) )

criterion = SSIM_Loss(data_range=1.0, size_average=True, channel=3)


class SegmentationDataSet(data.Dataset):
    def __init__(self,
                 inputs,
                 transform=transform
                 ):
        self.inputs = pd.read_csv(inputs)
        self.transform = transform
        self.inputs_dtype = torch.float32
        #self.target = pd.read_csv("/media/lunas/Elements/data_ma/Single_cells_masked_aligned_all_cells_compare_to_features/labels_of_cell_for_AE_compare_to_features_all_cells_only_labels_cleaned.csv")
        self.target = pd.read_csv("/projects/ag-bozek/lhahn2/data/Single_cells_masked_aligned_all_cells_compare_to_features/labels_of_cell_for_AE_compare_to_features_all_cells_for_hpc_cleaned_only_labels.csv")
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x=self.inputs.iloc[idx, 0]
        #path = "/media/lunas/Elements/data_ma/Single_cells_masked_aligned_labeled/"
        # Select the sample
        image_path =  x
        cell_class = self.inputs.iloc[idx, 1]
        #y = self.target[index]

        # Load input and target
        x = Image.open(image_path)#io.imread(image_path)
        out = transform(x)
        x = transform(x)
       
        x = np.array(x)
        x = np.moveaxis(x, 0, 2)
        ihc_hed = rgb2hed(x)
        #h_channel = ihc_hed[:, :, 0]
        #h_channel = np.expand_dims(h_channel, -1)
        #h_channel = np.moveaxis(h_channel, 2, 0)
        #h_channel = torch.from_numpy(h_channel)
        #x = torch.from_numpy(ihc_e)
        #h_channel = h_channel.float()
        #h_channel = h_channel.unsqueeze(-1)
        #e_channel = ihc_hed[:, :, 1]
        #gray = cv2.cvtColor(np.float32(x), cv2.COLOR_RGB2GRAY)
        null = np.zeros_like(ihc_hed[:, :, 0])
        ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
        ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
        ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))
        ihc_h = np.moveaxis(ihc_h, 2, 0)
        ihc_h = torch.from_numpy(ihc_h)
        ihc_h = ihc_h.float()
        ihc_e = np.moveaxis(ihc_e, 2, 0)
        ihc_e = torch.from_numpy(ihc_e)
        ihc_e = ihc_e.float()
        #ihc_h = np.moveaxis(ihc_h, 2, 0)
        #ihc_e = np.moveaxis(ihc_e, 2, 0)
        #x = torch.from_numpy(ihc_h)
        #x = torch.from_numpy(ihc_e)
        #x = x.float()
   
        

        return out, cell_class, image_path #, e_channel
    
#%%
#import needed dfs
features_df = pd.read_csv('/projects/ag-bozek/lhahn2/data/Feature_gathering_from_og_contours_with_interpolate_pca_from_labeled_with_hed_channel.csv', sep=',')  

ae_df = pd.read_csv('/projects/ag-bozek/lhahn2/data/Single_cells_masked_aligned_labeled_compare_to_features/labled_data_out_to_compare_ae_and_features_for_hpc_cleaned.csv', sep=';')  

#%%
#load single cell images
inputs="/projects/ag-bozek/lhahn2/data/Single_cells_masked_aligned_labeled_compare_to_features/labels_of_cell_for_AE_compare_to_features_for_hpc_cleaned.csv"

ts="/projects/ag-bozek/lhahn2/data/Single_cells_masked_aligned_labeled/Single_cells_masked_aligned_labeled_for_zip/labels_of_cell_for_AE_cleaned_for_cheops.csv"

dataset =  SegmentationDataSet(inputs=inputs)

indices = torch.randperm(len(dataset))[:7100]
#indices = torch.randperm(len(dataset))[:65000]
#indices = torch.randperm(len(dataset))[:100]
dataset = data_utils.Subset(dataset, indices)



#%%
#split dataset and create data loaders
train_split = 0.8
test_split = 0.2
val_split = 0.2
batchsize = 2

dataset_size = len(dataset) #200
val_size = int(val_split * dataset_size)
train_size = dataset_size - val_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset,[train_size, val_size])
train_dataset, test_dataset = torch.utils.data.random_split(train_dataset,[int(len(train_dataset)*0.8), int(len(train_dataset)*0.2)])


train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(dataset=val_dataset, batch_size=batchsize, shuffle=True, pin_memory=True, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False, pin_memory=True, num_workers=4)
test_loader2 = DataLoader(dataset=test_dataset, batch_size=batchsize, shuffle=False, pin_memory=True, num_workers=4)

#%%
#define Autoencoder
latent_variable_size=32
class Autoencoder(nn.Module):
    def __init__(self, ndf = 64, ngf = 64, latent_variable_size=32, Only_Decode = False):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        #Only_Decode = self.Only_Decode
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), 
            nn.LeakyReLU(), # [batch, 12, 16, 16]
            nn.Conv2d(32, 64, 4, stride=2, padding=1), 
            nn.LeakyReLU(),# [batch, 24, 8, 8]
        	#nn.Conv2d(64, 64, 3, stride=1, padding=1), 
            #nn.LeakyReLU(), # [batch, 48, 4, 4]
        )       
        self.fc1 = nn.Linear(64*8*8, latent_variable_size)       
        self.decoder = nn.Sequential(
            #nn.ConvTranspose2d(64, 64, 3, stride=1 padding=1),  # [batch, 48, 4, 4]
            #nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, output_padding=0),  # [batch, 48, 4, 4]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, output_padding=0),  # [batch, 48, 4, 4]
            nn.Sigmoid(),
        ) 
        self.d1 = nn.Linear(latent_variable_size, 64*8*8)
        
    def forward(self, x, Only_Decode):
        
        if Only_Decode == False:
            #print(x.shape)
            encoded = self.encoder(x)
            #print(encoded.shape)
            encoded = encoded.view(-1, 64*8*8)
            encoded = self.fc1(encoded)
            decoded = self.d1(encoded)
            decoded = decoded.view(-1, 64, 8, 8)
            decoded = self.decoder(decoded)
            #decoded = self.decoder(encoded)
            return encoded, decoded
        
        else:
            decoded = self.d1(x)
            decoded = decoded.view(-1, 64, 8, 8)
            decoded = self.decoder(decoded)
            #decoded = self.decoder(encoded)
            return decoded
            
    
#%%
#define hyperparameters
Only_Decode = False
model = Autoencoder()
model = model.to(dev)
lr = 1e-3
# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# mean-squared error loss
criterion = nn.MSELoss()
#criterion = SSIM_Loss(data_range=1.0, size_average=True, channel=3)

epochs = 500
print("epochs:", epochs)
print("learning rate:", lr)
print("loss:", criterion)
print("batchsize:", batchsize)
print("data used:", len(dataset))
print("scheduler:", scheduler)
print("model:", model)
#%%
#train and evaluate the model
train_losses = []
list_of_labels = []
val_losses = []
train_images_og = []
train_images_decoded = []
val_images_og = []
val_images_decoded = []
count = 0
encoded_numpy = 0
for epoch in range(epochs):
    #count = count + 1
    loss_train = 0
    loss_val = 0
    train_images_og = []
    train_images_decoded = []
    val_images_og = []
    val_images_decoded = []
    #for n, (x,y) in enumerate(zip(train_loader, val_loader)):
    for n,(x,y) in enumerate(zip(train_loader, train_loader)):   #hier ist zweiter train loader eigetnlich val loader
        # reshape mini-batch data to [N, 784] matrix
        # load it to the active device
        #batch_features = batch_features.view(-1, 750)
        labelx = x[1]
        list_of_labels.append(labelx)
        x = x[0].to(dev)
        y = y[0].to(dev)
        #print(x.shape)
        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()
        #print(x[1])
        # compute reconstructions
        #print(x[0].shape, y[0].shape)

        encodedx, outputsx = model(x, Only_Decode)
        encodedy, outputsy = model(y, Only_Decode)
        
        #to get encoded
        if count == 0:
            encoded_numpy = encodedx.cpu().detach().numpy()
            count = count + 1
            #print("hi")
        else:
            encoded_numpy = np.concatenate((encoded_numpy, encodedx.cpu().detach().numpy()), axis=0)

        train_loss = criterion(outputsx, x)
        val_loss = criterion(outputsy, y)
        train_loss.backward()
        
        # perform parameter update based on current gradients
        optimizer.step()
        
        # add the mini-batch training loss to epoch loss
        loss_train += train_loss.item()
        loss_val += val_loss.item()
        
        #save images
        train_images_og.append(x[0])
        train_images_decoded.append(outputsx[0].detach())
#        val_images_og.append(y[0])
#        val_images_decoded.append(outputsy[0].detach())
    # compute the epoch training loss
    loss_train = loss_train / len(train_loader)
    train_losses.append(loss_train)
    loss_val = loss_val / len(val_loader)
    val_losses.append(loss_val)
    
   
    # display the epoch training loss
    print("epoch : {}/{}, train_loss  = {:.6f}".format(epoch + 1, epochs, loss_train))    
    print("epoch : {}/{}, val_loss = {:.6f}".format(epoch + 1, epochs, loss_val)) 
#    print("encoded and output shapes:", encodedx.shape, outputsx.shape)
#%%
#create latent and list of labels for latent (for train)
flat_list_train = [item for sublist in list_of_labels for item in sublist]
encoded_numpy_train = torch.flatten(torch.tensor(encoded_numpy) , start_dim=1)
#encoded_numpy = f.cpu().detach().numpy()
#print("shape of latent:", encoded_numpy_train.shape, len(flat_list_train))
#%%
#test trained ae
list_for_pd = []
list_of_labels = []
for x in test_loader:
    image_path = x[2]
    labelx = x[1]
    entry = [image_path, labelx]
    list_for_pd.append(entry)
    list_of_labels.append(labelx)
    x = x[0].to(dev)
    print(x.shape)
    encodedx, outputsx = model(x, Only_Decode)
    encoded_numpy = encodedx.cpu().detach().numpy()
    test_loss = criterion(outputsx, x)
    test_images_og = x#[0]
    test_images_decoded = outputsx.detach()
    print("test_loss", test_loss.item())
    #print(test_images_og.shape, test_images_decoded.shape)
    
test_losses = []
for i in range(0,len(train_losses)):
    test_losses.append(test_loss.item())

aedf = pd.DataFrame(list_for_pd)
print(aedf.head())
#%%
#change on dim of latent at a time for the average latent
mean_encodedx = torch.mean(encodedx, 0)
Only_Decode = True
#zeros = torch.zeros(200, 32)
#zeros = zeros.to(dev)
decode_this = mean_encodedx.to(dev)
for i in range(0,32):#zeros.shape[1]): 
    f = plt.figure()
    count = 0
    for e in [-20, -10, -5, -2, -1, -0.5, 1, 1.5, 2, 3, 6, 11, 21]:
        count = count + 1
        #print(i, e)
        decode_this[i] = e*decode_this[i]
        x = model(decode_this, Only_Decode)
        x = x.detach()[0]
        #print(x.shape)
        f.add_subplot(1,13, count)
        hi = x.cpu().permute(1, 2, 0).numpy()
        plt.imshow((hi* 255).astype('uint8'))
        decode_this[i] = decode_this[i]/e
        plt.axis('off')
        plt.savefig('/projects/ag-bozek/lhahn2/plots/Deep_CAE_Averagelatent_decoded_dim' + str(i) + '2.png')
        plt.savefig('/projects/ag-bozek/lhahn2/plots/Deep_CAE_Averagelatent_decoded_dim' + str(i) + '2.svg')

#%%
#create AE features from test latent and combine them with the CV features and the labels for the test cells
flat_list_test = [item for sublist in list_of_labels for item in sublist]
f = torch.flatten(encodedx, start_dim=1)
encoded_numpy_test = f.cpu().detach().numpy()
print("shape of latent:", encoded_numpy_test.shape)



print("encoded_numpy_test", type(encoded_numpy_test), len(encoded_numpy_test))
encoded_df = pd.DataFrame(encoded_numpy_test)

#print("\n##########################################################\n")
df = pd.DataFrame(list(zip(image_path, labelx)))
print(df.shape)
print(df.head())
#print(encoded_df.head())

result = pd.concat([df, encoded_df], axis=1, ignore_index=True)
result.columns=["R"+str(i) for i in range(1, len(result.columns) + 1)]
print("result of ae", result.shape)
#print(result.head())


features_df.columns=["F"+str(i) for i in range(1, len(features_df.columns) + 1)]
features_df["merge"] = features_df["F1"].astype(str) + features_df["F2"].astype(str)
print("features", features_df.shape)
print(features_df.head())
#ae_df = pd.read_csv('/projects/ag-bozek/lhahn2/data/Single_cells_masked_aligned_labeled_compare_to_features/labled_data_out_to_compare_ae_and_features_for_hpc_cleaned.csv', sep=';')  
#ae_df = pd.read_csv("/media/lunas/Elements/data_ma/Single_cells_masked_aligned_labeled_compare_to_features/labled_data_out_to_compare_ae_and_features.csv", sep=';')
ae_df.columns=["A"+str(i) for i in range(1, len(ae_df.columns) + 1)]
ae_df["merge"] = ae_df["A1"].astype(str) + ae_df["A2"].astype(str)
print("ae_df_shape", ae_df.shape)
#print("ae_df")
print(ae_df.head())

result=result.rename(columns = {'R1':'merge_here'})
ae_df=ae_df.rename(columns = {'A5':'merge_here'})
result_merged = result.merge(ae_df, how='inner')
#result.merge(ae_df,left_on="R1", right_on="A5")
print("result_merged shape ",result_merged.shape)


result_merged2 = result_merged.merge(features_df, how='inner')
print("shape_merged2",result_merged2.shape)
#%%
print(result_merged2.columns)
print(result_merged2.head())
###bis hier klappt
#%%
result_merged2.to_csv("/projects/ag-bozek/lhahn2/data/Single_cells_masked_aligned_labeled_compare_to_features/features_and_latent_space2.csv", sep=';')

#features = result_merged2.iloc[:, np.r_[0:514, 522:538]]
features = result_merged2.iloc[:, np.r_[0:34, 42:62]]
features.to_csv("/projects/ag-bozek/lhahn2/data/Single_cells_masked_aligned_labeled_compare_to_features/features_and_latent_space2.csv", sep=';')
print("features column names", features.columns)
print("features head ", features.head())

#%%
#ANALYSIS of AE + CV features

#%%
features = pd.read_csv("/projects/ag-bozek/lhahn2/data/Single_cells_masked_aligned_labeled_compare_to_features/features_and_latent_space2.csv", sep=';')
print(features.columns)
#%%
#PCA
scaler = StandardScaler()
cell_features_scaled = scaler.fit_transform(features.iloc[:, 3:])
cell_features_scaled = np.nan_to_num(cell_features_scaled)
pd_cell_features_scaled = pd.DataFrame(cell_features_scaled, columns = features.iloc[:, 3:].columns)
#%%
#PCA
data = []
pca = PCA(n_components=3)
pca_result = pca.fit_transform(cell_features_scaled)
features["1pca_from_features_tumor_vs_non_tumor"] = pca_result[:,0]
features["2pca_from_features_tumor_vs_non_tumor"] = pca_result[:,1]

# get column for tumor vs non tumor
features["Tumor_vs_Non_Tumor"] = features["R2"]
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

features.Tumor_vs_Non_Tumor = [mapping[item] for item in features.Tumor_vs_Non_Tumor]
print("Verteilung fuer tumor vs non tumor", features.groupby('Tumor_vs_Non_Tumor').count())


#print(len(pca_result[:,0]))
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
print(pca.components_)
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])

fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
#scatter = ax.scatter(pca_result[:,0], pca_result[:,1])# c=flat_list )
tum = plt.scatter(features[features["R2"] == "Tumor"]["1pca_from_features_tumor_vs_non_tumor"], features[features["R2"] == "Tumor"]["2pca_from_features_tumor_vs_non_tumor"], color="red", label='Tumor', facecolors='none')
#tum = plt.scatter(df[df["labels"] != "Tumor"]["pc1"], df[df["labels"] != "Tumor"]["pc2"], color="blue", label='Non_Tumor', facecolors='none')
tum = plt.scatter(features[features["R2"] == "Stroma"]["1pca_from_features_tumor_vs_non_tumor"], features[features["R2"] == "Stroma"]["2pca_from_features_tumor_vs_non_tumor"], color="green", label='Stroma', facecolors='none')
tum = plt.scatter(features[features["R2"] == "normal small lymphocyte"]["1pca_from_features_tumor_vs_non_tumor"], features[features["R2"] == "normal small lymphocyte"]["2pca_from_features_tumor_vs_non_tumor"], color="blue", label='normal small lymphocyte', facecolors='none')
tum = plt.scatter(features[features["R2"] == "large leucocyte"]["1pca_from_features_tumor_vs_non_tumor"], features[features["R2"] == "large leucocyte"]["2pca_from_features_tumor_vs_non_tumor"], color="orange", label='large leucocyte', facecolors='none')
tum = plt.scatter(features[features["R2"] == "Epithelial cell"]["1pca_from_features_tumor_vs_non_tumor"], features[features["R2"] == "Epithelial cell"]["2pca_from_features_tumor_vs_non_tumor"], color="purple", label='Epithelial cell', facecolors='none')

plt.legend()





#legend1 = ax.legend(*scatter.legend_elements(num=5),
#                  loc="upper right", title="Classes")
plt.savefig('/projects/ag-bozek/lhahn2/plots/' + "AEandfeatures32TumorvsNontumor" + '_PCA_labeled_cells_from_AE_and_features2.png')
plt.savefig('/projects/ag-bozek/lhahn2/plots/' + "AEandfeatures32TumorvsNontumor" + '_PCA_labeled_cells_from_AE_and_features2.svg')
#%%
#t-SNE

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
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])

tum = plt.scatter(features[features["R2"] == "Tumor"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["R2"] == "Tumor"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="red", label='Tumor', facecolors='none')
#tum = plt.scatter(df[df["labels"] != "Tumor"]["pc1"], df[df["labels"] != "Tumor"]["pc2"], color="blue", label='Non_Tumor', facecolors='none')
tum = plt.scatter(features[features["R2"] == "Stroma"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["R2"] == "Stroma"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="green", label='Stroma', facecolors='none')
tum = plt.scatter(features[features["R2"] == "normal small lymphocyte"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["R2"] == "normal small lymphocyte"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="blue", label='normal small lymphocyte', facecolors='none')
tum = plt.scatter(features[features["R2"] == "large leucocyte"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["R2"] == "large leucocyte"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="orange", label='large leucocyte', facecolors='none')
tum = plt.scatter(features[features["R2"] == "Epithelial cell"]["1tsne_dim_from_features_tumor_vs_non_tumor"], features[features["R2"] == "Epithelial cell"]["2tsne_dim_from_features_tumor_vs_non_tumor"], color="purple", label='Epithelial cell', facecolors='none')

plt.legend()


plt.savefig('/projects/ag-bozek/lhahn2/plots/' + "AEandfeatures32TumorvsNontumor" + 'TSNE_labeled_cells_from_AE_and_features2.png')
plt.savefig('/projects/ag-bozek/lhahn2/plots/' + "AEandfeatures32TumorvsNontumor" + 'TSNE_labeled_cells_from_AE_and_features2.svg')
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
).fit_transform(cell_features_scaled)



#%%
features["1umap_dim_from_features_tumor_vs_non_tumor"] = clusterable_embedding[:,0]
features["2umap_dim_from_features_tumor_vs_non_tumor"] = clusterable_embedding[:,1]
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])

tum = plt.scatter(features[features["R2"] == "Tumor"]["1umap_dim_from_features_tumor_vs_non_tumor"], features[features["R2"] == "Tumor"]["2umap_dim_from_features_tumor_vs_non_tumor"], color="red", label='Tumor', facecolors='none')
#tum = plt.scatter(df[df["labels"] != "Tumor"]["pc1"], df[df["labels"] != "Tumor"]["pc2"], color="blue", label='Non_Tumor', facecolors='none')
tum = plt.scatter(features[features["R2"] == "Stroma"]["1umap_dim_from_features_tumor_vs_non_tumor"], features[features["R2"] == "Stroma"]["2umap_dim_from_features_tumor_vs_non_tumor"], color="green", label='Stroma', facecolors='none')
tum = plt.scatter(features[features["R2"] == "normal small lymphocyte"]["1umap_dim_from_features_tumor_vs_non_tumor"], features[features["R2"] == "normal small lymphocyte"]["2umap_dim_from_features_tumor_vs_non_tumor"], color="blue", label='normal small lymphocyte', facecolors='none')
tum = plt.scatter(features[features["R2"] == "large leucocyte"]["1umap_dim_from_features_tumor_vs_non_tumor"], features[features["R2"] == "large leucocyte"]["2umap_dim_from_features_tumor_vs_non_tumor"], color="orange", label='large leucocyte', facecolors='none')
tum = plt.scatter(features[features["R2"] == "Epithelial cell"]["1umap_dim_from_features_tumor_vs_non_tumor"], features[features["R2"] == "Epithelial cell"]["2umap_dim_from_features_tumor_vs_non_tumor"], color="purple", label='Epithelial cell', facecolors='none')

plt.legend()

plt.savefig('/projects/ag-bozek/lhahn2/plots/' + "AEandfeatures32TumorvsNontumor" + 'UMAP_labeled_cells_from_AE_and_features2.png')
plt.savefig('/projects/ag-bozek/lhahn2/plots/' + "AEandfeatures32TumorvsNontumor" + 'UMAP_labeled_cells_from_AE_and_features2.svg')
#%%
#RF AND LASSO for AE + CV features
#print("USE ONLY FEATURES FOR RF AND LOGREG")
print("USE AE AND CV FEATURES FOR RF AND LASSO-REGRESSION")
import numpy as np
import pandas as pd
#from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
#import shap
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
print("Tumor_vs_Non_Tumor", features["Tumor_vs_Non_Tumor"].value_counts())
print("R2", features["R2"].value_counts())
train_features, test_features, train_labels, test_labels = train_test_split(pd_cell_features_scaled , features["R2"], test_size = 0.2, random_state = 42, stratify=features["R2"])


print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
#%%
rf = RandomForestClassifier(random_state = 42, n_estimators=64, min_samples_split=10, min_samples_leaf=2, max_features="sqrt", max_depth=4, bootstrap=False)
print(rf.get_params())
rf.fit(train_features , train_labels)
y_pred_test = rf.predict(train_features)
#print(y_pred_test)
print("test_labels value counts", test_labels.value_counts())
print("Random Forest: accuracy score for train", accuracy_score(train_labels , y_pred_test))
#print(accuracy_score)
y_pred_test = rf.predict(test_features)
print("Random Forest: accuracy score for test", accuracy_score(test_labels , y_pred_test))
#print(accuracy_score)
#%%
#feature importance
sorted_idx = rf.feature_importances_.argsort()
ar = rf.feature_importances_
ar = pd.DataFrame(ar.reshape(1,-1))
liste = list(train_features.columns)
ar.columns = liste
ar = ar.sort_values(by =0, axis=1, ascending=False)
ar.to_csv("/projects/ag-bozek/lhahn2/plots/rf_feature_importance_using_ae-and-features2.csv", sep=';')


#%%
perm_importance1 = permutation_importance(rf, test_features, test_labels)
#%%
permimportances = perm_importance1['importances_mean']
#%%
perm_importance = pd.DataFrame(perm_importance1['importances_mean'].reshape(1,-1))
liste = list(train_features.columns)
perm_importance.columns = liste
perm_importance = perm_importance.sort_values(by =0, axis=1, ascending=False)
perm_importance.to_csv("/projects/ag-bozek/lhahn2/plots/rf_permutated_feature_importance_using_ae-and-features2.csv", sep=';')

#%%
#LASSO REGRESSION
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0, penalty='l2',solver='liblinear').fit(train_features , train_labels)
print("Logistic Regression: mean accuracy for train", clf.score(train_features, train_labels))
print("Logistic Regression: mean accuracy for test", clf.score(test_features, test_labels))
#%%
#tumor vs non tumor
print("tumor vs non-tumor")
train_features, test_features, train_labels, test_labels = train_test_split(pd_cell_features_scaled , features["Tumor_vs_Non_Tumor"], test_size = 0.2, random_state = 42, stratify=features["Tumor_vs_Non_Tumor"])


print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
#%%
rf = RandomForestClassifier(random_state = 42, n_estimators=64, min_samples_split=10, min_samples_leaf=2, max_features="sqrt", max_depth=4, bootstrap=False)
print(rf.get_params())
rf.fit(train_features , train_labels)
y_pred_test = rf.predict(train_features)
#print(y_pred_test)
print("test_labels value counts", test_labels.value_counts())
print("Random Forest: accuracy score for train", accuracy_score(train_labels , y_pred_test))
#print(accuracy_score)
y_pred_test = rf.predict(test_features)
print("Random Forest: accuracy score for test", accuracy_score(test_labels , y_pred_test))
#print(accuracy_score)
#%%
#feature importance
sorted_idx = rf.feature_importances_.argsort()
ar = rf.feature_importances_
ar = pd.DataFrame(ar.reshape(1,-1))
liste = list(train_features.columns)
ar.columns = liste
ar = ar.sort_values(by =0, axis=1, ascending=False)
ar.to_csv("/projects/ag-bozek/lhahn2/plots/rf_feature_importance_using_ae-and-features2.csv", sep=';')


#%%
perm_importance1 = permutation_importance(rf, test_features, test_labels)
#%%
permimportances = perm_importance1['importances_mean']
#%%
perm_importance = pd.DataFrame(perm_importance1['importances_mean'].reshape(1,-1))
liste = list(train_features.columns)
perm_importance.columns = liste
perm_importance = perm_importance.sort_values(by =0, axis=1, ascending=False)
perm_importance.to_csv("/projects/ag-bozek/lhahn2/plots/rf_permutated_feature_importance_using_ae-and-features2.csv", sep=';')

#%%
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0, penalty='l2',solver='liblinear').fit(train_features , train_labels)
print("Logistic Regression: mean accuracy for train", clf.score(train_features, train_labels))
print("Logistic Regression: mean accuracy for test", clf.score(test_features, test_labels))


#############################################
##############################################
#%%
#Same analysis but for the ae only

#Image reconstruction
print("ONLY AE")
f = plt.figure()
count = 0
count1 = 9 
for og, decode in zip(test_images_og, test_images_decoded):
    #print(og.shape, decode.shape)
    ##print(f, b)
    count = count + 1
    f.add_subplot(2,9, count)
    plt.imshow(og.cpu().permute(1, 2, 0))
    #plt.title("Train_OG_" + str(count))
    plt.axis('off')
    count1 = count1 + 1
    f.add_subplot(2,9, count1)
    plt.imshow(decode.cpu().permute(1, 2, 0))
    #plt.title("Train_Decoded_" + str(count))
    plt.axis('off')
    if count > 8:
        break
plt.savefig('/projects/ag-bozek/lhahn2/plots/' + str(latent_variable_size) + 'Test_images_unlabeled_loss2.png')
plt.savefig('/projects/ag-bozek/lhahn2/plots/' + str(latent_variable_size) + 'Test_images_unlabeled_loss2.svg')
#plt.savefig('/home/lunas/Desktop/Test_images_unlabeled_loss.png')      
#%%
#Loss for AE
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.title("Loss for training the Autoencoder")
plt.plot(val_losses,label="val")
plt.plot(train_losses,label="train")
plt.plot(test_losses,label="test")
plt.xlabel("epoch")
plt.ylabel("MSE-Loss")
plt.legend()
#plt.savefig('/home/lunas/Desktop/CAE_unlabeled_loss.png')
plt.savefig('/projects/ag-bozek/lhahn2/plots/'+ str(latent_variable_size) +'CAE_unlabeled_loss2.png')
plt.savefig('/projects/ag-bozek/lhahn2/plots/'+ str(latent_variable_size) +'CAE_unlabeled_loss2.svg')
#plt.show()


#%%
#PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
data = []
pca = PCA(n_components=3)
pca_result = pca.fit_transform(encoded_numpy_test)
first = pca_result[:,0]
second = pca_result[:,1]

flat_list_test_tumor_non_tumor = []
df = pd.DataFrame(first)
df[1] = second
df[2] = flat_list_test
for i in flat_list_test:
    if i == "Tumor":
        flat_list_test_tumor_non_tumor.append("Tumor")
    else:
        flat_list_test_tumor_non_tumor.append("Non_Tumor")
df[3] = flat_list_test_tumor_non_tumor
df.columns = ["pc1","pc2","labels", "labels_tumor_vs_non_tumor"]
#print(df)
#print(type(first))
#print(len(pca_result[:,0]))
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
#print(pca.components_)
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
#scatter = ax.scatter(pca_result[:,0], pca_result[:,1])# c=flat_list )
tum = plt.scatter(df[df["labels"] == "Tumor"]["pc1"], df[df["labels"] == "Tumor"]["pc2"], color="red", label='Tumor', facecolors='none')
#tum = plt.scatter(df[df["labels"] != "Tumor"]["pc1"], df[df["labels"] != "Tumor"]["pc2"], color="blue", label='Non_Tumor', facecolors='none')
tum = plt.scatter(df[df["labels"] == "Stroma"]["pc1"], df[df["labels"] == "Stroma"]["pc2"], color="green", label='Stroma', facecolors='none')
tum = plt.scatter(df[df["labels"] == "normal small lymphocyte"]["pc1"], df[df["labels"] == "normal small lymphocyte"]["pc2"], color="blue", label='normal small lymphocyte', facecolors='none')
tum = plt.scatter(df[df["labels"] == "large leucocyte"]["pc1"], df[df["labels"] == "large leucocyte"]["pc2"], color="orange", label='large leucocyte', facecolors='none')
tum = plt.scatter(df[df["labels"] == "Epithelial cell"]["pc1"], df[df["labels"] == "Epithelial cell"]["pc2"], color="purple", label='Epithelial cell', facecolors='none')
#tum = plt.scatter(df[df["labels"] == "Positive"]["pc1"], df[df["labels"] == "Positive"]["pc2"], color="yellow", label='Positive', facecolors='none')
plt.legend()
plt.savefig('/projects/ag-bozek/lhahn2/plots/'+ str(latent_variable_size) +'PCA_labeled_cells_from_VAE2.png')
plt.savefig('/projects/ag-bozek/lhahn2/plots/'+ str(latent_variable_size) +'PCA_labeled_cells_from_VAE2.svg')
#%%   
#t-SNE
from sklearn.manifold import TSNE
import time

time_start = time.time()

tsne = TSNE(n_components=2, verbose=1, perplexity=100, n_iter=300)
#tsne_results = tsne.fit_transform(six_pcas_as_features)
tsne_results = tsne.fit_transform(encoded_numpy_test)
#tsne_results = tsne.fit_transform(np.nan_to_num(features.iloc[:, 3:17]))
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
#%%
first = tsne_results[:,0]
second = tsne_results[:,1]

df = pd.DataFrame(first)
df[1] = second
df[2] = flat_list_test
df.columns = ["dim1","dim2","labels"]

#print(pca.components_)
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
#scatter = ax.scatter(pca_result[:,0], pca_result[:,1])# c=flat_list )
tum = plt.scatter(df[df["labels"] == "Tumor"]["dim1"], df[df["labels"] == "Tumor"]["dim2"], color="red", label='Tumor', facecolors='none')
#tum = plt.scatter(df[df["labels"] != "Tumor"]["dim1"], df[df["labels"] != "Tumor"]["dim2"], color="blue", label='Non_Tumor', facecolors='none')
tum = plt.scatter(df[df["labels"] == "Stroma"]["dim1"], df[df["labels"] == "Stroma"]["dim2"], color="green", label='Stroma', facecolors='none')
tum = plt.scatter(df[df["labels"] == "normal small lymphocyte"]["dim1"], df[df["labels"] == "normal small lymphocyte"]["dim2"], color="blue", label='normal small lymphocyte', facecolors='none')
tum = plt.scatter(df[df["labels"] == "large leucocyte"]["dim1"], df[df["labels"] == "large leucocyte"]["dim2"], color="orange", label='large leucocyte', facecolors='none')
tum = plt.scatter(df[df["labels"] == "Epithelial cell"]["dim1"], df[df["labels"] == "Epithelial cell"]["dim2"], color="purple", label='Epithelial cell', facecolors='none')
#tum = plt.scatter(df[df["labels"] == "Positive"]["dim1"], df[df["labels"] == "Positive"]["dim2"], color="yellow", label='Positive', facecolors='none')
plt.legend()
plt.savefig('/projects/ag-bozek/lhahn2/plots/'+ str(latent_variable_size) +'TSNE_labeled_cells_from_VAE2.png')
plt.savefig('/projects/ag-bozek/lhahn2/plots/'+ str(latent_variable_size) +'TSNE_labeled_cells_from_VAE2.svg')
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
    n_neighbors=60,
    min_dist=0.0,
    n_components=5,
    random_state=42,
).fit_transform(encoded_numpy_test)

#%%
first = clusterable_embedding[:,0]
second = clusterable_embedding[:,1]

df = pd.DataFrame(first)
df[1] = second
df[2] = flat_list_test
df.columns = ["dim1","dim2","labels"]

#print(pca.components_)
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
#scatter = ax.scatter(pca_result[:,0], pca_result[:,1])# c=flat_list )
tum = plt.scatter(df[df["labels"] == "Tumor"]["dim1"], df[df["labels"] == "Tumor"]["dim2"], color="red", label='Tumor', facecolors='none')
#tum = plt.scatter(df[df["labels"] != "Tumor"]["dim1"], df[df["labels"] != "Tumor"]["dim2"], color="blue", label='Non_Tumor', facecolors='none')
tum = plt.scatter(df[df["labels"] == "Stroma"]["dim1"], df[df["labels"] == "Stroma"]["dim2"], color="green", label='Stroma', facecolors='none')
tum = plt.scatter(df[df["labels"] == "normal small lymphocyte"]["dim1"], df[df["labels"] == "normal small lymphocyte"]["dim2"], color="blue", label='normal small lymphocyte', facecolors='none')
tum = plt.scatter(df[df["labels"] == "large leucocyte"]["dim1"], df[df["labels"] == "large leucocyte"]["dim2"], color="orange", label='large leucocyte', facecolors='none')#tum = plt.scatter(df[df["labels"] == "Epithelial cell"]["dim1"], df[df["labels"] == "Epithelial cell"]["dim2"], color="purple", label='Epithelial cell', facecolors='none')
#tum = plt.scatter(df[df["labels"] == "Positive"]["dim1"], df[df["labels"] == "Positive"]["dim2"], color="yellow", label='Positive', facecolors='none')
plt.legend()
plt.savefig('/projects/ag-bozek/lhahn2/plots/'+ str(latent_variable_size) +'UMAP_labeled_cells_from_VAE2.png')
plt.savefig('/projects/ag-bozek/lhahn2/plots/'+ str(latent_variable_size) +'UMAP_labeled_cells_from_VAE2.svg')
#%%
#use only ae for prediction
import numpy as np
import pandas as pd
#from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
#import shap
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from collections import Counter
#cell_features_scaled , features["R2"]
train_features, test_features, train_labels, test_labels = train_test_split(encoded_numpy_test , flat_list_test , test_size = 0.2, random_state = 42, stratify=flat_list_test)

#%%
#RF
print("rf_and_log_only_ae")
#print("test_labels value counts", test_labels.value_counts())
sample_list = ["Tumor", "Non_Tumor"]
#print("test labels value counts", Counter(sample_list))
print("count tumor", train_labels.count("Tumor"))
print("count non-tumor", train_labels.count("Non_Tumor"))

rf = RandomForestClassifier(random_state = 42, n_estimators=64, min_samples_split=10, min_samples_leaf=2, max_features="sqrt", max_depth=4, bootstrap=False)
print(rf.get_params())
rf.fit(train_features , train_labels)
y_pred_test = rf.predict(train_features)
#print(y_pred_test)
print("Random Forest: accuracy score for train", accuracy_score(train_labels , y_pred_test))
#print(accuracy_score)
y_pred_test = rf.predict(test_features)
print("Random Forest: accuracy score for test",accuracy_score(test_labels , y_pred_test))

#%%
#LASSO
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0, penalty='l2',solver='liblinear').fit(train_features , train_labels)
print("Logistic Regression: mean accuracy for train", clf.score(train_features, train_labels))
print("Logistic Regression: mean accuracy for test", clf.score(test_features, test_labels))



#%%
#only ae tumor vs non tumor
print("only ae tumor vs non tumor")


train_features, test_features, train_labels, test_labels = train_test_split(encoded_numpy_test , flat_list_test_tumor_non_tumor , test_size = 0.2, random_state = 42, stratify=flat_list_test_tumor_non_tumor)

#%%
print("rf_and_log_only_ae")
#print("test_labels value counts", test_labels.value_counts())
sample_list = ["Tumor", "Non_Tumor"]
#print("test labels value counts", Counter(sample_list))
print("count tumor", train_labels.count("Tumor"))
print("count non-tumor", train_labels.count("Non_Tumor"))

rf = RandomForestClassifier(random_state = 42, n_estimators=64, min_samples_split=10, min_samples_leaf=2, max_features="sqrt", max_depth=4, bootstrap=False)
print(rf.get_params())
rf.fit(train_features , train_labels)
y_pred_test = rf.predict(train_features)
#print(y_pred_test)
print("Random Forest: accuracy score for train", accuracy_score(train_labels , y_pred_test))
#print(accuracy_score)
y_pred_test = rf.predict(test_features)
print("Random Forest: accuracy score for test",accuracy_score(test_labels , y_pred_test))

#%%
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0, penalty='l2',solver='liblinear').fit(train_features , train_labels)
print("Logistic Regression: mean accuracy for train", clf.score(train_features, train_labels))
print("Logistic Regression: mean accuracy for test", clf.score(test_features, test_labels))
#%%
print("Time for script to run:", datetime.now() - startTime)




