#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 08:49:23 2022

@author: lunas
"""

"""
vae with labeled data 
"""
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
random.seed(1)
torch.manual_seed(1)
startTime = datetime.now()                                                                                                                                                                                                                                      
#import pytorch_ssim                                                                                                                                                                                                                                                           
random.seed(1)
torch.manual_seed(1)

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
        #self.targets_dtype = torch.long

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
#features_df = pd.read_csv('/projects/ag-bozek/lhahn2/data/Feature_data/Feature_gathering_from_og_contours_with_interpolate_pca_from_labeled.csv', sep=',')  

#use hed-channel
features_df = pd.read_csv('/projects/ag-bozek/lhahn2/data/Feature_gathering_from_og_contours_with_interpolate_pca_from_labeled_with_hed_channel.csv', sep=',')  

ae_df = pd.read_csv('/projects/ag-bozek/lhahn2/data/Single_cells_masked_aligned_labeled_compare_to_features/labled_data_out_to_compare_ae_and_features_for_hpc_cleaned.csv', sep=';')  
print(features_df.head())
print(ae_df.head())
#%%
inputs="/projects/ag-bozek/lhahn2/data/Single_cells_masked_aligned_labeled_compare_to_features/labels_of_cell_for_AE_compare_to_features_for_hpc_cleaned.csv"
#inputs="/media/lunas/Elements/data_ma/Single_cells_masked_aligned_labeled/labels_of_cell_for_AE.csv"
#inputs="/media/lunas/Elements/data_ma/Single_cells_masked_aligned_labeled/labels_of_cell_for_AE_cleaned.csv"
#inputs="D:/data_ma/Single_cells_masked_aligned_labeled/labels_of_cell_for_AE_cleaned.csv"
#inputs="/projects/ag-bozek/lhahn2/data/Single_cells_masked_aligned_labeled/Single_cells_masked_aligned_labeled_for_zip/labels_of_cell_for_AE_cleaned_for_cheops.csv"
dataset =  SegmentationDataSet(inputs=inputs)
indices = torch.randperm(len(dataset))[:7100]
#indices = torch.randperm(len(dataset))[:200]
#indices = torch.randperm(len(dataset))[:100]
dataset = data_utils.Subset(dataset, indices)
#%%
train_split = 0.8
test_split = 0.2
val_split = 0.2
batchsize = 2

dataset_size = len(dataset) #200
val_size = int(val_split * dataset_size)
train_size = dataset_size - val_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset,[train_size, val_size])
train_dataset, test_dataset = torch.utils.data.random_split(train_dataset,[int(len(train_dataset)*0.8), int(len(train_dataset)*0.2)])
#print(len(train_dataset))

train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(dataset=val_dataset, batch_size=batchsize, shuffle=True, pin_memory=True, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False, pin_memory=True, num_workers=4)
test_loader2 = DataLoader(dataset=test_dataset, batch_size=batchsize, shuffle=False, pin_memory=True, num_workers=4)
#%%
# class Autoencoder(nn.Module):
#     def __init__(self, ndf = 64, ngf = 64, latent_variable_size=128):
#         super(Autoencoder, self).__init__()
#         # Input size: [batch, 3, 32, 32]
#         # Output size: [batch, 3, 32, 32]
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 4, 3, stride=1, padding=0), 
#             nn.LeakyReLU(), # [batch, 12, 16, 16]
#             #nn.ReLU(),
#             #torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
#             nn.Conv2d(4, 8, 3, stride=1, padding=0), 
#             nn.LeakyReLU(),# [batch, 24, 8, 8]
#             #nn.ReLU(),
#  			nn.Conv2d(8, 16, 3, stride=1, padding=0), 
#             nn.LeakyReLU(), # [batch, 48, 4, 4]
#             #nn.ReLU(),
#  			nn.Conv2d(16, 32, 3, stride=1, padding=0),  
#             nn.LeakyReLU(),# [batch, 96, 2, 2]
#             nn.Conv2d(32, 64, 3, stride=1, padding=0),  
#             nn.LeakyReLU(),
#             #nn.Conv2d(64, 128, 3, stride=1, padding=0),  
#             #nn.LeakyReLU(),
#             #nn.Conv2d(128, 64, 3, stride=1, padding=0),  
#             #nn.LeakyReLU(),
#             nn.Conv2d(64, 32, 3, stride=1, padding=0),  
#             nn.LeakyReLU(),
#          	nn.Conv2d(32, 16, 3, stride=1, padding=0),           # [batch, 96, 2, 2]
#             nn.LeakyReLU(),
#             nn.Conv2d(16, 8, 3, stride=1, padding=0),           # [batch, 96, 2, 2]
#             nn.LeakyReLU(),
#             nn.Conv2d(8, 4, 3, stride=1, padding=0),           # [batch, 96, 2, 2]
#             nn.LeakyReLU(),
#             #nn.Conv2d(4, 3, 3, stride=1, padding=0),           # [batch, 96, 2, 2]
#             #nn.LeakyReLU()
#               #nn.ReLU(),
#         )       
#         self.fc1 = nn.Linear(131072, 1000)       
#         self.decoder = nn.Sequential(
#             #nn.ConvTranspose2d(192, 96, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
#             #nn.LeakyReLU(),
#             #nn.ReLU(),
#             #nn.ConvTranspose2d(3, 4, 3, stride=1, padding=0),  # [batch, 48, 4, 4]
#             #nn.LeakyReLU(),
#             #nn.ReLU(),
#  			nn.ConvTranspose2d(4, 8, 3, stride=1, padding=0),  # [batch, 24, 8, 8]
#             nn.LeakyReLU(),
#             #nn.ReLU(),
#  			nn.ConvTranspose2d(8, 16, 3, stride=1, padding=0),  # [batch, 12, 16, 16]
#             nn.LeakyReLU(),
#             #nn.ReLU(),
#             nn.ConvTranspose2d(16, 32, 3, stride=1, padding=0),   # [batch, 3, 32, 32]
#             nn.LeakyReLU(),
#             nn.ConvTranspose2d(32, 64, 3, stride=1, padding=0),   # [batch, 3, 32, 32]
#             nn.LeakyReLU(),
#             #nn.ConvTranspose2d(64, 128, 3, stride=1, padding=0),   # [batch, 3, 32, 32]
#             #nn.LeakyReLU(),
#             #nn.ConvTranspose2d(128, 64, 3, stride=1, padding=0),   # [batch, 3, 32, 32]
#             #nn.LeakyReLU(),
#             nn.ConvTranspose2d(64, 32, 3, stride=1, padding=0),   # [batch, 3, 32, 32]
#             nn.LeakyReLU(),
#             nn.ConvTranspose2d(32, 16, 3, stride=1, padding=0),   # [batch, 3, 32, 32]
#             nn.LeakyReLU(),
#             nn.ConvTranspose2d(16, 8, 3, stride=1, padding=0),   # [batch, 3, 32, 32]
#             nn.LeakyReLU(),
#             nn.ConvTranspose2d(8, 4, 3, stride=1, padding=0),   # [batch, 3, 32, 32]
#             nn.LeakyReLU(),
#             nn.ConvTranspose2d(4, 3, 3, stride=1, padding=0),   # [batch, 3, 32, 32]
#             nn.Sigmoid(),
#             #nn.Tanh()
#         ) 
#         self.d1 = nn.Linear(1000, 131072)
#     def forward(self, x):
#         #print(x.shape)
#         encoded = self.encoder(x)
#         #print(encoded.shape)
#         #encoded = encoded.view(-1, 512*16*16)
#         #encoded = self.fc1(encoded)
#         #decoded = self.d1(encoded)
#         #decoded = decoded.view(-1, 512, 16, 16)
#         #decoded = self.decoder(decoded)
#         decoded = self.decoder(encoded)
#         return encoded, decoded
#%%
latent_variable_size=128
class Autoencoder(nn.Module):
    def __init__(self, ndf = 64, ngf = 64, latent_variable_size=128):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            #nn.Conv2d(3, 1024, 3, stride=1, padding=0), 
            #nn.LeakyReLU(), # [batch, 12, 16, 16]
            #nn.Conv2d(1024, 512, 3, stride=1, padding=0), 
            #nn.LeakyReLU(),# [batch, 24, 8, 8]
            nn.Conv2d(3, 512, 3, stride=1, padding=0), 
            nn.LeakyReLU(), # [batch, 12, 16, 16]
            #nn.ReLU(),
            #nn.Conv2d(512, 512, 3, stride=1, padding=0), 
            #nn.LeakyReLU(),
            #nn.Conv2d(512, 512, 3, stride=1, padding=0), 
            #nn.LeakyReLU(),
            #torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.Conv2d(512, 256, 3, stride=1, padding=0), 
            nn.LeakyReLU(),# [batch, 24, 8, 8]
            #nn.ReLU(),
        		nn.Conv2d(256, 128, 3, stride=1, padding=0), 
            nn.LeakyReLU(), # [batch, 48, 4, 4]
 			nn.Conv2d(128, 64, 3, stride=1, padding=0), 
            nn.LeakyReLU(), # [batch, 48, 4, 4]
            #nn.ReLU(),
 			nn.Conv2d(64, 32, 3, stride=1, padding=0),  
            nn.LeakyReLU(),# [batch, 96, 2, 2]
            # nn.ReLU(),
         	nn.Conv2d(32, 16, 3, stride=1, padding=0),           # [batch, 96, 2, 2]
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, 3, stride=1, padding=0),           # [batch, 96, 2, 2]
            nn.LeakyReLU(),
            nn.Conv2d(8, 4, 3, stride=1, padding=0),           # [batch, 96, 2, 2]
            nn.LeakyReLU(),
            #nn.Conv2d(4, 2, 3, stride=1, padding=0),           # [batch, 96, 2, 2]
            #nn.LeakyReLU(),
            #nn.Conv2d(256, 512, 3, stride=1, padding=0),           # [batch, 96, 2, 2]
            #nn.LeakyReLU()
              #nn.ReLU(),
        )       
        self.fc1 = nn.Linear(1024, latent_variable_size)       
        self.decoder = nn.Sequential(
            #nn.ConvTranspose2d(2, 4, 3, stride=1, padding=0),  # [batch, 48, 4, 4]
            #nn.LeakyReLU(),
            nn.ConvTranspose2d(4, 8, 3, stride=1, padding=0),  # [batch, 48, 4, 4]
            nn.LeakyReLU(),
            #nn.ReLU(),
            nn.ConvTranspose2d(8, 16, 3, stride=1, padding=0),  # [batch, 48, 4, 4]
            nn.LeakyReLU(),
            #nn.ReLU(),
 			nn.ConvTranspose2d(16, 32, 3, stride=1, padding=0),  # [batch, 24, 8, 8]
            nn.LeakyReLU(),
            #nn.ReLU(),
 			nn.ConvTranspose2d(32, 64, 3, stride=1, padding=0),  # [batch, 12, 16, 16]
            nn.LeakyReLU(),
            #nn.ReLU(),
            nn.ConvTranspose2d(64, 128, 3, stride=1, padding=0),   # [batch, 3, 32, 32]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 256, 3, stride=1, padding=0),   # [batch, 3, 32, 32]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 512, 3, stride=1, padding=0, output_padding=0),   # [batch, 3, 32, 32]
            nn.LeakyReLU(),
           # nn.ConvTranspose2d(512, 512, 3, stride=1, padding=0, output_padding=0),   # [batch, 3, 32, 32]
           # nn.LeakyReLU(),
            #nn.ConvTranspose2d(512, 512, 3, stride=1, padding=0, output_padding=0),   # [batch, 3, 32, 32]
            #nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 3, 3, stride=1, padding=0, output_padding=0),   # [batch, 3, 32, 32]
            #nn.LeakyReLU(),
            #nn.ConvTranspose2d(512, 1024, 3, stride=1, padding=0, output_padding=0),   # [batch, 3, 32, 32]
            #nn.LeakyReLU(),
            #nn.ConvTranspose2d(8, 4, 3, stride=1, padding=0),   # [batch, 3, 32, 32]
            #nn.LeakyReLU(),
            #nn.ConvTranspose2d(1024, 3, 3, stride=1, padding=0),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
            #nn.Tanh()
        ) 
        self.d1 = nn.Linear(latent_variable_size, 1024)
    def forward(self, x):
        #print(x.shape)
        encoded = self.encoder(x)
        #print(encoded.shape)
        encoded = encoded.view(-1, 4*16*16)
        encoded = self.fc1(encoded)
        decoded = self.d1(encoded)
        decoded = decoded.view(-1, 4, 16, 16)
        decoded = self.decoder(decoded)
        #decoded = self.decoder(encoded)
        return encoded, decoded
#%%
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

epochs = 50
print("epochs:", epochs)
print("learning rate:", lr)
print("loss:", criterion)
print("batchsize:", batchsize)
print("data used:", len(indices))
print("scheduler:", scheduler)
print("model:", model)
#%%
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
    for n,(x,y) in enumerate(zip(train_loader, val_loader)):   
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

        encodedx, outputsx = model(x)
        encodedy, outputsy = model(y)
        
        #to get encoded
        if count == 0:
            encoded_numpy = encodedx.cpu().detach().numpy()
            count = count + 1
            #print("hi")
        else:
            encoded_numpy = np.concatenate((encoded_numpy, encodedx.cpu().detach().numpy()), axis=0)
        #labelx = x[1]
        #print(encodedx.shape, outputsx.shape)

        # compute training reconstruction loss
        train_loss = criterion(outputsx, x)
        val_loss = criterion(outputsy, y)
        #lol = np.moveaxis(outputsx.detach().numpy()[0], 0, 2)
        #lel = np.moveaxis(x[0].detach().numpy()[0], 0, 2)
        #print("ssim: 0", ssim(lol, lel, channel_axis = 2))
        #print(outputsx.detach().numpy().shape)
        #print(ssim(outputsx.detach().numpy(), x[0].detach().numpy()), channel_axis = 0)
        # compute accumulated gradients
        train_loss.backward()
        
        # perform parameter update based on current gradients
        optimizer.step()
        
        # add the mini-batch training loss to epoch loss
        loss_train += train_loss.item()
        loss_val += val_loss.item()
        
        #save images
        train_images_og.append(x[0])
        train_images_decoded.append(outputsx[0].detach())
        val_images_og.append(y[0])
        val_images_decoded.append(outputsy[0].detach())
    # compute the epoch training loss
    loss_train = loss_train / len(train_loader)
    train_losses.append(loss_train)
    loss_val = loss_val / len(val_loader)
    val_losses.append(loss_val)
    
   
    # display the epoch training loss
    print("epoch : {}/{}, train_loss  = {:.6f}".format(epoch + 1, epochs, loss_train))    
    print("epoch : {}/{}, val_loss = {:.6f}".format(epoch + 1, epochs, loss_val)) 
    print("encoded and output shapes:", encodedx.shape, outputsx.shape)
#%%
flat_list_train = [item for sublist in list_of_labels for item in sublist]
encoded_numpy_train = torch.flatten(torch.tensor(encoded_numpy) , start_dim=1)
#encoded_numpy = f.cpu().detach().numpy()
#print("shape of latent:", encoded_numpy_train.shape, len(flat_list_train))
#%%
##########################################################
############################################################
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
    encodedx, outputsx = model(x)
    encoded_numpy = encodedx.cpu().detach().numpy()
    test_loss = criterion(outputsx, x)
    test_images_og = x#[0]
    test_images_decoded = outputsx.detach()
    print("test_loss", test_loss.item())
    #print(test_images_og.shape, test_images_decoded.shape)
    
test_losses = []
for i in range(0,len(val_losses)):
    test_losses.append(test_loss.item())

aedf = pd.DataFrame(list_for_pd)
print(aedf.head())
#%%
#print(encodedx.shape)
#print(encoded_numpy.shape)
#print(len(list_of_labels))
flat_list_test = [item for sublist in list_of_labels for item in sublist]
f = torch.flatten(encodedx, start_dim=1)
encoded_numpy_test = f.cpu().detach().numpy()
print("shape of latent:", encoded_numpy_test.shape)
#print("/n##########################################################/n")
#print("image_path", type(image_path), len(image_path))
#print("labelx", type(labelx), len(labelx))
#print("encoded_numpy_test", type(encoded_numpy_test), len(encoded_numpy_test))
encoded_df = pd.DataFrame(encoded_numpy_test)

#print("\n##########################################################\n")
df = pd.DataFrame(list(zip(image_path, labelx)))
#print(df.head())
#print(encoded_df.head())

result = pd.concat([df, encoded_df], axis=1, ignore_index=True)
result.columns=["R"+str(i) for i in range(1, len(result.columns) + 1)]
#print("result")
#print(result.head())


features_df.columns=["F"+str(i) for i in range(1, len(features_df.columns) + 1)]
features_df["merge"] = features_df["F1"].astype(str) + features_df["F2"].astype(str)
#print("features")
#print(features_df.head())
ae_df = pd.read_csv('/projects/ag-bozek/lhahn2/data/Single_cells_masked_aligned_labeled_compare_to_features/labled_data_out_to_compare_ae_and_features_for_hpc_cleaned.csv', sep=';')  
ae_df.columns=["A"+str(i) for i in range(1, len(ae_df.columns) + 1)]
ae_df["merge"] = ae_df["A1"].astype(str) + ae_df["A2"].astype(str)
#print("ae_df")
#print(ae_df.head())

#for col in result.columns:
#    print(col)    
#for col in ae_df.columns:
#    print(col)

#print("shape", result.shape)
result=result.rename(columns = {'R1':'merge_here'})
ae_df=ae_df.rename(columns = {'A5':'merge_here'})
result_merged = result.merge(ae_df, how='inner')
#result.merge(ae_df,left_on="R1", right_on="A5")
#print("result shape",result_merged.shape)
#print("features_df shape",features_df.shape)
#print("1 and 2")
#print(result_merged.iloc[:,0:2].head())
#print(result_merged.iloc[:,2:4].head())
#print(result_merged.iloc[:,4:6].head())
#print(result_merged.iloc[:,6:8].head())
#print(result_merged.iloc[:,8:10].head())
#print(result_merged.iloc[:,514:519].head())

result_merged2 = result_merged.merge(features_df, how='inner')
#print("shape",result_merged2.shape)

result_merged2.to_csv("/projects/ag-bozek/lhahn2/data/Single_cells_masked_aligned_labeled_compare_to_features/features_and_latent_space.csv", sep=';')

#features = result_merged2.iloc[:, np.r_[0:514, 522:538]]
#features.to_csv("/projects/ag-bozek/lhahn2/data/Single_cells_masked_aligned_labeled_compare_to_features/features_and_latent_space.csv", sep=';')


#analysis
#%%
#use only features
#features = pd.read_csv("/home/lunas/Documents/Uni/Masterarbeit/Data/Feature_gathering_from_og_contours_with_interpolate_pca_from_labeled.csv")
#scaler = StandardScaler()
#print(features.iloc[:, 3:17].columns)
#cell_features_scaled = scaler.fit_transform(features.iloc[:, 3:17])
#cell_features_scaled = np.nan_to_num(cell_features_scaled)

#use ae data
#features = pd.read_csv("C:/Users/lunas/OneDrive/Desktop/features_and_latent_space.csv",  sep=';')
features = pd.read_csv("/projects/ag-bozek/lhahn2/data/Single_cells_masked_aligned_labeled_compare_to_features/features_and_latent_space.csv", sep=';')
#features = features.iloc[:, np.r_[0:303, 313:327]]
#features = features.iloc[:, np.r_[0:4, 313:327]]
#with hed channel and no ll
features = features.iloc[:, np.r_[0:latent_variable_size + 2, latent_variable_size + 12:latent_variable_size + 32]]
#features = features.iloc[:, np.r_[0:1026, 1036:1056]]
#features = features.iloc[:, np.r_[0:4, 1036:1056]]
#columnsNamesArr = features.columns.values
#print(columnsNamesArr)
#%%
scaler = StandardScaler()
cell_features_scaled = scaler.fit_transform(features.iloc[:, 3:])
cell_features_scaled = np.nan_to_num(cell_features_scaled)
pd_cell_features_scaled = pd.DataFrame(cell_features_scaled, columns = features.iloc[:, 3:].columns)
#%%
#pca
data = []
pca = PCA(n_components=3)
pca_result = pca.fit_transform(cell_features_scaled)
features["1pca_from_features_tumor_vs_non_tumor"] = pca_result[:,0]
features["2pca_from_features_tumor_vs_non_tumor"] = pca_result[:,1]
#print(len(pca_result[:,0]))
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
print(pca.components_)
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
g = sns.scatterplot(data=features, x="1pca_from_features_tumor_vs_non_tumor", y="2pca_from_features_tumor_vs_non_tumor", hue="R2")#, style="class_non_tumor")
#legend1 = ax.legend(*scatter.legend_elements(num=5),
#                  loc="upper right", title="Classes")
plt.savefig('/projects/ag-bozek/lhahn2/plots/' + str(latent_variable_size) + '_PCA_labeled_cells_from_AE_and_features.png')
#%%
#TSNE

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
g = sns.scatterplot(data=features, x="1tsne_dim_from_features_tumor_vs_non_tumor", y="2tsne_dim_from_features_tumor_vs_non_tumor", hue="R2")#, style="class_non_tumor")
#legend1 = ax.legend(*scatter.legend_elements(num=5),
#                  loc="upper right", title="Classes")
plt.savefig('/projects/ag-bozek/lhahn2/plots/' + str(latent_variable_size) + 'TSNE_labeled_cells_from_AE_and_features.png')
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
g = sns.scatterplot(data=features, x="1umap_dim_from_features_tumor_vs_non_tumor", y="2umap_dim_from_features_tumor_vs_non_tumor", hue="R2")#, style="class_non_tumor")
#legend1 = ax.legend(*scatter.legend_elements(num=5),
#                  loc="upper right", title="Classes")
plt.savefig('/projects/ag-bozek/lhahn2/plots/' + str(latent_variable_size) + 'UMAP_labeled_cells_from_AE_and_features.png')
#%%
#print("USE ONLY FEATURES FOR RF AND LOGREG")
print("USE AE AND FEATURES FOR RF AND LOGREG")
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

print(features["R2"].value_counts())
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
ar.to_csv("/projects/ag-bozek/lhahn2/plots/rf_feature_importance_using_ae-and-features.csv", sep=';')


#%%
perm_importance1 = permutation_importance(rf, test_features, test_labels)
#%%
permimportances = perm_importance1['importances_mean']
#%%
perm_importance = pd.DataFrame(perm_importance1['importances_mean'].reshape(1,-1))
liste = list(train_features.columns)
perm_importance.columns = liste
perm_importance = perm_importance.sort_values(by =0, axis=1, ascending=False)
perm_importance.to_csv("/projects/ag-bozek/lhahn2/plots/rf_permutated_feature_importance_using_ae-and-features.csv", sep=';')

#%%
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0, penalty='l2',solver='liblinear').fit(train_features , train_labels)
print("Logistic Regression: mean accuracy for train", clf.score(train_features, train_labels))
print("Logistic Regression: mean accuracy for test", clf.score(test_features, test_labels))
#############################################
##############################################
#%%
f = plt.figure()
count = 0
count1 = 9 
for og, decode in zip(test_images_og, test_images_decoded):
    print(og.shape, decode.shape)
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
plt.savefig('/projects/ag-bozek/lhahn2/plots/' + str(latent_variable_size) + 'Test_images_unlabeled_loss.png')
#plt.savefig('/home/lunas/Desktop/Test_images_unlabeled_loss.png')      
#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.title("Training, Validation and Test Loss")
plt.plot(val_losses,label="val")
plt.plot(train_losses,label="train")
plt.plot(test_losses,label="test")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
#plt.savefig('/home/lunas/Desktop/CAE_unlabeled_loss.png')
plt.savefig('/projects/ag-bozek/lhahn2/plots/'+ str(latent_variable_size) +'CAE_unlabeled_loss.png')
#plt.show()


#%%
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
data = []
pca = PCA(n_components=3)
pca_result = pca.fit_transform(encoded_numpy_test)
first = pca_result[:,0]
second = pca_result[:,1]

df = pd.DataFrame(first)
df[1] = second
df[2] = flat_list_test
df.columns = ["pc1","pc2","labels"]
#print(df)
#print(type(first))
#print(len(pca_result[:,0]))
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
#print(pca.components_)
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
#scatter = ax.scatter(pca_result[:,0], pca_result[:,1])# c=flat_list )
tum = plt.scatter(df[df["labels"] == "Tumor"]["pc1"], df[df["labels"] == "Tumor"]["pc2"], color="red", label='Tumor', facecolors='none')
tum = plt.scatter(df[df["labels"] == "Stroma"]["pc1"], df[df["labels"] == "Stroma"]["pc2"], color="green", label='Stroma', facecolors='none')
tum = plt.scatter(df[df["labels"] == "normal small lymphocyte"]["pc1"], df[df["labels"] == "normal small lymphocyte"]["pc2"], color="blue", label='normal small lymphocyte', facecolors='none')
tum = plt.scatter(df[df["labels"] == "large leucocyte"]["pc1"], df[df["labels"] == "large leucocyte"]["pc2"], color="orange", label='large leucocyte', facecolors='none')
tum = plt.scatter(df[df["labels"] == "Epithelial cell"]["pc1"], df[df["labels"] == "Epithelial cell"]["pc2"], color="purple", label='Epithelial cell', facecolors='none')
plt.legend()
plt.savefig('/projects/ag-bozek/lhahn2/plots/'+ str(latent_variable_size) +'PCA_labeled_cells_from_VAE.png')
#%%   
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
tum = plt.scatter(df[df["labels"] == "Stroma"]["dim1"], df[df["labels"] == "Stroma"]["dim2"], color="green", label='Stroma', facecolors='none')
tum = plt.scatter(df[df["labels"] == "normal small lymphocyte"]["dim1"], df[df["labels"] == "normal small lymphocyte"]["dim2"], color="blue", label='normal small lymphocyte', facecolors='none')
tum = plt.scatter(df[df["labels"] == "large leucocyte"]["dim1"], df[df["labels"] == "large leucocyte"]["dim2"], color="orange", label='large leucocyte', facecolors='none')
tum = plt.scatter(df[df["labels"] == "Epithelial cell"]["dim1"], df[df["labels"] == "Epithelial cell"]["dim2"], color="purple", label='Epithelial cell', facecolors='none')
plt.legend()
plt.savefig('/projects/ag-bozek/lhahn2/plots/'+ str(latent_variable_size) +'TSNE_labeled_cells_from_VAE.png')
#%%
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
tum = plt.scatter(df[df["labels"] == "Stroma"]["dim1"], df[df["labels"] == "Stroma"]["dim2"], color="green", label='Stroma', facecolors='none')
tum = plt.scatter(df[df["labels"] == "normal small lymphocyte"]["dim1"], df[df["labels"] == "normal small lymphocyte"]["dim2"], color="blue", label='normal small lymphocyte', facecolors='none')
tum = plt.scatter(df[df["labels"] == "large leucocyte"]["dim1"], df[df["labels"] == "large leucocyte"]["dim2"], color="orange", label='large leucocyte', facecolors='none')
tum = plt.scatter(df[df["labels"] == "Epithelial cell"]["dim1"], df[df["labels"] == "Epithelial cell"]["dim2"], color="purple", label='Epithelial cell', facecolors='none')
plt.legend()
plt.savefig('/projects/ag-bozek/lhahn2/plots/'+ str(latent_variable_size) +'UMAP_labeled_cells_from_VAE.png')

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

#cell_features_scaled , features["R2"]
train_features, test_features, train_labels, test_labels = train_test_split(encoded_numpy_test , flat_list_test, test_size = 0.2, random_state = 42, stratify=flat_list_test)

#%%
print("rf_and_log_only_ae")
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

#%%
# import xgboost as xgb
# from sklearn.metrics import mean_squared_error
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import GridSearchCV
# X = pd_cell_features_scaled
# y = features["R2"]
# print(len(y), len(X))
# train_features, test_features, train_labels, test_labels = train_test_split(X , y, test_size = 0.2, random_state = 42, stratify=y)

# xgb_cl = xgb.XGBClassifier(objective="multi:softmax", num_classes=5 )

# print(type(xgb_cl))
# #%%
# param_grid = {
#     "max_depth": [3, 4, 5, 7],
#     "learning_rate": [0.1, 0.01, 0.05],
#     "gamma": [0, 0.25, 1],
#     "reg_lambda": [0, 1, 10],
#     "scale_pos_weight": [1, 3, 5],
#     "subsample": [0.8],
#     "colsample_bytree": [0.5],
# }

# grid_cv = GridSearchCV(xgb_cl, param_grid, n_jobs=-1, cv=3, scoring="roc_auc")
# _ = grid_cv.fit(train_features, train_labels)
# #%%
# print(grid_cv.best_score_)
# print(grid_cv.best_params_)
# #%%
# xgb_cl.fit(train_features, train_labels)
# preds = xgb_cl.predict(test_features)
# print(accuracy_score(test_labels, preds))






