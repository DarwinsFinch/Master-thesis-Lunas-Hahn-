#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 08:35:41 2022

@author: lunas
"""
                                                                                                                                                                                                                                                                      
###file for cluster                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                               
import numpy as np                                                                                                                                                                                                                                                             
import matplotlib.pyplot as plt                                                                                                                                                                                                                                                
import torchvision                                                                                                                                                                                                                                                             
import torch                                                                                                                                                                                                                                                                   
import matplotlib.pyplot as plt                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
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

#ssh -Y -L 16006:localhost:6006 lhahn2@cheops1.rrz.uni-koeln.de
#squeue | grep -E 'excellenc|zmmk' 
#sbatch --partition=excellence-exclusive /projects/ag-bozek/lhahn2/skripte/run_cae.sh
#sbatch --partition=excellence-exclusive --nodelist=cheops31102 /projects/ag-bozek/lhahn2/skripte/run_cae.sh
#scp /home/lunas/Desktop/deep_cae_for_cheops.py lhahn2@cheops1.rrz.uni-koeln.de:/projects/ag-bozek/lhahn2/skripte
#scp lhahn2@cheops1.rrz.uni-koeln.de:/projects/ag-bozek/lhahn2/plots/CAE_unlabeled_loss.png /home/lunas/Desktop/CAE_unlabeled_loss.png
#scp lhahn2@cheops1.rrz.uni-koeln.de:/projects/ag-bozek/lhahn2/plots/Test_images_unlabeled_loss.png /home/lunas/Desktop/Test_images_unlabeled_loss.png
#scancel 'jobnummer'


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
        #x = skimage.io.imread(image_path)
        #x = resize(x, (32, 32),anti_aliasing=True)
        #print(x)
        x = transform(x)
        # Preprocessing
        #if self.transform is not None:
        #    x = self.transform(x)
        print(x.shape)
        x = np.array(x)
        print(x.shape)
        x = np.moveaxis(x, 0, 2)
        x = rgb2hed(x)
        x = x[:, :, 1]
        print(x.shape)
        # Typecasting
        #x = torch.from_numpy(x).type(self.inputs_dtype)

        return x, cell_class
#%%
transform = transforms.Compose([#transforms.Grayscale(num_output_channels=1),
                                transforms.Resize([32, 32]),
                                transforms.ToTensor()])

#dataset = datasets.ImageFolder("/home/lunas/Documents/Uni/Masterarbeit/Data/Images_Bozek/Test_folder_for_Autoencoder", transform=transform)
#dataset = datasets.ImageFolder("/home/lunas/Documents/Uni/Masterarbeit/Data/Images_Bozek/Test_folder_for_Autoencoder_masked", transform=transform)
#dataset = datasets.ImageFolder("/media/lunas/Elements/data_ma/Single_cells_masked_aligned_unlabeled/", transform=transform)
dataset = datasets.ImageFolder("/projects/ag-bozek/lhahn2/data/Single_cells_masked_aligned_unlabeled/", transform=transform)

indices = torch.randperm(len(dataset))[:5000]
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

train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batchsize, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
test_loader2 = DataLoader(dataset=test_dataset, batch_size=batchsize, shuffle=False)


#%%
# pool = nn.MaxPool2d(2, stride=2, return_indices=True)
# unpool = nn.MaxUnpool2d(2, stride=2)
# x = next(iter(train_loader))
# print(x[0].shape)
# zerozero = nn.Conv2d(3, 4, 3, stride=1, padding=0) 
# zerozero_r = nn.ConvTranspose2d(4, 3, 3, stride=1, padding=0, output_padding=0)
# zero = nn.Conv2d(4, 8, 3, stride=1, padding=0) 
# zero_batch = nn.BatchNorm2d(4)
# zero_r = nn.ConvTranspose2d(4, 8, 3, stride=1, padding=0, output_padding=0)
# oneone= nn.Conv2d(8, 16, 3, stride=1, padding=0)
# oneone_r =  nn.ConvTranspose2d(8, 16, 3, stride=1, padding=0)
# one = nn.Conv2d(16, 32, 3, stride=1, padding=0)
# one_r =  nn.ConvTranspose2d(16, 32, 3, stride=1, padding=0)
# one2 = nn.Conv2d(8, 8, 3, stride=1, padding=0)
# one_r2 =  nn.ConvTranspose2d(8, 8, 3, stride=1, padding=0)
# one3 = nn.Conv2d(8, 8, 3, stride=1, padding=0)
# one_r3 =  nn.ConvTranspose2d(8, 8, 3, stride=1, padding=0)
# two = nn.Conv2d(32, 64, 3, stride=1, padding=0)
# two_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, return_indices=True) 
# two_r =  nn.ConvTranspose2d(64, 32, 3, stride=1, padding=0)
# two_unpool = nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2))
# three = nn.Conv2d(64, 128, 3, stride=1, padding=0)
# three_r =  nn.ConvTranspose2d(128, 64, 3, stride=1, padding=0)
# four = nn.Conv2d(32, 64, 3, stride=1, padding=0)
# four_r =  nn.ConvTranspose2d(64, 32, 3, stride=1, padding=0)
# five = nn.Conv2d(64, 128, 3, stride=1, padding=0)
# five_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, return_indices=True) 
# five_r =  nn.ConvTranspose2d(128, 64, 3, stride=1, padding=0)
# five_unpool = nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2))
# six = nn.Conv2d(128, 256, 3, stride=1, padding=0)
# six_r =  nn.ConvTranspose2d(256, 128, 3, stride=1, padding=0)
# seven_bottle = nn.Conv2d(256, 64, 1, stride=1, padding=0)
# seven = nn.Conv2d(64, 512, 3, stride=1, padding=0)
# seven_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)) 
# seven_r =  nn.ConvTranspose2d(512, 256, 3, stride=1, padding=0)

# #eight = nn.Conv2d(78, 142, 3, stride=1, padding=0)
# #eight_r =  nn.ConvTranspose2d(142, 78, 3, stride=1, padding=0)
# #nine = nn.Conv2d(142, 270, 3, stride=1, padding=0)
# #nine_r =  nn.ConvTranspose2d(270, 142, 3, stride=1, padding=0)

# fc1 = nn.Linear(9216, 4608)
# dc1 = nn.Linear(4608, 9216)

# x = zerozero(x[0])
# print("x", x.shape)
# x = zero(x)
# print("x", x.shape)
# #x = zero_batch(x)
# #print("x", x.shape)
# x = oneone(x)
# print("x", x.shape)
# x = one(x)
# print("x", x.shape)
# #x = one2(x)
# #print("x", x.shape)
# #x = one3(x)
# #print("x", x.shape)
# x = two(x)
# print("x", x.shape)
# x = three(x)
# print("x", x.shape)
# x, indices = pool(x)
# print("x", x.shape)
# x = unpool(x, indices)
# print("x", x.shape)
# #%%
# x = three(x)
# print("x", x.shape)
# x = four(x)
# print("x", x.shape)
# x = five(x)
# print("x", x.shape)
# x = six(x)
# print("x", x.shape)
# x = seven_bottle(x)
# print("x", x.shape)
# x = seven(x)
# print("x", x.shape)
# #x = torch.flatten(x, start_dim=1)
# #print("x", x.shape)
# x = seven_r(x)
# print("x", x.shape)
# x = six_r(x)
# print("x", x.shape)
# x = five_r(x)
# print("x", x.shape)
# x = four_r(x)
# print("x", x.shape)
# x = three_r(x)
# print("x", x.shape)
# x = two_r(x)
# print("x", x.shape)
# #x = one_r3(x)
# #print("x", x.shape)
# #x = one_r2(x)
# #print("x", x.shape)
# x = one_r(x)
# print("x", x.shape)
# x = zero_r(x)
# print("x", x.shape)

#%%
#vanilla version from juan
# class VanillaVAE(BaseVAE):


#     def __init__(self,
#                  in_channels: int,
#                  latent_dim: int,
#                  hidden_dims: List = None,
#                  **kwargs) -> None:
#         super(VanillaVAE, self).__init__()

#         self.latent_dim = latent_dim

#         modules = []
#         if hidden_dims is None:
#             hidden_dims = [32, 64, 128, 256, 512]

#         # Build Encoder
#         for h_dim in hidden_dims:
#             modules.append(
#                 nn.Sequential(
#                     nn.Conv2d(in_channels, out_channels=h_dim,
#                               kernel_size= 3, stride= 2, padding  = 1),
#                     nn.BatchNorm2d(h_dim),
#                     nn.LeakyReLU())
#             )
#             in_channels = h_dim

#         self.encoder = nn.Sequential(*modules)
#         self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
#         self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


#         # Build Decoder
#         modules = []

#         self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

#         hidden_dims.reverse()

#         for i in range(len(hidden_dims) - 1):
#             modules.append(
#                 nn.Sequential(
#                     nn.ConvTranspose2d(hidden_dims[i],
#                                        hidden_dims[i + 1],
#                                        kernel_size=3,
#                                        stride = 2,
#                                        padding=1,
#                                        output_padding=1),
#                     nn.BatchNorm2d(hidden_dims[i + 1]),
#                     nn.LeakyReLU())
#             )



#         self.decoder = nn.Sequential(*modules)

#         self.final_layer = nn.Sequential(
#                             nn.ConvTranspose2d(hidden_dims[-1],
#                                                hidden_dims[-1],
#                                                kernel_size=3,
#                                                stride=2,
#                                                padding=1,
#                                                output_padding=1),
#                             nn.BatchNorm2d(hidden_dims[-1]),
#                             nn.LeakyReLU(),
#                             nn.Conv2d(hidden_dims[-1], out_channels= 3,
#                                       kernel_size= 3, padding= 1),
#                             nn.Tanh())

#     def encode(self, input: Tensor) -> List[Tensor]:
#         """
#         Encodes the input by passing through the encoder network
#         and returns the latent codes.
#         :param input: (Tensor) Input tensor to encoder [N x C x H x W]
#         :return: (Tensor) List of latent codes
#         """
#         result = self.encoder(input)
#         result = torch.flatten(result, start_dim=1)

#         # Split the result into mu and var components
#         # of the latent Gaussian distribution
#         mu = self.fc_mu(result)
#         log_var = self.fc_var(result)

#         return [mu, log_var]

#     def decode(self, z: Tensor) -> Tensor:
#         """
#         Maps the given latent codes
#         onto the image space.
#         :param z: (Tensor) [B x D]
#         :return: (Tensor) [B x C x H x W]
#         """
#         result = self.decoder_input(z)
#         result = result.view(-1, 512, 2, 2)
#         result = self.decoder(result)
#         result = self.final_layer(result)
#         return result

#     def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
#         """
#         Reparameterization trick to sample from N(mu, var) from
#         N(0,1).
#         :param mu: (Tensor) Mean of the latent Gaussian [B x D]
#         :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
#         :return: (Tensor) [B x D]
#         """
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return eps * std + mu

#     def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
#         mu, log_var = self.encode(input)
#         z = self.reparameterize(mu, log_var)
#         return  [self.decode(z), input, mu, log_var]


#%%
"""
A Convolutional Variational Autoencoder
"""
class VAE(nn.Module):
    def __init__(self, imgChannels=3, featureDim=512*16*16, zDim=256):
        super(VAE, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(3, 4, 3)
        self.encConv2 = nn.Conv2d(4, 8, 3)
        #self.encConv22 = nn.Conv2d(8, 8, 3)
        self.encConv3 = nn.Conv2d(8, 16, 3)
        self.encConv4 = nn.Conv2d(16, 32, 3)
        self.encConv5 = nn.Conv2d(32, 64, 3)
        self.encConv6 = nn.Conv2d(64, 128, 3)
        self.encConv7 = nn.Conv2d(128, 256, 3)
        self.encConv8 = nn.Conv2d(256, 512, 3)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, stride=2)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv0000 = nn.ConvTranspose2d(512, 256, 3)
        self.decConv000 = nn.ConvTranspose2d(256, 128, 3)
        self.decConv00 = nn.ConvTranspose2d(128, 64, 3)
        self.decConv0 = nn.ConvTranspose2d(64, 32, 3)
        self.decConv1 = nn.ConvTranspose2d(32, 16, 3)
        self.decConv2 = nn.ConvTranspose2d(16, 8, 3)
        #self.decConv22 = nn.ConvTranspose2d(8, 8, 3)
        self.decConv3 = nn.ConvTranspose2d(8, 4, 3)
        self.decConv4 = nn.ConvTranspose2d(4, 3, 3)

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = nn.LeakyReLU(self.encConv1(x))
        x = nn.LeakyReLU(self.encConv2(x))
        #x = nn.relu(self.encConv22(x))
        x = nn.LeakyReLU(self.encConv3(x))
        #x, indices1 = pool(x)
        x = nn.LeakyReLU(self.encConv4(x))
        x = nn.LeakyReLU(self.encConv5(x))
        x = nn.LeakyReLU(self.encConv6(x))
        x = nn.LeakyReLU(self.encConv7(x))
        x = nn.LeakyReLU(self.encConv8(x))
        #x, indices2 = self.pool(x)
        #print("x", x.shape)
        #print("indices", x.shape)
        #x = unpool(x, indices)
        #print("x", x.shape)
        #print(x.shape)
        x = x.view(-1, 512*16*16)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar#, indices2#, indices2

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):#, indices2):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = nn.relu(self.decFC1(z))
        x = x.view(-1, 512, 16, 16)
        #print("input", x.shape)
        #print("indicies", indices2.shape)
        #x = self.unpool(x, indices2)
        #print("unppol", x.shape, output_size=output1.size())
        x = nn.LeakyReLU(self.decConv0000(x))
        x = nn.LeakyReLU(self.decConv000(x))
        x = nn.LeakyReLU(self.decConv00(x))
        x = nn.LeakyReLU(self.decConv0(x))
        x = nn.LeakyReLU(self.decConv1(x))
        #print("input", x.shape)
        #print("indicies", indices1.shape)
        #x = unpool(x, indices1)
        x = nn.LeakyReLU(self.decConv2(x))
        x = nn.LeakyReLU(self.decConv3(x))
        x = torch.sigmoid(self.decConv4(x))
        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        #mu, logVar, indices2 = self.encoder(x)
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        #out = self.decoder(z, indices2)
        out = self.decoder(z)
        return out, mu, logVar
#%%
"""
A Convolutional Variational Autoencoder
"""
# class VAE(nn.Module):
#     def __init__(self, imgChannels=3, featureDim=64*3*3, zDim=256):
#         super(VAE, self).__init__()

#         # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
#         self.encConv1 = nn.Conv2d(3, 16, 6, stride=2, padding=0)
#         self.encConv2 = nn.Conv2d(16, 32, 4, stride=2, padding=0)
#         self.encConv3 = nn.Conv2d(32, 64, 2, stride=2, padding=0)
#         #self.encConv4 = nn.Conv2d(16, 32, 3)
#         #self.encConv5 = nn.Conv2d(32, 64, 3)
#         #self.encConv6 = nn.Conv2d(64, 128, 3)
#         self.encFC1 = nn.Linear(featureDim, zDim)
#         self.encFC2 = nn.Linear(featureDim, zDim)
#         #pool = nn.MaxPool2d(2, stride=2, return_indices=True)
#         #unpool = nn.MaxUnpool2d(2, stride=2)

#         # Initializing the fully-connected layer and 2 convolutional layers for decoder
#         self.decFC1 = nn.Linear(zDim, featureDim)
#         self.decConv00 = nn.ConvTranspose2d(64, 32, 2, stride=2, padding=0 )
#         self.decConv0 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=0)
#         self.decConv1 = nn.ConvTranspose2d(16, 3, 6, stride=2, padding=0)
#         #self.decConv2 = nn.ConvTranspose2d(16, 8, 3)
#         #self.decConv22 = nn.ConvTranspose2d(8, 8, 3)
#         #self.decConv3 = nn.ConvTranspose2d(8, 4, 3)
#         #self.decConv4 = nn.ConvTranspose2d(4, 3, 3)

#     def encoder(self, x):

#         # Input is fed into 2 convolutional layers sequentially
#         # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
#         # Mu and logVar are used for generating middle representation z and KL divergence loss
#         x = F.relu(self.encConv1(x))
#         x = F.relu(self.encConv2(x))
#         #x = F.relu(self.encConv22(x))
#         x = F.relu(self.encConv3(x))
#         #print(x.shape)
#         #x = F.relu(self.encConv4(x))
#         #x = F.relu(self.encConv5(x))
#         #x = F.relu(self.encConv6(x))
#         #x, indices = pool(x)
#         #print("x", x.shape)
#         #print("indices", x.shape)
#         #x = unpool(x, indices)
#         #print("x", x.shape)
#         #print(x.shape)
#         x = x.view(-1, 64*3*3)
#         mu = self.encFC1(x)
#         logVar = self.encFC2(x)
#         return mu, logVar

#     def reparameterize(self, mu, logVar):

#         #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
#         std = torch.exp(logVar/2)
#         eps = torch.randn_like(std)
#         return mu + std * eps

#     def decoder(self, z):

#         # z is fed back into a fully-connected layers and then into two transpose convolutional layers
#         # The generated output is the same size of the original input
#         x = F.relu(self.decFC1(z))
#         x = x.view(-1, 64, 3, 3)
#         #print("input", x.shape)
#         #print("indicies", indices.shape)
#         #x = unpool(x, indices)
#         #print(x.shape)
#         x = F.relu(self.decConv00(x))
#         #print(x.shape)
#         x = F.relu(self.decConv0(x))
#         #print(x.shape)
#         x = F.relu(self.decConv1(x))
#         #print(x.shape)
#         #x = F.relu(self.decConv2(x))
#         #x = F.relu(self.decConv22(x))
#         #x = F.relu(self.decConv3(x))
#         #x = torch.sigmoid(self.decConv4(x))
#         return x

#     def forward(self, x):

#         # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
#         # output, mu, and logVar are returned for loss computation
#         mu, logVar = self.encoder(x)
#         z = self.reparameterize(mu, logVar)
#         out = self.decoder(z)
#         return out, mu, logVar
#%%
"""
Initialize the network and the Adam optimizer
"""
net = VAE()
net = net.to(dev)
batch_size = 2
learning_rate = 1e-3
num_epochs = 25
# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# mean-squared error loss
criterion = nn.MSELoss()
#criterion = SSIM_Loss(data_range=1.0, size_average=True, channel=3)


print("epochs:", num_epochs)
print("learning rate:", learning_rate)
print("loss:", criterion)
print("batchsize:", batchsize)
print("data used:", len(indices))
print("scheduler:", scheduler)
print("model:", net)



#%%
train_losses = []
val_losses = []
train_images_og = []
train_images_decoded = []
val_images_og = []
val_images_decoded = []
count = 0

for epoch in range(num_epochs):
    count = count + 1
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
        x = x[0].to(dev)
        
        optimizer.zero_grad()
        
        outx, mu, logVar = net(x)
        # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
        #train_loss = F.binary_cross_entropy(outx, x, size_average=False) + kl_divergence
        train_loss = criterion(outx, x) + kl_divergence
        #print(train_loss)
        train_loss.backward()
        optimizer.step()
        loss_train += train_loss.item()
        train_images_og.append(x[0])
        train_images_decoded.append(outx[0].detach())
               
        torch.no_grad()
        net.eval() 
        y = y[0].to(dev)
        outy, mu, logVar = net(x)
        # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
        #val_loss = F.binary_cross_entropy(outy, y, size_average=False) + kl_divergence
        val_loss = criterion(outy, y) + kl_divergence
        loss_val += val_loss.item()
        val_images_og.append(y[0])
        val_images_decoded.append(outy[0].detach())
        net.train() 
         
         
        val_images_og.append(y[0])
        val_images_decoded.append(outy[0].detach())
         
    loss_train = loss_train / len(train_loader)
    train_losses.append(loss_train)
    loss_val = loss_val / len(val_loader)
    val_losses.append(loss_val)
    # display the epoch training loss
    print("epoch : {}/{}, train_loss  = {:.6f}".format(epoch + 1, num_epochs , loss_train))    
    print("epoch : {}/{}, val_loss = {:.6f}".format(epoch + 1,num_epochs , loss_val)) 
    #print("encoded_shape:", encodedx.shape)
    #print("output:", outputsx.shape)
    
    scheduler.step()
 
print(len(val_images_og), len(val_images_decoded), len(train_images_og), len(train_images_decoded))        
#%%
for x in test_loader:
    x = x[0].to(dev)
    outx, mu, logVar = net(x)
    kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
    #print(kl_divergence)
    #test_loss = F.binary_cross_entropy(outx, x, size_average=False) + kl_divergence
    test_loss = criterion(outx, x) + kl_divergence
    test_images_og = x#[0]
    test_images_decoded = outx.detach()
    print("test_loss", test_loss.item())
      
test_losses = []
for i in range(0,len(val_losses)):
    test_losses.append(test_loss.item())
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
plt.savefig('/projects/ag-bozek/lhahn2/plots/Test_images_unlabeled_loss.png')
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
plt.savefig('/projects/ag-bozek/lhahn2/plots/CAE_unlabeled_loss.png')
plt.show()

#%%
print("Time for script to run:", datetime.now() - startTime)



#%%
# ##mytry
# # """
# # A Convolutional Variational Autoencoder
# # """
# # class VAE(nn.Module):
# #     def __init__(self, imgChannels=3, featureDim=32*24*24, zDim=256):
# #         super(VAE, self).__init__()

# #         # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        
        
# #         self.encoder = nn.Sequential(
# #           nn.Conv2d(3, 4, 3, stride=1, padding=0),
# #           nn.LeakyReLU(),
# #           nn.Conv2d(4, 8, 3, stride=1, padding=0),
# #           nn.LeakyReLU(),
# #           nn.Conv2d(8, 16, 3, stride=1, padding=0),
# #           nn.LeakyReLU(),
# #           nn.Conv2d(16, 32, 3, stride=1, padding=0),
# #           nn.LeakyReLU(),            
# #           nn.Conv2d(32, 64, 3, stride=1, padding=0),
# #           nn.LeakyReLU(),
# #           nn.Conv2d(64, 128, 3, stride=1, padding=0),
# #           nn.LeakyReLU(),
# #           nn.Conv2d(128, 256, 3, stride=1, padding=0),
# #           nn.LeakyReLU(),
# #           nn.Conv2d(256, 512, 3, stride=1, padding=0),
# #           nn.LeakyReLU(),
# #           nn.Conv2d(512, 32, 1, stride=1, padding=0),
# #           nn.LeakyReLU())
        
        
        
# #         self.decoder = nn.Sequential(
# #             nn.ConvTranspose2d(32, 512, 1, stride=1, padding=0),
# #             nn.LeakyReLU(),
# #             nn.ConvTranspose2d(512, 256, 3, stride=1, padding=0),
# #             nn.LeakyReLU(),
# #             nn.ConvTranspose2d(256, 128, 3, stride=1, padding=0),
# #             nn.LeakyReLU(),
# #             nn.ConvTranspose2d(128, 64, 3, stride=1, padding=0),
# #             nn.LeakyReLU(),
# #             nn.ConvTranspose2d(64, 32, 3, stride=1, padding=0),
# #             nn.LeakyReLU(),
# #             nn.ConvTranspose2d(32, 16, 3, stride=1, padding=0),
# #             nn.LeakyReLU(),
# #             nn.ConvTranspose2d(16, 8, 3, stride=1, padding=0),
# #             nn.LeakyReLU(),
# #             nn.ConvTranspose2d(8, 4, 3, stride=1, padding=0),
# #             nn.LeakyReLU(),
# #             nn.ConvTranspose2d(4, 3, 3, stride=1, padding=0),
# #             nn.Sigmoid())
        
        
# #         self.fc_mu = nn.Linear(8192, 1000)
# #         self.fc_var = nn.Linear(8192, 1000)
        
        

# #         # Initializing the fully-connected layer and 2 convolutional layers for decoder
# #         self.decFC1 = nn.Linear(1000, 8192)
# #         self.decConv1 = nn.ConvTranspose2d(32, 16, 5)
# #         self.decConv2 = nn.ConvTranspose2d(16, imgChannels, 5)


# #     def reparameterize(self, mu, logVar):#

# #         #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
# #         std = torch.exp(logVar/2)
# #         eps = torch.randn_like(std)
# #         return mu + std * eps



# #     def forward(self, x):
# #         encoded = self.encoder(x)
# #         print(encoded.shape)
# #         flatten = torch.flatten(encoded, start_dim=1)
# #         #print(flatten.shape)
# #         mu = self.fc_mu(flatten)
# #         logVar = self.fc_var(flatten)
# #         #print(logVar.shape, mu.shape)
# #         # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
# #         # output, mu, and logVar are returned for loss computation
# #         #mu, logVar = self.encoder(x)
# #         z = self.reparameterize(mu, logVar)
# #         #print(z.shape)
# #         z2 = self.decFC1(z)
# #         #print(z2.shape)
# #         z3 = z2.view(-1, 32, 16, 16)
# #         #print(z3.shape)
# #         out = self.decoder(z3)
# #         #print(out.shape)
# #         return z, out, mu, logVar

# #%%

# # class VAE(nn.Module):
# #     def __init__(self):
# #         """Variational Auto-Encoder Class"""
# #         super(VAE, self).__init__()
# #         # Encoding Layers
# #         self.e_input2hidden = nn.Linear(in_features=infeatures, out_features=e_hidden)
# #         self.e_hidden2hidden2 = nn.Linear(in_features=e_hidden, out_features=e_hidden2)#
# #         self.e_hidden22mean = nn.Linear(in_features=e_hidden2, out_features=latent_dim)
# #         self.e_hidden22logvar = nn.Linear(in_features=e_hidden2, out_features=latent_dim)
        
# #         # Decoding Layers
# #         self.d_latent2hidden2 = nn.Linear(in_features=latent_dim, out_features=d_hidden2)
# #         self.d_hidden22hidden = nn.Linear(in_features=d_hidden2, out_features=d_hidden)#
# #         self.d_hidden2image = nn.Linear(in_features=d_hidden, out_features=infeatures)
        
# #     def forward(self, x):
# #         # Shape Flatten image to [batch_size, input_features]
# #         x = x.view(-1, infeatures)
        
# #         # Feed x into Encoder to obtain mean and logvar
# #         x = F.relu(self.e_input2hidden(x))
# #         x = F.relu(self.e_hidden2hidden2(x))
# #         mu, logvar = self.e_hidden22mean(x), self.e_hidden22logvar(x)
        
# #         # Sample z from latent space using mu and logvar
# #         if self.training:
# #             z = torch.randn_like(mu).mul(torch.exp(0.5*logvar)).add_(mu)
# #         else:
# #             z = mu
        
# #         # Feed z into Decoder to obtain reconstructed image. Use Sigmoid as output activation (=probabilities)
# #         z =torch.relu(self.d_latent2hidden2(z))
# #         z =torch.relu(self.d_hidden22hidden(z))
# #         x_recon = torch.sigmoid(self.d_hidden2image(z))

        
# #         return x_recon, mu, logvar


# # # x_recon ist der im forward im Model erstellte recon_batch, x ist der originale x Batch, mu ist mu und logvar ist logvar 
# # def vae_loss(x_recon, x, mu, logvar):
# #     mse_loss = nn.MSELoss(reduction="sum")
# #     loss_MSE = mse_loss(x_recon, x)
# #     loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

# #     return loss_MSE + loss_KLD, loss_MSE, loss_KLD

# # # Instantiate VAE with Adam optimizer
# # vae = VAE()
# # #vae = vae.to(device)    # send weights to GPU. Do this BEFORE defining Optimizer
# # optimizer = optim.Adam(params=vae.parameters(), lr=learning_rate, weight_decay=weight_decay)
# # vae.train()            # tell the network to be in training mode. Useful to activate Dropout layers & other stuff

# #%%
# def criterion_final(bce_loss, mu, logvar):
#     """
#     This function will add the reconstruction loss (BCELoss) and the 
#     KL-Divergence.
#     KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     :param bce_loss: recontruction loss
#     :param mu: the mean from the latent vector
#     :param logvar: log variance from the latent vector
#     """
#     BCE = bce_loss 
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return BCE + KLD


# model = VAE()
# model = model.to(dev)
# lr = 1e-3
# # create an optimizer object
# # Adam optimizer with learning rate 1e-3
# optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# # mean-squared error loss
# #criterion = nn.BCELoss(reduction='sum')
# criterion = nn.MSELoss()
# #criterion_final = final_loss()

# epochs = 30
# print("epochs:", epochs)
# print("learning rate:", lr)
# print("loss:", criterion)
# print("batchsize:", batchsize)
# print("data used:", len(indices))
# print("scheduler:", scheduler)
# print("model:", model)

# #%%
# train_losses = []
# val_losses = []
# train_images_og = []
# train_images_decoded = []
# val_images_og = []
# val_images_decoded = []
# count = 0
# for epoch in range(epochs):
#     count = count + 1
#     loss_train = 0
#     loss_val = 0
#     train_images_og = []
#     train_images_decoded = []
#     val_images_og = []
#     val_images_decoded = []
#     #for n, (x,y) in enumerate(zip(train_loader, val_loader)):
#     for n,(x,y) in enumerate(zip(train_loader, val_loader)):   
#         # reshape mini-batch data to [N, 784] matrix
#         # load it to the active device
#         #batch_features = batch_features.view(-1, 750)
#         x = x[0].to(dev)
        
#         optimizer.zero_grad()
        
#         latentx, outputsx, mux, logVarx = model(x)
 
#         train_loss = criterion(outputsx, x)
#         #print("train 1:", train_loss)
#         train_loss = criterion_final (train_loss, mux, logVarx)
#         #print("train 2:", train_loss)
        
#         train_loss.backward()
        
#         optimizer.step()
        
#         loss_train += train_loss.item()
        
#         train_images_og.append(x[0])
#         train_images_decoded.append(outputsx[0].detach())
        
        
#         torch.no_grad()
#         model.eval() 
#         y = y[0].to(dev)
#         # reset the gradients back to zero
#         # PyTorch accumulates gradients on subsequent backward passes
#         #optimizer.zero_grad()
#         #print(x[1])
#         # compute reconstructions
#         #encodedx, outputsx = model(x)
#         latenty, outputsy, muy, logVary = model(y)
#         #print(outputs.shape)

#         # compute training reconstruction loss
#         #train_loss = criterion(outputsx, x)
#         val_loss = criterion(outputsy, y)
#         val_loss = criterion_final (val_loss, muy, logVary)
#         # compute accumulated gradients
#         #train_loss.backward()
        
#         # perform parameter update based on current gradients
#         #optimizer.step()
        
#         # add the mini-batch training loss to epoch loss
#         #loss_train += train_loss.item()
#         loss_val += val_loss.item()
        
#         #save images
#         #train_images_og.append(x[0])
#         #train_images_decoded.append(outputsx[0].detach())
#         val_images_og.append(y[0])
#         val_images_decoded.append(outputsy[0].detach())
#         model.train() 
#     # compute the epoch training loss
    

#     loss_train = loss_train / len(train_loader)
#     train_losses.append(loss_train)
#     loss_val = loss_val / len(val_loader)
#     val_losses.append(loss_val)
#     # display the epoch training loss
#     print("epoch : {}/{}, train_loss  = {:.6f}".format(epoch + 1, epochs, loss_train))    
#     print("epoch : {}/{}, val_loss = {:.6f}".format(epoch + 1, epochs, loss_val)) 
#     #print("encoded_shape:", encodedx.shape)
#     #print("output:", outputsx.shape)
    
#     scheduler.step()
 
# print(len(val_images_og), len(val_images_decoded), len(train_images_og), len(train_images_decoded))

# #%%
# for x in test_loader:
#     x = x[0].to(dev)
#     #print(x.shape)
#     latentx, outputsx, mux, logVarx = model(x)
#     latentx_numpy = latentx.cpu().detach().numpy()
    
#     test_loss = criterion(outputsx, x)
#     test_loss = criterion_final(test_loss, mux, logVarx)
    
#     #test_loss = criterion(outputsx, x)
#     test_images_og = x#[0]
#     test_images_decoded = outputsx.detach()
#     print("test_loss", test_loss.item())
#     #print(test_images_og.shape, test_images_decoded.shape)
    
# test_losses = []
# for i in range(0,len(val_losses)):
#     test_losses.append(test_loss.item())
# #%%
# f = plt.figure()
# count = 0
# count1 = 9 
# for og, decode in zip(test_images_og, test_images_decoded):
#     print(og.shape, decode.shape)
#     ##print(f, b)
#     count = count + 1
#     f.add_subplot(2,9, count)
#     plt.imshow(og.cpu().permute(1, 2, 0))
#     #plt.title("Train_OG_" + str(count))
#     plt.axis('off')
#     count1 = count1 + 1
#     f.add_subplot(2,9, count1)
#     plt.imshow(decode.cpu().permute(1, 2, 0))
#     #plt.title("Train_Decoded_" + str(count))
#     plt.axis('off')
#     if count > 8:
#         break
# plt.savefig('/projects/ag-bozek/lhahn2/plots/Test_images_unlabeled_loss.png')
# #plt.savefig('/home/lunas/Desktop/Test_images_unlabeled_loss.png')      
# #%%
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10,5))
# plt.title("Training, Validation and Test Loss")
# plt.plot(val_losses,label="val")
# plt.plot(train_losses,label="train")
# plt.plot(test_losses,label="test")
# plt.xlabel("iterations")
# plt.ylabel("Loss")
# plt.legend()
# #plt.savefig('/home/lunas/Desktop/CAE_unlabeled_loss.png')
# plt.savefig('/projects/ag-bozek/lhahn2/plots/CAE_unlabeled_loss.png')
# plt.show()

# #%%
# print("Time for script to run:", datetime.now() - startTime)

