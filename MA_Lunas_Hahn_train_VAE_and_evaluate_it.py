#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 08:35:41 2022

@author: lunas
"""
#this script builds and trains a VAE                     
#https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
                                                                                                                                                                                                                                               
###file for cluster                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                               
import numpy as np                                                                                                                                                                                                                                                             
import matplotlib.pyplot as plt                                                                                                                                                                                                                                                
import torchvision                                                                                                                                                                                                                                                             
import torch                                                                                                                                                                                                                                                                   
import matplotlib.pyplot as plt                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
from torchvision import datasets, transforms                                                                                                                                                                                                                                   
#import helper                                                                                                                                                                                                                                                                 
from torch.utils.data import Dataset, DataLoader   
from torch.nn import functional as F                                                                                                                                                                                                                            
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
import seaborn as sns
startTime = datetime.now()                                                                                                                                                                                                                                            
#import pytorch_ssim                                                                                                                                                                                                                                                           
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


torch.cuda.is_available()

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 

print(dev)


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


#all cells on server
inputs = "/projects/ag-bozek/lhahn2/data/Single_cells_masked_aligned_all_cells_compare_to_features/labels_of_cell_for_AE_compare_to_features_all_cells_for_hpc_cleaned.csv"



dataset =  SegmentationDataSet(inputs=inputs)
print(len(dataset))
#indices = torch.randperm(len(dataset))[:7100]
indices = torch.randperm(len(dataset))[:65000]
#indices = torch.randperm(len(dataset))[:1000]
dataset = data_utils.Subset(dataset, indices)

#%%
#input like in deep cae with feautre comparision
features_df = pd.read_csv('/projects/ag-bozek/lhahn2/data/Feature_gathering_from_og_contours_with_interpolate_pca_from_labeled_with_hed_channel.csv', sep=',')  

ae_df = pd.read_csv('/projects/ag-bozek/lhahn2/data/Single_cells_masked_aligned_labeled_compare_to_features/labled_data_out_to_compare_ae_and_features_for_hpc_cleaned.csv', sep=';')  

#inputs="/projects/ag-bozek/lhahn2/data/Single_cells_masked_aligned_labeled_compare_to_features/labels_of_cell_for_AE_compare_to_features_for_hpc_cleaned.csv"




#%%
train_split = 0.8
test_split = 0.2
val_split = 0.2
batchsize = 2

dataset_size = len(dataset) #200
val_size = int(val_split * dataset_size)
train_size = dataset_size - val_size


train_dataset, test_dataset = torch.utils.data.random_split(dataset,[train_size, val_size])
#train_dataset, test_dataset = torch.utils.data.random_split(train_dataset,[int(len(train_dataset)*0.9), int(len(train_dataset)*0.1)])

#train_dataset, val_dataset = torch.utils.data.random_split(dataset,[train_size, val_size])
#train_dataset, test_dataset = torch.utils.data.random_split(train_dataset,[int(len(train_dataset)*0.9), int(len(train_dataset)*0.1)])
#print(len(train_dataset))

train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=True)
#val_loader = DataLoader(dataset=val_dataset, batch_size=batchsize, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
test_loader2 = DataLoader(dataset=test_dataset, batch_size=batchsize, shuffle=False)
val_loader = test_loader2



#%%
"""
A Convolutional Variational Autoencoder
"""
zDim=1024

class VAE(nn.Module):
    def __init__(self, imgChannels=3, featureDim=4*16*16, zDim=1024, Only_Decode = False):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), 
            nn.LeakyReLU(), # [batch, 12, 16, 16]
            nn.Conv2d(32, 64, 4, stride=2, padding=1), 
            nn.LeakyReLU(), # [batch, 12, 16, 16]
         

        )       
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, output_padding= 0),  # [batch, 48, 4, 4]
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, output_padding= 0),  # [batch, 48, 4, 4]
            nn.LeakyReLU(),
          
        ) 
         
       
        self.encFC1 = nn.Linear(64*8*8, 512)
        self.encFC2 = nn.Linear(64*8*8, 512)
        self.decFC1 = nn.Linear(512, 64*8*8)

        self.fc1 = nn.Linear(64*8*8, 512)  
        self.d1 = nn.Linear(512, 64*8*8)

    def encoding(self, x):
       
        x = self.encoder(x)
        #print(x.shape)
        x_flat = x.view(-1, 64*8*8)
        mu = self.encFC1(x_flat)
        logVar = self.encFC2(x_flat)
        return mu, logVar, x# indices2#, indices2

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar*0.5)
        eps = torch.randn_like(std)
        
        return mu + std * eps

    def decoding(self, z):#, indices2):

        x = self.decoder(z)
        return x

    def forward(self, x, Only_Decode):
        
        #print(Only_Decode)
        if Only_Decode == False:
            mu, logVar, encoded = self.encoding(x)
            z = self.reparameterize(mu, logVar) 
            decoded = self.d1(z)
            decoded = decoded.view(-1, 64, 8, 8)
            decoded = self.decoding(decoded)
            return decoded, mu, logVar, z
        
        else:
            decoded = self.d1(x)
            decoded = decoded.view(-1, 64, 8, 8)
            decoded = self.decoding(decoded)
            return decoded
        
        #return out, mu, logVar, z
#%%
#%%
"""
Initialize the network and train it.
"""
hyperparamters = [1]
Collect_test_losses = []
Only_Decode = False
for i in hyperparamters:

    net = VAE()
    #net = Autoencoder()
    net = net.to(dev)
    batch_size = 2
    learning_rate = 1e-3
    num_epochs = 200
    devider = 1/num_epochs
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


##%%
#hyperparamters = [0, 0.5, 1, 1.5, 2]
#Collect_test_losses = []
#for i in hyperparamters:
    weight = i
    list_of_labels = []
    train_losses = []
    train_losses_mse = []
    val_losses_mse = []
    test_losses_mse = []
    train_losses_kld = []
    val_losses_kld = []
    test_losses_kld = []
    val_losses = []
    train_images_og = []
    train_images_decoded = []
    val_images_og = []
    val_images_decoded = []
    count = 0
    count_kld_weighting = 0 
    
    for epoch in range(num_epochs):
        count_kld_weighting =  count_kld_weighting + 1
        loss_train = 0
        loss_train_mse = 0
        loss_train_kld = 0
        loss_val_mse = 0
        loss_val_kld = 0
        loss_val = 0
        loss_test_mse = 0
        loss_test_kld = 0
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
            labely = y[1]
            #print("labelx", labelx)
            list_of_labels.append(labelx)
            x = x[0].to(dev)
            y = y[0].to(dev)
            #print("x-shape", x.shape)
            
            optimizer.zero_grad()
            
            outx, mu, logVar, latentx = net(x, Only_Decode)
            
            # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
            kl_divergence_og = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
            kl_divergence = kl_divergence_og*epoch*weight*devider
            loss_train_kld += kl_divergence_og.item()
            #start kld later
            if  count_kld_weighting == 0:
                kl_divergence = kl_divergence*0
            if  count_kld_weighting == 1:
                kl_divergence = kl_divergence*0
            if  count_kld_weighting == 2:
                kl_divergence = kl_divergence*0
            if  count_kld_weighting == 3:
                kl_divergence = kl_divergence*0
            if  count_kld_weighting == 4:
                kl_divergence = kl_divergence*0
            if  count_kld_weighting == 5:
                kl_divergence = kl_divergence*0
            if  count_kld_weighting == 6:
                #kl_divergence = kl_divergence*0.00001
                kl_divergence = kl_divergence*0.1
            if  count_kld_weighting == 7:
                #kl_divergence = kl_divergence*0.00001
                kl_divergence = kl_divergence*0.5
            if  count_kld_weighting == 8:
                #kl_divergence = kl_divergence*0.0001
                kl_divergence = kl_divergence*1
            if  count_kld_weighting == 9:
                #kl_divergence = kl_divergence*0.001
                kl_divergence = kl_divergence*1
                
            elif count_kld_weighting >= 10 :
                #kl_divergence = kl_divergence*0.1 
                kl_divergence = kl_divergence*1 
                count_kld_weighting = 0
                #if count_kld_weighting == 13:
                #    count_kld_weighting = 0
            
           #kl_divergence = kl_divergence*0
            #print("KL", kl_divergence)
            #kld_loss = torch.mean(-0.5 * torch.sum(1 + logVar - mu ** 2 - logVar.exp(), dim = 1), dim = 0)
            #print("KLD", kl_divergence)
            #train_loss = F.binary_cross_entropy(outx, x, size_average=False) + kl_divergence
            #print("MSE", criterion(outx, x))
            #print("BCE", F.binary_cross_entropy(outx, x, reduction='sum'))
            #train_loss = F.binary_cross_entropy(outx, x, reduction='sum') + kl_divergence
            train_loss = criterion(outx, x) + kl_divergence#*beta_norm
            #print(train_loss)
            train_loss.backward()
            optimizer.step()
            loss_train += train_loss.item()
            loss_train_mse += criterion(outx, x).item()
            #loss_train_kld += kl_divergence.item()
            
            #kann weg weil ich eh nur test bilder angucke
            #train_images_og.append(x[0])
            #train_images_decoded.append(outx[0].detach())
            
            #validation
            outy, mu, logVar, latentx = net(x, Only_Decode)
            
            # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
            kl_divergence_og = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
            kl_divergence = kl_divergence_og*epoch*weight*devider
            loss_val_kld += kl_divergence_og.item()
            #start kld later
            if  count_kld_weighting == 0:
                kl_divergence = kl_divergence*0
            if  count_kld_weighting == 1:
                kl_divergence = kl_divergence*0
            if  count_kld_weighting == 2:
                kl_divergence = kl_divergence*0
            if  count_kld_weighting == 3:
                kl_divergence = kl_divergence*0
            if  count_kld_weighting == 4:
                kl_divergence = kl_divergence*0
            if  count_kld_weighting == 5:
                kl_divergence = kl_divergence*0
            if  count_kld_weighting == 6:
                #kl_divergence = kl_divergence*0.00001
                kl_divergence = kl_divergence*0.1
            if  count_kld_weighting == 7:
                #kl_divergence = kl_divergence*0.0001
                kl_divergence = kl_divergence*0.5
            if  count_kld_weighting == 8:
                #kl_divergence = kl_divergence*0.001
                kl_divergence = kl_divergence*1
            if  count_kld_weighting == 9:
                #kl_divergence = kl_divergence*0.01
                kl_divergence = kl_divergence*1
            elif count_kld_weighting >= 10 :
                #kl_divergence = kl_divergence*0.1
                kl_divergence = kl_divergence*1
            
            #kl_divergence = kl_divergence*0
            val_loss = criterion(outy, y) + kl_divergence#*beta_norm
            loss_val += val_loss.item()
            loss_val_mse += criterion(outy, y).item()


            #get latents
            #if count == 0:
            #    encoded_numpy = latentx.cpu().detach().numpy()
            #    count = count + 1
            #    #print("hi")
            #else:
            #    encoded_numpy = np.concatenate((encoded_numpy, latentx.cpu().detach().numpy()), axis=0)
                   
             
        loss_train = loss_train / len(train_loader)
        train_losses.append(loss_train)
        
        loss_val = loss_val / len(val_loader)
        val_losses.append(loss_val)
        
        loss_train_mse = loss_train_mse / len(train_loader)
        train_losses_mse.append(loss_train_mse)
        
        loss_val_mse = loss_val_mse / len(val_loader)
        val_losses_mse.append(loss_val_mse)
        
        loss_train_kld = loss_train_kld / len(train_loader)
        train_losses_kld.append(loss_train_kld)
        
        loss_val_kld = loss_val_kld / len(val_loader)
        val_losses_kld.append(loss_val_kld)
        
        
        # display the epoch training loss
        print("KL", kl_divergence)
        print("KLog", kl_divergence_og)
        #print("KLD", kld_loss)
        print("MSE", criterion(outx, x))
        print("epoch : {}/{}, train_loss  = {:.6f}".format(epoch + 1, num_epochs , loss_train))    
        #print("epoch : {}/{}, val_loss = {:.6f}".format(epoch + 1,num_epochs , loss_val)) 
        #print("encoded_shape:", encodedx.shape)
        #print("output:", outputsx.shape)
        
        scheduler.step()
    
     
    #print(len(val_images_og), len(val_images_decoded), len(train_images_og), len(train_images_decoded))        
    ##%%
    #flat_list_train = [item for sublist in list_of_labels for item in sublist]
    #encoded_numpy_train = torch.flatten(torch.tensor(encoded_numpy) , start_dim=1)
    
    ##%%
    
    list_for_pd = []
    list_of_labels = []
    for x in test_loader:
        image_path = x[2]
        labelx = x[1]
        entry = [image_path, labelx]
        list_for_pd.append(entry)
        list_of_labels.append(labelx)
        x = x[0].to(dev)
        #print(x.shape)
        outx, mu, logVar, encodedx = net(x, Only_Decode)
        encoded_numpy = encodedx.cpu().detach().numpy()
        
        kl_divergence_og = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
        kl_divergence = kl_divergence_og*epoch*weight*devider*1
        loss_test_kld += kl_divergence_og.item()
        #print(kl_divergence)
        #test_loss = F.binary_cross_entropy(outx, x, size_average=False) + kl_divergence
        test_loss = criterion(outx, x) + kl_divergence
        loss_test_mse += criterion(outx, x).item()
        test_images_og = x#[0]
        test_images_decoded = outx.detach()
        print("test_loss", test_loss.item())
        print("testMSE", criterion(outx, x))
        print("testKL", kl_divergence)
        print("testKLog", kl_divergence_og)
        #test_losses_kld.append(loss_test_kld)
        #test_losses_mse.append(loss_test_mse)
        
    test_losses = []
    test_losseskld = []
    test_lossesmse = []
    for i in range(0,len(val_losses)):
        test_losses.append(test_loss.item())
    for i in range(0,len(val_losses)):
        test_losseskld.append(loss_test_kld)
    for i in range(0,len(val_losses)):
        test_lossesmse.append(loss_test_mse)
    
    aedf = pd.DataFrame(list_for_pd)
    Collect_test_losses.append(test_loss.item())
#%%
#for i in Collect_test_losses:
#    print(i)
#print(encoded_numpy.shape)
print("ENCODED NUMPY TEST MEAN AND STD")
mean = encoded_numpy.mean(axis=0)
std = encoded_numpy.std(axis=0)
print(encoded_numpy.mean(axis=0))
print(encoded_numpy.std(axis=0))
print(mean.shape)
print(type(mean))
print(max(mean))

print("MEAN")
maxElement = np.amax(mean)
result = np.where(mean == np.amax(mean))
print("max element", maxElement)
print("positions", result)
print("posiitons in mean", mean[result])


print("STD")
maxElement = np.amax(std)
result = np.where(std == np.amax(std))
print("max element", maxElement)
print("positions", result)
print("posiitons in mean", std[result])
#%%


flat_list_test = [item for sublist in list_of_labels for item in sublist]
f = torch.flatten(encodedx, start_dim=1)
encoded_numpy_test = f.cpu().detach().numpy()
print("shape of latent:", encoded_numpy_test.shape)

projects/ag-bozek/lhahn2/plots/AndreWith_Weight_Small_Latent_Cyclic_Zeros_decoded_dim' + str(i) + '.png')


#%%
#Evaluate the VAE
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
plt.savefig('/projects/ag-bozek/lhahn2/plots/AndreWith_Weight_Small_Latent_Cyclic' + str(zDim) + '_Test_images_labeled_loss_CAE2.png')
plt.savefig('/projects/ag-bozek/lhahn2/plots/AndreWith_Weight_Small_Latent_Cyclic' + str(zDim) + '_Test_images_labeled_loss_CAE2.svg')
#plt.savefig('/projects/ag-bozek/lhahn2/plots/Cyclic' + str(zDim) + '_Test_images_labeled_loss_VAE.svg')
#plt.savefig('/home/lunas/Desktop/Test_images_unlabeled_loss.png')      
#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(val_losses,label="Validation")
plt.plot(train_losses,label="Training")
#plt.plot(test_losses,label="test")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
#plt.savefig('/home/lunas/Desktop/CAE_unlabeled_loss.png')
plt.savefig('/projects/ag-bozek/lhahn2/plots/AndreWith_Weight_Small_Latent_Cyclic'+ str(zDim) +'_CAE_labeled_loss_CAE2.png')
plt.savefig('/projects/ag-bozek/lhahn2/plots/AndreWith_Weight_Small_Latent_Cyclic'+ str(zDim) +'_CAE_labeled_loss_CAE2.svg')
#plt.savefig('/projects/ag-bozek/lhahn2/plots/Cyclic'+ str(zDim) +'_CAE_labeled_loss_VAE.svg')
plt.show()
#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.title("Training and Validation KLD-Loss")
plt.plot(val_losses_kld,label="Validation")
plt.plot(train_losses_kld,label="Training")
#plt.plot(test_losseskld,label="test")
plt.xlabel("Epochs")
plt.ylabel("KLD-Loss")
plt.yscale("log")
plt.legend()
#plt.savefig('/home/lunas/Desktop/CAE_unlabeled_loss.png')
plt.savefig('/projects/ag-bozek/lhahn2/plots/AndreWith_Weight_Small_Latent_Cyclic'+ str(zDim) +'_CAE_labeled_kldloss_CAE2.png')
plt.savefig('/projects/ag-bozek/lhahn2/plots/AndreWith_Weight_Small_Latent_Cyclic'+ str(zDim) +'_CAE_labeled_kldloss_CAE2.svg')
#plt.savefig('/projects/ag-bozek/lhahn2/plots/Cyclic'+ str(zDim) +'_CAE_labeled_loss_VAE.svg')
plt.show()
#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.title("Training and Validation MSE-Loss")
plt.plot(val_losses_mse,label="Validation")
plt.plot(train_losses_mse,label="Training")
#plt.plot(test_lossesmse,label="test")
plt.xlabel("Epochs")
plt.ylabel("MSE-Loss")
plt.legend()
#plt.savefig('/home/lunas/Desktop/CAE_unlabeled_loss.png')
plt.savefig('/projects/ag-bozek/lhahn2/plots/AndreWith_Weight_Small_Latent_Cyclic'+ str(zDim) +'_CAE_labeled_mseloss_CAE2.png')
plt.savefig('/projects/ag-bozek/lhahn2/plots/AndreWith_Weight_Small_Latent_Cyclic'+ str(zDim) +'_CAE_labeled_mseloss_CAE2.svg')
#plt.savefig('/projects/ag-bozek/lhahn2/plots/Cyclic'+ str(zDim) +'_CAE_labeled_loss_VAE.svg')
plt.show()

print("Time for script to run:", datetime.now() - startTime)

