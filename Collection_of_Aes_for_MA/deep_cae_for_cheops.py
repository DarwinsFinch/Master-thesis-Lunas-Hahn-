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
transform = transforms.Compose([#transforms.Grayscale(num_output_channels=1),
                                transforms.Resize([32, 32]),
                                transforms.ToTensor()])

#dataset = datasets.ImageFolder("/media/lunas/Elements/data_ma/Single_cells_masked_aligned_unlabeled/", transform=transform)
dataset = datasets.ImageFolder("/projects/ag-bozek/lhahn2/data/Single_cells_masked_aligned_unlabeled/", transform=transform)

indices = torch.randperm(len(dataset))[:5000]
dataset = data_utils.Subset(dataset, indices)

#%%
train_split = 0.8
test_split = 0.2
val_split = 0.2

dataset_size = len(dataset) #200
val_size = int(val_split * dataset_size)
train_size = dataset_size - val_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset,[train_size, val_size])
train_dataset, test_dataset = torch.utils.data.random_split(train_dataset,[int(len(train_dataset)*0.8), int(len(train_dataset)*0.2)])
#print(len(train_dataset))

train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
test_loader2 = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)

#%%
class Autoencoder(nn.Module):
    def __init__(self, ndf = 64, ngf = 64, latent_variable_size=128):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 4, 3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(4, 8, 3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, 3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3, stride=1, padding=0),
            nn.LeakyReLU(),            
            nn.Conv2d(32, 64, 3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 2048, 3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(2048, 4096, 3, stride=1, padding=0),
            nn.LeakyReLU(),
            #nn.Conv2d(4096, 8192, 3, stride=1, padding=0),
            #nn.LeakyReLU(),
        )
        
        #self.fc1 = nn.Linear(131072, 32768)
        #self.fc2 = nn.Linear(768, 384)
        
        self.decoder = nn.Sequential(
            #nn.ConvTranspose2d(8192, 4096, 3, stride=1, padding=0),
            #nn.LeakyReLU(),
            nn.ConvTranspose2d(4096, 2048, 3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2048, 1024, 3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(1024, 512, 3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 256, 3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 4, 3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4, 3, 3, stride=1, padding=0),
            nn.Sigmoid(),
        )
        
        #self.d1 = nn.Linear(32768, 131072)
        
    def forward(self, x):
        encoded = self.encoder(x)
        #encoded = encoded.view(-1, 1024*14*14)
        #encoded = self.fc1(encoded)
        #decoded = self.d1(encoded)
        #decoded = decoded.view(-1, 1024, 14, 14)
        decoded = self.decoder(encoded)
        return encoded, decoded
#%%
#model = AE(input_shape=750)
model = Autoencoder()
model = model.to(dev)

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# mean-squared error loss
criterion = nn.MSELoss()
#criterion = SSIM_Loss(data_range=1.0, size_average=True, channel=3)

epochs = 50
#%%
train_losses = []
val_losses = []
train_images_og = []
train_images_decoded = []
val_images_og = []
val_images_decoded = []
count = 0
for epoch in range(epochs):
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
        y = y[0].to(dev)
        # reset the gradients back to zero
        # PyTorch accumulates gradients on subsequent backward passes
        optimizer.zero_grad()
        #print(x[1])
        # compute reconstructions
        encodedx, outputsx = model(x)
        encodedy, outputsy = model(y)
        #print(outputs.shape)

        # compute training reconstruction loss
        train_loss = criterion(outputsx, x)
        val_loss = criterion(outputsy, y)
        
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
 
print(len(val_images_og), len(val_images_decoded), len(train_images_og), len(train_images_decoded))
#%%
for x in test_loader :
    x = x[0].to(dev)
    print(x.shape)
    encodedx, outputsx = model(x)
    encoded_numpy = encodedx.cpu().detach().numpy()
    test_loss = criterion(outputsx, x)
    test_images_og = x#[0]
    test_images_decoded = outputsx.detach()
    print(test_loss.item())
    print(test_images_og.shape, test_images_decoded.shape)
    
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
