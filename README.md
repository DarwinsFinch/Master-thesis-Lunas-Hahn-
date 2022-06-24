# Master-thesis-Lunas-Hahn-
This repository contains the scripts that were used for my master thesis project, titled 'Implementing data science and
machine learning algorithms to extract morphological information from histopathology slides of DLBCL tissue'

The repository includes the follwoing scripts;

1. XXX
This is used to interpolate the contour points from each cell, so that there are 50 points for each single cell. 
Further, the script creates a csv file that includes the cells name, label, number, and contour points. 

2. XXX
This is used to get the single cell images out of the whole slide image

3. XXX
This is used to calculate the different CV features and do the PCA on the contour points. 

4. XXX
This is used to scale and analyse the CV and PCA feautures; With this I determine the pearson correlation, 
use PCA, t-SNE and UMAP and train the RF and Lasso regression model to evaluate the meaningfulness of the features.
It also combines both feature vector and check the meaningfulness for this vector with the same approach as above. 

5. XXX
This builds and trains the AE on the labeled single cell images. It also uses the latent as a feature vector and does 
PCA, t-SNE and UMAP as well as RF and Lasso with this vector. It also combines the AE and CV features and uses this 
vector for the same ML approaches. 

6. XXX
This builds and trains the VAE using a cyclic weighting of the KLD-loss.
