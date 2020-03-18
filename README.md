# Deepfake Detection Challenge, Kaggle

## Accessing data
The training/testing data are in `/gpfs/ysm/project/amth/amth_jch97/deepfake/`. See [https://docs.ycrc.yale.edu/](https://docs.ycrc.yale.edu/) for more info.

## Model
Suggestions from Prof. Krishnaswamy:
* Tranformers for classifying as fake or not fake
* Convolutional autoencoder for dimensionality reduction on image frame (a preprocessing step)


Plan: 
1. Figure out # of convoltuional layers, etc. in the autoencoder
2. see if the autoencoder represtnation is good? test different #s of reduced parameters 
2. code transformer and then do hyperparameter tuning 
