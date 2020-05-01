# Deepfake Detection Challenge, Kaggle

## TODO
* Potentially lower the framerate on the videos. This can be done with 
```
ffmpeg -i <input> -filter:v fps=fps=30 <output>
```
* Add transformer for audio and combine it with image seq transformer with a linear layer

## Setup 

The first time:
```
module load miniconda
conda create -n deepfake python=3.7 pytorch
```

To install a new package:
```
conda install opencv
```

Every time:
```
./init
```
in `deepfake`.

For jupyter notebooks:
```
./jupinit
```
then copy the line
```
MacOS or linux terminal command to create your ssh tunnel
ssh -N -L {port}:{node}:{port} {un}@farnam.hpc.yale.edu
```
For a GPU-enabled server:
```
./gpuinit
```

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
