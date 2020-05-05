# Deepfake Detection Challenge, Kaggle

## Ideas to make the model learn
* Right now, the last layer of the model is a linear layer which takes as input 1 audio feature (output of audio transformer) and 3600 video features (output of video transformer). We should try having the video features go through a separate linear layer so that the video features don't drown out the single audio feature.
* Just consider faces. Just use an out-of-the-box face cropper to get 128x128 face crops to use as input to the transformer. 

## TODO
* Potentially lower the framerate on the videos. This can be done with 
```
ffmpeg -i <input> -filter:v fps=fps=15 <output>
```
Update: run time would be about 13 days, takes too long. Instead, consider having the iterator skip every other frame etc.

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
