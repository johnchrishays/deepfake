# Deepfake Detection Challenge, Kaggle

## TODO
* Chris: Currently trying to write `process_all_autoencoder` to map each video frame to its encoded representation. Issue: with more than 10 frames, get error like `RuntimeError: CUDA out of memory. Tried to allocate 3.86 GiB (GPU 0; 10.92 GiB total capacity; 5.17 GiB already allocated; 1.52 GiB free; 3.72 GiB cached)`. Need to either figure out how to break videos into chunks (with a small number of frames in each), or get more memory. Seems like the GPUs max out at 16 GB vRAM/GPU.  

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
