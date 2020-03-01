#Deepfake Detection Challenge, Kaggle

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
source activate deepfake
```

## Accessing data
The training/testing data are in `/gpfs/ysm/project/amth/amth_jch97/deepfake/`. See [https://docs.ycrc.yale.edu/](https://docs.ycrc.yale.edu/) for more info.

## Model
Suggestions from Prof. Krishnaswamy:
* Convolutional autoencoder for dimensionality reduction on image frame. See seminal paper [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf) for an overview.
* Tranformers for classifying as fake or not fake




