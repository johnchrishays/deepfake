# Deepfake Detection Challenge, Kaggle

## Setup (Yale Center for Research Computing internal)

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

## Files
Models: `models.py`
Training: `train_*.py`
Testing: `test_*.py`
Dataset classes: `datasets.py`
Models: `*.py`
Dataset: `train/` (50/50 real/fake training set in `train/balanced/` and test in `train/test_balanced`
Cache of images already processed through CAE: `encode_cache/` or `face_encode_cache/`
