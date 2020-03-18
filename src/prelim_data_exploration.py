import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt

DEEPFAKE_DIR = '/gpfs/ysm/project/amth/amth_jch97/deepfake/'
TRAIN_DIR = 'train/'
TRAIN_PART = 'dfdc_train_part_'
TEST_DIR = 'test/test_videos/'

num_train = 0 
for i in range(50):
	path = os.path.join(DEEPFAKE_DIR, TRAIN_DIR + TRAIN_PART + str(i))
	num_train += len(os.listdir(path))

num_test = os.path.join(DEEPFAKE_DIR, TEST_DIR)
print(f"Train samples: {num_train}")
print(f"Test samples: {num_test}")
