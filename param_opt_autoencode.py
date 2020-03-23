# import matplotlib.pyplot as plt
import datetime

from models import Autoencoder
from datasets import DeepfakeDataset
import train_autoencoder
import test_autoencoder

start_time = datetime.datetime.now()
print(f"train_encoder start time: {str(start_time)}")
for offset in range(0, 9, 2):
	train_err, train_exec_time = train_autoencoder.train_autoencoder(n_out_channels1=8+offset, n_out_channels2=8+offset, n_out_channels3=4+offset, kernel_size=5)
	test_err, test_exec_time = test_autoencoder.test_autoencoder()


