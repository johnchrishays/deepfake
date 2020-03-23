import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import random
import glob
import datetime

from models import Autoencoder
from datasets import DeepfakeDataset

start_time = datetime.datetime.now()
print(f"train_encoder start time: {str(start_time)}")
with torch.no_grad():
    train_folders = [
        f'train/dfdc_train_part_{random.randint(0,49)}',
    ]
    # train_dataset = DeepfakeDataset(train_folders)
    train_dataset = DeepfakeDataset(train_folders, n_frames=1) # only load the first frame of every video

    # model = Autoencoder().cuda()
    model = Autoencoder()

    latest_ae = max(glob.glob('./*.pt'), key=os.path.getctime)
    model.load_state_dict(torch.load(latest_ae))

    num_vids = len(glob.glob(train_folders[0]+"/*"))
    ind = random.randint(0,num_vids-1)

    data, _ = train_dataset[ind]

    in_frame = data[0,:,:,:]
    plt.title('original')
    plt.imshow(in_frame.permute(1, 2, 0))
    plt.savefig('original.png')

    out = model(data)

    out_frame = out[0,:,:,:]
    plt.title('decoded')
    plt.imshow(out_frame.permute(1, 2, 0))
    plt.savefig('decoded.png')

    hidden = model.encode(data)
    print(f"Input size: [3, 1920, 1080]")
    print(f"Hidden size: {hidden.size()}")

    hidden_frame = out[0,0,:,:]
    plt.title('encoded')
    plt.imshow(hidden_frame)
    plt.savefig('encoded.png')
end_time = datetime.datetime.now()
print(f"executed in: {str(end_time - start_time)}, finished {str(end_time)}")
