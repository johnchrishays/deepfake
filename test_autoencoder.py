import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import random
import glob
import datetime
import numpy as np

from models import Autoencoder
from datasets import DeepfakeDataset

def test_autoencoder():
    with torch.no_grad():
        test_folder = [
            'test/test_videos',
        ]
        test_dataset = DeepfakeDataset(test_folder, n_frames=1, train=False)

        model = Autoencoder()
        criterion = nn.MSELoss()

        latest_ae = max(glob.glob('./*.pt'), key=os.path.getctime)
        model.load_state_dict(torch.load(latest_ae))

        num_vids = len(glob.glob(test_folder[0]+"/*"))
        sample = np.random.choice(a=num_vids, size=40, replace=False)
        
        loss = 0
        for ind in sample:
            data = test_dataset[ind]
            out = model(data)
            loss += criterion(out, data).item()

    return loss / num_vids
        
if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print(f"train_encoder start time: {str(start_time)}")
    print(test_autoencoder())
    end_time = datetime.datetime.now()
    print(f"executed in: {str(end_time - start_time)}, finished {str(end_time)}")

# with torch.no_grad():
#     train_folders = [
#         f'train/dfdc_train_part_{random.randint(0,49)}',
#     ]
#     train_dataset = DeepfakeDataset(train_folders, n_frames=1) # only load the first frame of every video

#     # model = Autoencoder().cuda()
#     model = Autoencoder()

#     latest_ae = max(glob.glob('./*.pt'), key=os.path.getctime)
#     model.load_state_dict(torch.load(latest_ae))

#     num_vids = len(glob.glob(train_folders[0]+"/*"))
#     ind = random.randint(0,num_vids-1)

#     data, _ = train_dataset[ind]

#     in_frame = data[0,:,:,:]
#     plt.title('original')
#     plt.imshow(in_frame.permute(1, 2, 0))
#     plt.savefig('original.png')

#     out = model(data)

#     out_frame = out[0,:,:,:]
#     plt.title('decoded')
#     plt.imshow(out_frame.permute(1, 2, 0))
#     plt.savefig('decoded.png')

#     hidden = model.encode(data)
#     print(f"Input size: [3, 1920, 1080]")
#     print(f"Hidden size: {hidden.size()}")

#     hidden_frame = out[0,0,:,:]
#     plt.title('encoded')
#     plt.imshow(hidden_frame)
#     plt.savefig('encoded.png')
# end_time = datetime.datetime.now()
# print(f"executed in: {str(end_time - start_time)}, finished {str(end_time)}")
