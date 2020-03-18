import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from models import Autoencoder
from datasets import DeepfakeDataset

with torch.no_grad():
    train_folders = [
        'train/dfdc_train_part_0',
    ]
    # train_dataset = DeepfakeDataset(train_folders)
    train_dataset = DeepfakeDataset(train_folders, n_frames=1) # only load the first frame of every video

    # model = Autoencoder().cuda()
    model = Autoencoder()

    model.load_state_dict(torch.load('autoencoder1583450281.8973327.pt'))

    data, _ = train_dataset[0]

    in_frame = data[0,:,:,:]
    plt.title('original')
    plt.imshow(in_frame.permute(1, 2, 0))
    plt.savefig('original.png')

    out = model(data)

    out_frame = out[0,:,:,:]
    plt.title('encoded')
    plt.imshow(out_frame.permute(1, 2, 0))
    plt.savefig('encoded.png')

    hidden = model.encode(data)
    print(hidden.size())

    hidden_frame = out[0,0,:,:]
    plt.title('hidden')
    plt.imshow(hidden_frame)
    plt.savefig('hidden.png')
