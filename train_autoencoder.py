import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import datetime

from models import Autoencoder
from datasets import DeepfakeDataset

num_folders = 10
train_folder_inds = np.random.randint(0,49,num_folders)
train_folders = [
    f'train/dfdc_train_part_{train_folder_inds[0]}' for i in range(num_folders)
]

def train_autoencoder(epoch_size=100):
    start_time = datetime.datetime.now()
    print(f"train_encoder start time: {str(start_time)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print('Using device:', device)
    train_dataset = DeepfakeDataset(train_folders, n_frames=1, device=device) # only load the first frame of every video

    model = Autoencoder()
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    num_epochs = 10
    batch_size = 1

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        epoch_start_time = datetime.datetime.now()
        for i, batch in enumerate(dataloader):
            if i * batch_size >= epoch_size:
                break
            data, _ = batch
            data = data.to(device)
            data = data.reshape(data.shape[0] * data.shape[1], data.shape[2], data.shape[3], data.shape[4])
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
        epoch_end_time = datetime.datetime.now()
        exec_time = epoch_end_time - epoch_start_time
        print(f'epoch: {epoch}, loss: {loss}, executed in: {str(exec_time)}')
    end_time = datetime.datetime.now()
    torch.save(model.state_dict(), f'autoencoder_{end_time.strftime("H%HM%MS%S_%m-%d-%y")}.pt')
    exec_time = end_time - start_time
    print(f"train_encoder executed in: {str(exec_time)}, end time: {str(end_time)}")
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
    return (loss.item(), exec_time)

if __name__ == "__main__":
    train_autoencoder()
