import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
import datetime

from models import Autoencoder
from datasets import DeepfakeDataset

train_folders = [
    f'train/dfdc_train_part_{random.randint(0,49)}',
]

def train_autoencoder(n_out_channels1=16, n_out_channels2=16, n_out_channels3=8, kernel_size=5):
    train_dataset = DeepfakeDataset(train_folders, n_frames=1) # only load the first frame of every video

    # model = Autoencoder().cuda()
    model = Autoencoder(n_out_channels1=n_out_channels1, n_out_channels2=n_out_channels2, n_out_channels3=n_out_channels3, kernel_size=kernel_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    num_epochs = 5
    batch_size = 1

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        epoch_start_time = datetime.datetime.now()
        for i, batch in enumerate(dataloader):
            if i * batch_size >= 10: # only train 100 videos per epoch
                break
            data, _ = batch
            data = data.reshape(data.shape[0] * data.shape[1], data.shape[2], data.shape[3], data.shape[4])
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
        epoch_end_time = datetime.datetime.now()
        exec_time = str(epoch_end_time - epoch_start_time)
        print(f'epoch: {epoch}, loss: {loss}, executed in: {exec_time}')
    now = datetime.datetime.now()
    torch.save(model.state_dict(), f'autoencoder_{now.strftime("H%HM%MS%S_%m-%d-%y")}.pt')
    return (loss, exec_time)

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print(f"train_encoder start time: {str(start_time)}")
    train_autoencoder()
    end_time = datetime.datetime.now()
    print(f"executed in: {str(end_time - start_time)}, finished {str(end_time)}")
