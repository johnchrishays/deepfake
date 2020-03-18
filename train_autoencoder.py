import time
import torch
import torch.nn as nn
import torch.optim as optim

from models import Autoencoder
from datasets import DeepfakeDataset

train_folders = [
    'train/dfdc_train_part_0',
]
# train_dataset = DeepfakeDataset(train_folders)
train_dataset = DeepfakeDataset(train_folders, n_frames=1) # only load the first frame of every video

# model = Autoencoder().cuda()
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
num_epochs = 5
batch_size = 1

# dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
        if i * batch_size >= 100: # only train 100 videos per epoch
            break
        data, _ = batch
        data = data.reshape(data.shape[0] * data.shape[1], data.shape[2], data.shape[3], data.shape[4])
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
    print(f'epoch: {epoch}, loss: {loss}')

torch.save(model.state_dict(), f'autoencoder{time.time()}.pt')
