import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
import datetime

from models import Autoencoder,Classifier
from datasets import DeepfakeDataset

TRAIN_FOLDERS = [
    f'train/dfdc_train_part_{random.randint(0,49)}',
]

AUTOENCODER = 'autoencoder_H00M00S36_03-25-20.pt'

batch_size = 1
num_epochs = 5
epoch_size = 100
n_frames = 30
n_features = 1000
n_head = 8
n_layers = 6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

autoencoder = Autoencoder(n_out_channels1=10, n_out_channels2=10, n_out_channels3=6, kernel_size=5)
autoencoder.load_state_dict(torch.load(AUTOENCODER))
autoencoder.to(device)
autoencoder.eval()

model = Classifier(n_features, n_head, n_layers)
model = model.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

start_time = datetime.datetime.now()
print(f'start time: {str(start_time)}')
print(f'using device: {device}')

train_dataset = DeepfakeDataset(TRAIN_FOLDERS, n_frames=n_frames, device=device)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
for epoch in range(num_epochs):
    epoch_start_time = datetime.datetime.now()
    for i, batch in enumerate(dataloader):
        if i * batch_size >= epoch_size:
            break
        data, labels = batch
        data = data.to(device)
        n_videos = data.shape[0]
        seq_length = data.shape[1]
        with torch.no_grad():
            data = data.reshape(n_videos * seq_length, data.shape[2], data.shape[3], data.shape[4])
            encoding = autoencoder.encoder(data)
            encoding = encoding.reshape(n_videos, seq_length, -1)
            encoding = encoding.permute(1, 0, 2) # convert to seq, batch, features
            encoding = encoding[:,:,:n_features] # XXX
        optimizer.zero_grad()
        output = model(encoding)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    epoch_end_time = datetime.datetime.now()
    epoch_exec_time = epoch_end_time - epoch_start_time
    print(f'epoch: {epoch}, loss: {loss}, executed in: {str(epoch_exec_time)}')
end_time = datetime.datetime.now()
print(f"end time: {str(end_time)}")
exec_time = end_time - start_time
print(f"executed in: {str(exec_time)}")
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
torch.save(model.state_dict(), f'classifier_{end_time.isoformat()}.pt')
