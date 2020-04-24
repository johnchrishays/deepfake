import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
import datetime

from models import Autoencoder,Classifier
from datasets import EncodedDeepfakeDataset

TRAIN_FOLDERS = [
    #f'train/dfdc_train_part_{random.randint(0,49)}',
    f'train/dfdc_train_part_0',
]

AUTOENCODER = 'autoencoder_H18M05S37_04-23-20.pt'

batch_size = 10
num_epochs = 30
epoch_size = float("inf")
n_frames = None
n_features = 3600
n_head = 8
n_layers = 6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

autoencoder = Autoencoder()
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

train_dataset = EncodedDeepfakeDataset(TRAIN_FOLDERS, autoencoder.encoder, n_frames=n_frames, device=device, cache_folder="encode_cache")
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
for epoch in range(num_epochs):
    epoch_start_time = datetime.datetime.now()
    for i, batch in enumerate(dataloader):
        if i * batch_size >= epoch_size:
            break
        data, labels = batch
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        print('.', end='', flush=True)
    print()
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
