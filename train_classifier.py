import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
import datetime

from models import Autoencoder,Classifier
from datasets import EncodedDeepfakeDataset

TRAIN_FOLDERS = [
    # f'train/dfdc_train_part_30',
    'train/balanced'
]

AUTOENCODER = 'autoencoder_H18M05S37_04-23-20.pt'

batch_size = 10
num_epochs = 20
epoch_size = 1000
n_frames = 30
n_vid_features = 3600
n_aud_features = 1
n_head = 3
n_layers = 3
dim_feedforward = 128
lr = 0.01

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

autoencoder = Autoencoder()
autoencoder.load_state_dict(torch.load(AUTOENCODER))
autoencoder.to(device)
autoencoder.eval()

model = Classifier(n_vid_features, n_aud_features, n_head, n_layers, dim_feedforward)
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

start_time = datetime.datetime.now()
print(f'start time: {str(start_time)}')
print(f'using device: {device}')

train_dataset = EncodedDeepfakeDataset(TRAIN_FOLDERS, autoencoder.encoder, n_frames=n_frames, n_audio_reads=576, device=device, cache_folder="encode_cache", n_videos=epoch_size)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
for epoch in range(num_epochs):
    epoch_start_time = datetime.datetime.now()
    epoch_loss = 0
    for i, batch in enumerate(dataloader):
        video_data, audio_data, labels = batch
        video_data = video_data.to(device)
        audio_data = audio_data.to(device)
        video_data = video_data.permute(1,0,2)
        audio_data = audio_data.permute(1,0,2)
        optimizer.zero_grad()
        output = model(video_data, audio_data)
        loss = criterion(output, labels)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        print('.', end='', flush=True)
    epoch_end_time = datetime.datetime.now()
    epoch_exec_time = epoch_end_time - epoch_start_time
    print(f'\nepoch: {epoch}, loss: {epoch_loss/(epoch_size/batch_size)}, executed in: {str(epoch_exec_time)}')
end_time = datetime.datetime.now()
print(f"end time: {str(end_time)}")
exec_time = end_time - start_time
print(f"executed in: {str(exec_time)}")
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
torch.save(model.state_dict(), f'classifier2_{end_time.isoformat()}.pt')
