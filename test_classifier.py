import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
import datetime

from models import Autoencoder,Classifier
from datasets import DeepfakeDataset

VAL_FOLDERS = [
    f'train/dfdc_train_part_{random.randint(0,49)}',
]

AUTOENCODER = 'autoencoder_H00M00S36_03-25-20.pt'
CLASSIFIER = 'classifier_2020-04-23T00:20:01.482689.pt'

batch_size = 1
epoch_size = float("inf")
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
model.eval()

start_time = datetime.datetime.now()
print(f'start time: {str(start_time)}')
print(f'using device: {device}')

count = 0
count_wrong = 0

val_dataset = DeepfakeDataset(VAL_FOLDERS, n_frames=n_frames, device=device)
dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
for i, batch in enumerate(dataloader):
    if i * batch_size >= epoch_size:
        print(f'here: {i} {batch_size}')
        print(f'here: {i * batch_size}')
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
        output = model(encoding)
        output = output.round()
        n_wrong = (labels - output).abs().sum()
        count_wrong += n_wrong
        count += labels.shape[0]

end_time = datetime.datetime.now()
print(f"end time: {str(end_time)}")
exec_time = end_time - start_time
print(f"executed in: {str(exec_time)}")

count_right = count - count_wrong

print()
print(f"total: {count}")
print(f"correct: {count_right}")
print(f"incorrect: {count_wrong}")
print(f"accuracy: {count_right / count}")
