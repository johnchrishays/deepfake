import glob
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
import datetime

from models2 import Autoencoder,Classifier
from datasets import EncodedDeepfakeDataset

VAL_FOLDERS = [
    f'train/balanced_test',
]

AUTOENCODER = 'autoencoder_H18M05S37_04-23-20.pt'
CLASSIFIER = max(glob.glob('./classifier2_*.pt'), key=os.path.getctime)
# CLASSIFIER = 'classifier_2020-04-23T21:48:05.265500.pt'

test_size = 1000
batch_size = 10
n_frames = 30
n_vid_features = 3600
n_aud_features = 1
n_head = 3
n_layers = 3
dim_feedforward = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

autoencoder = Autoencoder()
autoencoder.load_state_dict(torch.load(AUTOENCODER))
autoencoder.to(device)
autoencoder.eval()

model = Classifier(n_vid_features, n_aud_features, n_head, n_layers, dim_feedforward)
model.load_state_dict(torch.load(CLASSIFIER))
model = model.to(device)
model.eval()

start_time = datetime.datetime.now()
print(f'start time: {str(start_time)}')
print(f'using device: {device}')

count = 0
count_wrong = 0

val_dataset = EncodedDeepfakeDataset(VAL_FOLDERS, autoencoder.encoder, n_frames=n_frames, n_audio_reads=576, device=device, cache_folder="encode_cache")
dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
for i, batch in enumerate(dataloader):
    if i * batch_size >= test_size:
        break
    video_data, audio_data, labels = batch
    video_data = video_data.to(device)
    audio_data = audio_data.to(device)
    video_data = video_data.permute(1,0,2)
    audio_data = audio_data.permute(1,0,2)
    with torch.no_grad():
        output = model(video_data, audio_data)
        output = torch.sigmoid(output)
        output = output.round()
        n_wrong = (labels - output).abs().sum()
        count_wrong += n_wrong
        count += labels.shape[0]
    print('.', end='', flush=True)

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
