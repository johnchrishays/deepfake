import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
import datetime

from models import Autoencoder,Classifier
from datasets import EncodedDeepfakeDataset

VAL_FOLDERS = [
    f'train/dfdc_train_part_1',
]

AUTOENCODER = 'autoencoder_H18M05S37_04-23-20.pt'
CLASSIFIER = 'classifier_2020-04-23T21:48:05.265500.pt'

test_size = 100
batch_size = 1
n_frames = None
n_vid_features = 3600
n_aud_features = 1
n_head = 8
n_layers = 6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

autoencoder = Autoencoder()
autoencoder.load_state_dict(torch.load(AUTOENCODER))
autoencoder.to(device)
autoencoder.eval()

model = Classifier(n_vid_features, n_aud_features, n_head, n_layers)
model = model.to(device)
model.eval()

start_time = datetime.datetime.now()
print(f'start time: {str(start_time)}')
print(f'using device: {device}')

count = 0
count_wrong = 0
count_real = 0

val_dataset = EncodedDeepfakeDataset(VAL_FOLDERS, autoencoder.encoder, n_frames=n_frames, n_audio_reads=576, device=device, cache_folder="encode_cache")
dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
for i, batch in enumerate(dataloader):
    if i * batch_size >= test_size:
        break
    video_data, audio_data, labels = batch
    video_data = video_data.to(device)
    audio_data = audio_data.to(device)
    with torch.no_grad():
        output = model(video_data, audio_data)
        print(output)
        output = torch.sigmoid(output)
        output = output.round()
        n_wrong = (labels - output).abs().sum()
        count_real += (output == 0).sum()
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
print(f"# of REAL guesses: {count_real}")
