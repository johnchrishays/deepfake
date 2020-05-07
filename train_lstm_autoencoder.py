import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
import datetime

from models import LstmAutoencoder
from datasets import DeepfakeDatasetAudio # 441344
LSTM_SIZE = 64


def train_lstm_autoencoder():
    start_time = datetime.datetime.now()
    print(f"train_lstm_autoencoder start time: {str(start_time)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print('Using device:', device)

    BATCH_SIZE = 10
    ITERATIONS = 1000
    SEQ_LENGTH = 50 # 441344
    LSTM_SIZE = 16

    model = LstmAutoencoder(device, BATCH_SIZE, SEQ_LENGTH, LSTM_SIZE)
    model = model.to(device)
    model.train()

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(),lr=0.1)
    
    epoch_loss = 0

    TRAIN_FOLDERS = [
        'train/dfdc_train_part_0',
    ]


    train_dataset = DeepfakeDatasetAudio(TRAIN_FOLDERS, device=device)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    num_epochs = 25
    epoch_size = 100
    i = 0
    for epoch in range(num_epochs):
        epoch_start_time = datetime.datetime.now()
        epoch_loss = 0
        for i, batch in enumerate(dataloader):
            if i * BATCH_SIZE > epoch_size:
                break
            data, _ = batch
            data = data.to(device)
            data = data.permute(1,0)
            data.unsqueeze_(2) #Add "feature" third axis (num_features=1)
            for j in range(0,100,5):
                data_slice = data[j:j+SEQ_LENGTH,...]
                optimizer.zero_grad()
                output = model(data_slice)
                loss = loss_fn(output, data_slice)
                # print(f"loss: {loss.item()}")
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            print('.', end='', flush=True)
        print(f"\nEpoch {epoch} loss: {epoch_loss/(i * BATCH_SIZE)}")


train_lstm_autoencoder()