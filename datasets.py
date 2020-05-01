import cv2
import itertools
import json
import numpy as np
import pandas as pd
import random
import os.path
import torch
import glob
import datetime
import subprocess
from scipy.io import wavfile

class CapIter:
    def __init__(self, cap, n_frames=None):
        self.cap = cap
        self.n_frames = n_frames
        self.i = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self.n_frames != None and self.i == self.n_frames:
            raise StopIteration
        ok, frame = self.cap.read()
        if not ok:
            raise StopIteration
        self.i += 1
        return frame

class DeepfakeDataset(torch.utils.data.Dataset):
    def __init__(self, folders, n_frames=None, train=True, device=None):
        self.n_frames = n_frames
        self.videos = []
        self.train = train
        self.device = device
        for folder in folders:
            if (train):
                with open(os.path.join(folder, 'metadata.json')) as f:
                    videos = json.load(f)
                    videos = [(os.path.join(folder, video), metadata) for (video, metadata) in videos.items()]
                    self.videos += videos
            else:
                self.videos += glob.glob(folder+"/*")
    def __process_frame(self, frame):
        frame = cv2.UMat(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        frame = torch.tensor(cv2.UMat.get(frame))
        if self.device:
            frame.to(self.device)
        frame = frame.permute(2, 0, 1)
        frame = frame / 255.
        return frame
    def __getitem__(self, n):
        if (self.train):
            (video, metadata) = self.videos[n]
        else:
            video = self.videos[n]
        with torch.no_grad():
            cap = cv2.VideoCapture(video)
            it = CapIter(cap, self.n_frames)
            frames = list(map(self.__process_frame, it))
            cap.release()
            if (self.train):
                label = 0.
                if metadata['label'] == 'FAKE':
                    label = 1.
                return (torch.stack(frames), torch.FloatTensor([label]).to(self.device))
            else:
                return torch.stack(frames)
    def __len__(self):
        return len(self.videos)

class EncodedDeepfakeDataset(torch.utils.data.Dataset):
    def __init__(self, folders, encoder, n_frames=None, train=True, device=None, cache_folder=None):
        self.n_frames = n_frames
        self.videos = []
        self.train = train
        self.device = device
        self.cache_folder = cache_folder
        self.encoder = encoder
        for folder in folders:
            if (train):
                with open(os.path.join(folder, 'metadata.json')) as f:
                    videos = json.load(f)
                    videos = [(os.path.join(folder, video), metadata) for (video, metadata) in videos.items()]
                    self.videos += videos
            else:
                self.videos += glob.glob(folder+"/*")
    def __process_frame(self, frame):
        frame = cv2.UMat(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        frame = torch.tensor(cv2.UMat.get(frame))
        if self.device:
            frame = frame.to(self.device)
        frame = frame.permute(2, 0, 1)
        frame = frame / 255.
        frame = frame.unsqueeze(0)
        encoded = self.encoder(frame)[0]
        encoded = encoded.view(-1)
        return encoded
    def __getitem__(self, n):
        start_time = datetime.datetime.now()
        if (self.train):
            (video, metadata) = self.videos[n]
        else:
            video = self.videos[n]
        cache_path = None
        encoded = None
        if self.cache_folder:
            cache_path = os.path.join(self.cache_folder, video) + '.pt'
            if os.path.isfile(cache_path):
                encoded = torch.load(cache_path)
                encoded = encoded[:self.n_frames]
        if encoded is None:
            with torch.no_grad():
                cap = cv2.VideoCapture(video)
                it = CapIter(cap, self.n_frames)
                frames = list(map(self.__process_frame, it))
                cap.release()
                encoded = torch.stack(frames)
            if cache_path:
                d = os.path.dirname(cache_path)
                if not os.path.exists(d):
                    os.makedirs(d)
                torch.save(encoded, cache_path)
        if self.device:
            encoded = encoded.to(self.device)
        if (self.train):
            label = 0.
            if metadata['label'] == 'FAKE':
                label = 1.
            return (encoded, torch.FloatTensor([label]).to(self.device))
        else:
            return encoded
    def __len__(self):
        return len(self.videos)

class DeepfakeDatasetAudio(torch.utils.data.Dataset):
    def __init__(self, folders, train=True, device=None):
        self.videos = []
        self.train = train
        self.device = device
        for folder in folders:
            if (train):
                with open(os.path.join(folder, 'metadata.json')) as f:
                    videos = json.load(f)
                    videos = [(os.path.join(folder, video), metadata) for (video, metadata) in videos.items()]
                    self.videos += videos
            else:
                self.videos += glob.glob(folder+"/*")
    def __getitem__(self, n):
        if (self.train):
            (video, metadata) = self.videos[n]
        else:
            video = self.videos[n]
        with torch.no_grad():
            rate, data = wavfile.read(video[:-4]+".wav")
            if (self.train):
                label = 0.
                if metadata['label'] == 'FAKE':
                    label = 1.
                return (torch.FloatTensor(data) / 2**14, torch.FloatTensor([label]).to(self.device))
            else:
                return torch.FloatTensor(data) / 2**14
    def __len__(self):
        return len(self.videos)

################################################################################
##      dataset statistics
################################################################################
# if __name__ == "__main__":
#     TRAIN_FOLDERS = [
#         # f'train/dfdc_train_part_{i}' for i in range(50)
#         'test/test_videos'
#     ]
#     num_vids = 0
#     num_fake = 0
#     num_vids_list = []
#     num_fake_list = []
#     for folder in TRAIN_FOLDERS:
#         folder_num_vids = 0
#         folder_num_fake = 0
#         with open(os.path.join(folder, 'metadata.json')) as f:
#             videos = json.load(f)
#             videos = videos.items()
#             folder_num_vids += len(videos)
#             for _, metadata in videos:
#                 folder_num_fake += 1 if metadata['label'] == 'FAKE' else 0
#         num_vids += folder_num_vids
#         num_fake += folder_num_fake
#         print(f"{folder}: \n\t num_vids: {folder_num_vids} \n\t num_fake: {folder_num_fake} \n\t pct: {folder_num_fake/folder_num_vids:.2f}")
#     print(f"total \n\t num_vids: {num_vids} \n\t num_fake: {num_fake} \n\t pct: {folder_num_fake/folder_num_vids:.2f}")

if __name__ == "__main__":
    TRAIN_FOLDERS = [
        f'train/dfdc_train_part_0'
        # 'test/test_videos'
    ]
    for folder in TRAIN_FOLDERS:
        print(f"using folder: {folder}")
        with open(os.path.join(folder, 'metadata.json')) as f:
            videos = json.load(f)
            videos = videos.items()
            for filename, metadata in videos:
                full_fname = os.path.join(folder, filename)
                audio_fname = os.path.join(folder, filename[:-4] + ".wav")
                command = f"ffmpeg -i {full_fname} -vn {audio_fname} -y -ar 500 -hide_banner -loglevel panic"
                subprocess.call(command, shell=True)

# if __name__ == "__main__":
#     TRAIN_FOLDERS = [
#         f'train/dfdc_train_part_0'
#     ]
#     max_val = 0
#     min_val = 0
#     train_dataset = DeepfakeDatasetAudio(TRAIN_FOLDERS)
#     dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
#     for i, batch in enumerate(dataloader):
#         data, labels = batch
#         print(data.size())
#         break
#     print(max_val, min_val)
    


