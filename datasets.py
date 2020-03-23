import cv2
import itertools
import json
import numpy as np
import os.path
import torch
import glob

class CapIter:
    def __init__(self, cap, n_frames):
        self.cap = cap
        self.n_frames = n_frames
        self.i = 0
    def __iter__(self):
        return self
    def __next__(self):
        ok, frame = self.cap.read()
        if (ok and self.i < self.n_frames):
            self.i += 1
            return frame
        else:
            raise StopIteration

class DeepfakeDataset(torch.utils.data.Dataset):
    def __init__(self, folders, n_frames=float("inf"), train=True):
        self.n_frames = n_frames
        self.videos = []
        self.train = train
        for folder in folders:
            if (train):
                with open(os.path.join(folder, 'metadata.json')) as f:
                    videos = json.load(f)
                    videos = [(os.path.join(folder, video), metadata) for (video, metadata) in videos.items()]
                    self.videos += videos
            else:
                self.videos = glob.glob(folder+"/*")
    def __process_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.tensor(frame)
        frame = frame.permute(2, 0, 1)
        frame = frame / 255.
        return frame
    def __getitem__(self, n):
        if (self.train):
            (video, metadata) = self.videos[n]
        else:
            video = self.videos[n]
        cap = cv2.VideoCapture(video)
        it = CapIter(cap, self.n_frames)
        frames = list(map(self.__process_frame, it))
        cap.release()
        if (self.train):
            return (torch.stack(frames), metadata['label'])
        else:
            return torch.stack(frames)
    def __len__(self):
        return len(self.videos)
