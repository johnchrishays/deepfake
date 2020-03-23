import cv2
import itertools
import json
import numpy as np
import os.path
import torch

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
    def __init__(self, folders, n_frames=float("inf")):
        self.n_frames = n_frames
        self.videos = []
        for folder in folders:
            with open(os.path.join(folder, 'metadata.json')) as f:
                videos = json.load(f)
                videos = [(os.path.join(folder, video), metadata) for (video, metadata) in videos.items()]
                self.videos += videos
    def __process_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.tensor(frame)
        frame = frame.permute(2, 0, 1)
        frame = frame / 255.
        return frame
    def __getitem__(self, n):
        (video, metadata) = self.videos[n]
        cap = cv2.VideoCapture(video)
        it = CapIter(cap, self.n_frames)
        frames = list(map(self.__process_frame, it))
        cap.release()
        return (torch.stack(frames), metadata['label'])
    def __len__(self):
        return len(self.videos)
