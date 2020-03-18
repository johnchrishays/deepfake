import cv2
import itertools
import json
import numpy as np
import os.path
import torch

class DeepfakeDataset(torch.utils.data.Dataset):
    def __init__(self, folders, n_frames=float("inf")):
        self.n_frames = n_frames
        self.videos = []
        for folder in folders:
            with open(os.path.join(folder, 'metadata.json')) as f:
                videos = json.load(f)
                videos = [(os.path.join(folder, video), metadata) for (video, metadata) in videos.items()]
                self.videos += videos

    def __getitem__(self, n):
        (video, metadata) = self.videos[n]
        cap = cv2.VideoCapture(video)
        frames = []
        for i in itertools.count():
            if i >= self.n_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.tensor(frame)
            frame = frame.permute(2, 0, 1)
            frame = frame / 255.
            frames.append(frame)
        cap.release()
        return (torch.stack(frames), metadata['label'])

    def __len__(self):
        return len(self.videos)
