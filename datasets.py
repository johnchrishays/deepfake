import cv2
import itertools
import json
import numpy as np
import pandas as pd
import random
import os.path
import time
import torch
from torch import nn
from torch.nn import functional
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
    def __init__(self, folders, encoder, n_frames=None, n_audio_reads=50027, train=True, device=None, cache_folder=None):
        """ n_audio_reads controls the length of the audio sequence: 5000 readings/sec """
        self.n_frames = n_frames
        self.n_audio_reads = n_audio_reads
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
        with torch.no_grad():
            frame = cv2.UMat.get(frame)
            frame = torch.tensor(frame, dtype=torch.float32)
            if self.device:
                frame = frame.to(self.device)
            frame = frame.unsqueeze(0)
            if (frame.size(1) == 1920 and frame.size(2) == 1080):
                frame = frame.permute(0, 3, 1, 2)
            elif (frame.size(2) == 1920 and frame.size(1) == 1080):
                frame = frame.permute(0, 3, 2, 1)
            elif (frame.size(1) == 1280 and frame.size(2) == 720):
                frame = frame.permute(0, 3, 1, 2)
                frame = functional.interpolate(frame, size=(1920, 1080))
            elif (frame.size(2) == 1280 and frame.size(1) == 720):
                frame = frame.permute(0, 3, 2, 1)
                frame = functional.interpolate(frame, size=(1920, 1080))
            else: # if some other size, will be some stretching etc but w/e
                frame = frame.permute(0, 3, 2, 1)
                frame = functional.interpolate(frame, size=(1920, 1080))
            frame = frame / 255.
            encoded = self.encoder(frame)[0]
            encoded = encoded.view(-1)
            return encoded
    def __getitem__(self, n):
        start_time = datetime.datetime.now()
        if (self.train):
            (video, metadata) = self.videos[n]
        else:
            video = self.videos[n]
        # img data
        cache_path = None
        encoded = None
        if self.cache_folder:
            cache_path = os.path.join(self.cache_folder, video) + '.pt'
            if os.path.isfile(cache_path):
                encoded = torch.load(cache_path)
                encoded = encoded[:self.n_frames]
        if encoded is None:
            with torch.no_grad():
                if os.path.islink(video):
                    video = os.readlink(video)
                cap = cv2.VideoCapture(video)
                it = CapIter(cap, self.n_frames)
                try:
                    frames = list(map(self.__process_frame, it))
                except TypeError as e:
                    print(f"Error with {video}:", e)
                    raise
                cap.release()
                try:
                    encoded = torch.stack(frames)
                except RuntimeError as e:
                    print(e, video)
                    raise
            if cache_path:
                d = os.path.dirname(cache_path)
                if not os.path.exists(d):
                    os.makedirs(d)
                torch.save(encoded, cache_path)
        if self.device:
            encoded = encoded.to(self.device)

        # audio data
        if os.path.islink(video):
            video = os.readlink(video)
        wav_file = video[:-4]+".wav"
        if not os.path.exists(wav_file): # create wav file if doesn't exist
            command = f"ffmpeg -i {video} -ar 5000 -vn {wav_file} -y -hide_banner"
            proc = subprocess.call(command, shell=True)
        if os.path.exists(wav_file):
            _, audio_data = wavfile.read(video[:-4]+".wav")
            audio_data = torch.FloatTensor(audio_data) / 2**14
        else:
            audio_data = torch.zeros((self.n_audio_reads))
        if self.n_audio_reads and self.n_audio_reads <= audio_data.size(0):
            audio_data = audio_data[:self.n_audio_reads]
        audio_data = audio_data.unsqueeze(1)

        # return 
        if (self.train):
            label = 0.
            if metadata['label'] == 'FAKE':
                label = 1.
            return (encoded, audio_data, torch.FloatTensor([label]).to(self.device))
        else:
            return (encoded, audio_data)
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
##      dataset manipulations
################################################################################
def real_fake_statistics():
    """ Calculate pct fake videos in the dataset. """
    TRAIN_FOLDERS = [
        f'train/dfdc_train_part_{i}' for i in range(50)
    ]
    num_vids = 0
    num_fake = 0
    num_vids_list = []
    num_fake_list = []
    for folder in TRAIN_FOLDERS:
        folder_num_vids = 0
        folder_num_fake = 0
        with open(os.path.join(folder, 'metadata.json')) as f:
            videos = json.load(f)
            videos = videos.items()
            folder_num_vids += len(videos)
            for _, metadata in videos:
                folder_num_fake += 1 if metadata['label'] == 'FAKE' else 0
        num_vids += folder_num_vids
        num_fake += folder_num_fake
        print(f"{folder}: \n\t num_vids: {folder_num_vids} \n\t num_fake: {folder_num_fake} \n\t pct: {folder_num_fake/folder_num_vids:.2f}")
    print(f"total \n\t num_vids: {num_vids} \n\t num_fake: {num_fake} \n\t pct: {folder_num_fake/folder_num_vids:.2f}")

def frame_size_statistics():
    """ Calculate pct fake videos in the dataset. """
    TRAIN_FOLDERS = [
        f'train/dfdc_train_part_{i}' for i in range(13, 50)
    ]
    num_videos = 0
    frame_size_dict = dict()
    for folder in TRAIN_FOLDERS:
        print(f"folder {folder}")
        with open(os.path.join(folder, 'metadata.json')) as f:
            videos = json.load(f)
            videos = [(os.path.join(folder, video), metadata) for (video, metadata) in videos.items()]
            for video, metadata in videos:
                cap = cv2.VideoCapture(video)
                dims = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                if (dims in frame_size_dict):
                    frame_size_dict[dims] += 1
                else:
                    frame_size_dict[dims] = 1
                num_videos += 1
        for dim in frame_size_dict.keys():
            print(f"running tally: frame size {dim}: {frame_size_dict[dim]/num_videos}%")

def extract_audio():
    """ Writes .wav files of audio for each of the videos in TRAIN_FOLDERS at sample rate of 5000Hz. """
    TRAIN_FOLDERS = [
        f'train/dfdc_train_part_{i}' for i in range(1,50)
        # 'test/test_videos'
    ]
    start_time = datetime.datetime.now()
    print(f'start time: {str(start_time)}')
    for folder in TRAIN_FOLDERS:
        print(f"using folder: {folder}")
        with open(os.path.join(folder, 'metadata.json')) as f:
            videos = json.load(f)
            videos = videos.items()
            for filename, metadata in videos:
                full_fname = os.path.join(folder, filename)
                audio_fname = os.path.join(folder, filename[:-4] + ".wav")
                command = f"ffmpeg -i {full_fname} -ar 5000 -vn {audio_fname} -y -hide_banner -loglevel panic"
                subprocess.call(command, shell=True)
    end_time = datetime.datetime.now()
    print(f"end time: {str(end_time)}")
    exec_time = end_time - start_time
    print(f"executed in: {str(exec_time)}")

def get_max_audioval():
    TRAIN_FOLDERS = [
        f'train/dfdc_train_part_0'
    ]
    max_val = 0
    min_val = 0
    train_dataset = DeepfakeDatasetAudio(TRAIN_FOLDERS)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    for i, batch in enumerate(dataloader):
        data, labels = batch
        print(data.size())
        break
    print(max_val, min_val)

def lower_framerate():
    """ Reduces video framerate from 30fps to 15fps. """
    TRAIN_FOLDERS = [
        f'train/dfdc_train_part_0'
        # 'test/test_videos'
    ]
    start_time = datetime.datetime.now()
    fps = 15
    print(f'start time: {str(start_time)}')
    for folder in TRAIN_FOLDERS:
        print(f"using folder: {folder}")
        with open(os.path.join(folder, 'metadata.json')) as f:
            videos = json.load(f)
            videos = videos.items()
            for filename, metadata in videos:
                full_fname = os.path.join(folder, filename)
                new_fname = os.path.join(folder, filename[:-4] + f"_{fps}fps.mp4")
                command = f"ffmpeg -i {full_fname} -filter:v fps=fps=15 {new_fname} -y -hide_banner -loglevel panic"
                subprocess.call(command, shell=True)
    end_time = datetime.datetime.now()
    print(f"end time: {str(end_time)}")
    exec_time = end_time - start_time
    print(f"executed in: {str(exec_time)}")

def symlink_balanced_dataset(orig_folders, balanced_folder):
    """ Generates new balanced training (~50% real/fake) by simlinking videos to new folder. """
    if os.path.exists(balanced_folder):
        for filename in os.listdir(balanced_folder):
            file_path = os.path.join(balanced_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        os.mkdir(balanced_folder)
    num_real = 0
    num_fake = 0
    with open(os.path.join(balanced_folder,'metadata.json'), 'w') as metadata_file:
        new_metadata_dict = dict()
        for folder in orig_folders:
            print(f"folder: {folder}")
            with open(os.path.join(folder, 'metadata.json')) as f:
                videos = json.load(f)
                videos = videos.items()
                for filename, metadata in videos: 
                    p = np.random.uniform()
                    if metadata['label'] == 'FAKE' and p >= .81:
                        src = os.path.join(folder, filename)
                        dst = os.path.join(balanced_folder, filename)
                        new_metadata_dict[filename] = metadata
                        try:
                            os.symlink(src, dst)
                        except FileExistsError as e:
                            print(f'Error with {filename}: {e}')
                        num_fake += 1
                    if metadata['label'] == 'REAL':
                        src = os.path.join(folder, filename)
                        dst = os.path.join(balanced_folder, filename)
                        new_metadata_dict[filename] = metadata
                        try:
                            os.symlink(src, dst)
                        except FileExistsError as e:
                            print(f'Error with {filename}: {e}')
                        num_real += 1
        json.dump(new_metadata_dict, metadata_file)
    print(f"Num files: {num_real+num_fake}\nPercent real: {num_real/(num_real+num_fake)}")

if __name__ == "__main__":
    # train_inds = np.random.choice(50, 30, replace=False) 
    # test_inds = list(set(range(50)).difference(train_inds))
    # TRAIN_FOLDERS = [
    #     f'train/dfdc_train_part_{i}' for i in train_inds 
    # ]
    # TEST_FOLDERS = [
    #     f'train/dfdc_train_part_{i}' for i in test_inds 
    # ]
    # BALANCED_TRAIN_FOLDER = 'train/balanced'
    # BALANCED_TEST_FOLDER = 'train/balanced_test'
    # symlink_balanced_dataset(TRAIN_FOLDERS, BALANCED_TRAIN_FOLDER)
    # symlink_balanced_dataset(TEST_FOLDERS, BALANCED_TEST_FOLDER)
    video = "train/dfdc_train_part_26/hycdczdeby.mp4"
    if os.path.islink(video):
        video = os.readlink(video)
    wav_file = video[:-4]+".wav"
    if not os.path.exists(wav_file): # create wav file if doesn't exist
        command = f"ffmpeg -i {video} -ar 5000 -vn {wav_file} -y -hide_banner"
        proc = subprocess.call(command, shell=True)
    if os.path.exists(wav_file):
        _, audio_data = wavfile.read(video[:-4]+".wav")
        audio_data = torch.FloatTensor(audio_data) / 2**14
        print(audio_data.size())
    else:
        audio_data = torch.zeros((50027 if n_audio_reads == None else n_audio_reads))
        print(audio_data.size())
        


