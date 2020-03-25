import datetime
import glob
import os
import re
import torch
import torch.nn as nn

from models import Autoencoder
from datasets import DeepfakeDataset

start_time = datetime.datetime.now()
print(f"test_encoder start time: {str(start_time)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

with torch.no_grad():
    model = Autoencoder(n_out_channels1=10, n_out_channels2=10, n_out_channels3=6, kernel_size=5)
    model = model.to(device)
    criterion = nn.MSELoss()

    latest_ae = max(glob.glob('./*.pt'), key=os.path.getctime)
    model.load_state_dict(torch.load(latest_ae))

    for folder_ind in range(1):
        read_folder = [f'train/dfdc_train_part_{folder_ind}']
        folder_start_time = datetime.datetime.now()
        print(f"{read_folder} start time: {str(folder_start_time)}")
        write_folder = f'train_cae_feature_vectors/dfdc_train_part_{folder_ind}'
        try: # make the dir if it doesn't already exist
            os.mkdir(write_folder)
        except FileExistsError:
            pass

        dataset = DeepfakeDataset(read_folder, n_frames=50, train=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        num_vids = len(glob.glob(read_folder[0]+"/*"))

        for i, batch in enumerate(dataloader):
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            if (i > 2):
                break
            data, metadata = batch
            name = re.search(r"train/dfdc_train_part_\d/(.*)\.mp4", metadata["path"][0])
            name = name.group(1)
            data = data.to(device)
            hidden = model.encode(data[0])
            torch.save(hidden, os.path.join(write_folder, name + ".pt"))


        folder_end_time = datetime.datetime.now()
        exec_time = folder_end_time - folder_start_time
        print(f"executed in: {str(exec_time)}, finished {str(folder_end_time)}")

    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

