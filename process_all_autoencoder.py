import datetime
import glob
import os
import re
import torch
import torch.nn as nn

from models import Autoencoder
from datasets import DeepfakeDataset

start_time = datetime.datetime.now()
print(f"process_all_autoencoder start time: {str(start_time)}")
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
        if (os.path.exists(write_folder)): # clear dir
            os.system(f"rm {write_folder}/*")
        else:
            os.mkdir(write_folder)

        dataset = DeepfakeDataset(read_folder, n_frames=300, train=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        num_vids = len(glob.glob(read_folder[0]+"/*"))

        for i, batch in enumerate(dataloader):
            file_start_time = datetime.datetime.now()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            data, metadata = batch
            name = re.search(r"train/dfdc_train_part_\d/(.*)\.mp4", metadata["path"][0]).group(1)
            name = os.path.join(write_folder, name + ".pt")
            data = data.to(device)
            for j in range(30):
                frames_to_encode = data[0].narrow_copy(0, j*10, 10)
                new_hidden_frames = model.encode(frames_to_encode)
                if (os.path.exists(name)):
                    previous_hidden_frames = torch.load(name)
                    torch.save(torch.cat((new_hidden_frames, previous_hidden_frames), 0), name)
                    del previous_hidden_frames
                else:
                    torch.save(new_hidden_frames, name)
                del new_hidden_frames
            exec_time = datetime.datetime.now() - file_start_time
            print(f"wrote file {name}, executed in: {exec_time}")

        folder_end_time = datetime.datetime.now()
        exec_time = folder_end_time - folder_start_time
        print(f"executed in: {str(exec_time)}, finished {str(folder_end_time)}")

    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

