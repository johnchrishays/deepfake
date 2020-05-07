import cv2
import json
import os

folders = [
    f'train/dfdc_train_part_{i}' for i in range(50)
]

def vidcmp(video1, video2):
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)
    ok1, frame1 = cap1.read()
    while ok1:
        ok2, frame2 = cap2.read()
        if not ok2:
            return False
        diff = frame1 != frame2
        if diff.any():
            return False
        ok1, frame1 = cap1.read()
    ok2, frame2 = cap2.read()
    if ok2:
        return False
    print("same")
    return True

i = 0
for folder in folders:
    with open(os.path.join(folder, 'metadata.json')) as f:
        videos = json.load(f)
        videos = [(os.path.join(folder, video), metadata) for (video, metadata) in videos.items()]
        for (video, metadata) in videos:
            link = None
            if metadata['label'] == 'REAL':
                link = f'clean/real/{os.path.basename(video)}'
            else:
                original = f'{folder}/{metadata["original"]}'
                if not vidcmp(video, original):
                    link = f'clean/fake/{os.path.basename(video)}'
                else:
                    print('here')
            if link is not None and not os.path.isfile(link):
                os.symlink(f'../../{video}', link)
            i += 1
            if i % 100 == 0:
                print(i)
