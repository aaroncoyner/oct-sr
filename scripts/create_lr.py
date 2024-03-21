import os
import cv2
from tqdm import tqdm

def down_sample(img, start, scale):
    down_sampled = img[start::scale * 2, start::scale, :]
    return down_sampled

def resize(img):
    resized = cv2.resize(img, (800, 800))
    return resized

MODEL = 'pix2pix'
SCALE = 4
SRC = '../datasets/cropped/'
dest = f'../datasets/lr-{MODEL}/'


os.makedirs(dest, exist_ok=True)

for f_name in tqdm(os.listdir(SRC)):
    img = cv2.imread(os.path.join(SRC, f_name))
    for start in range(SCALE):
        save_name = os.path.splitext(f_name)[0] + '-start' + str(start + 1) + '.png'
        down_sampled = down_sample(img, start, SCALE)
        if MODEL == 'pix2pix':
            down_sampled = resize(down_sampled)
        cv2.imwrite(os.path.join(dest, save_name), down_sampled)
