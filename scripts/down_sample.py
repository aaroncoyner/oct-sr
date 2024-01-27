import os
import cv2
from tqdm import tqdm

def down_sample(image_path):
    img = cv2.imread(image_path)
    down_sampled = img[::8, ::8, :]
    
    return down_sampled


src = '../datasets/b_cropped/'
dest = '../datasets/c_down_sampled/'
os.makedirs(dest, exist_ok=True)

for f_name in tqdm(os.listdir(src)):
	img = down_sample(src + f_name)
	cv2.imwrite(dest + f_name, img)
