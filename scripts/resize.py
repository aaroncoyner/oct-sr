import os
import cv2
from tqdm import tqdm


def down_sample(image_path):
	img = cv2.imread(image_path)
	resized = cv2.resize(img, (800, 800))
	return resized


src = '../datasets/c_down_sampled/'
dest = '../datasets/d_resized/'
os.makedirs(dest, exist_ok=True)

for f_name in tqdm(os.listdir(src)):
	if not f_name.startswith('.'):
		img = down_sample(src + f_name)
		cv2.imwrite(dest + f_name, img)
