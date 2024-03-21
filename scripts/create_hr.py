import os
import shutil
import cv2
from tqdm import tqdm


SCALE = 4
SRC = '../datasets/cropped/'
DEST = f'../datasets/hr'


os.makedirs(DEST, exist_ok=True)

for f_name in tqdm(os.listdir(SRC)):
	if not f_name.startswith('.'):
		for i in range(1, SCALE + 1):
			save_name = os.path.join(DEST, os.path.splitext(f_name)[0] + '-start' + str(i) + '.png')
			shutil.copyfile(os.path.join(SRC, f_name), save_name)
