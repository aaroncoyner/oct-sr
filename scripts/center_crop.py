import os
import cv2
from tqdm import tqdm


def center_crop_image(image_path):
	img = cv2.imread(image_path)
	height, width, _ = img.shape
	center_x, center_y = width // 2, height // 2
	top_left_x = max(center_x - height // 2, 0)
	top_left_y = max(center_y - height // 2, 0)
	cropped_img = img[top_left_y:top_left_y + height, top_left_x:top_left_x + height]

	return cropped_img


SRC = '../datasets/raw'
DEST = '../datasets/cropped'


os.makedirs(DEST, exist_ok=True)

for f_name in tqdm(os.listdir(SRC)):
	try:
	   img = center_crop_image(os.path.join(SRC, f_name))
	   cv2.imwrite(os.path.join(DEST, f_name), img)
	except:
		print(f'Error: {f_name}')
