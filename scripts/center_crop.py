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


src = '../datasets/a_raw/'
dest = '../datasets/b_cropped/'
os.makedirs(dest, exist_ok=True)

for f_name in tqdm(os.listdir(src)):
	img = center_crop_image(src + f_name)
	cv2.imwrite(dest + f_name, img)
