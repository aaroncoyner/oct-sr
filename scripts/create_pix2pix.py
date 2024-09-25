import os
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import itertools


def setup_directories(hr_dest, lr_dest):
    os.makedirs(hr_dest, exist_ok=True)
    os.makedirs(lr_dest, exist_ok=True)


def center_crop_image(img):
    height, width, _ = img.shape
    center_x, center_y = width // 2, height // 2
    top_left_x = max(center_x - height // 2, 0)
    top_left_y = max(center_y - height // 2, 0)
    return img[top_left_y:top_left_y + height, top_left_x:top_left_x + height]
    

def resize_image(img, size=(800, 800)):
    return cv2.resize(img, size)


def subsample_image(img, output_size):
    height, width, _ = img.shape
    subsampled_images = []
    
    if output_size == (800, 100):
        for start_row in range(8):
            subsampled_images.append(img[start_row::8, :, :])
    elif output_size == (400, 100):
        for start_row in range(8):
            for start_col in range(2):
                subsampled_images.append(img[start_row::8, start_col::2, :])
    elif output_size == (400, 400):
        for start_row in range(2):
            for start_col in range(2):
                subsampled_images.append(img[start_row::2, start_col::2, :])
    else:
        raise ValueError("Invalid output size. Choose from (800, 100), (400, 100), or (400, 400).")
    
    return subsampled_images



def save_images(hr, lr, hr_dest, lr_dest, base_name, index):
    save_name = f'{base_name}-{str(index + 1)}.png'
    cv2.imwrite(os.path.join(hr_dest, save_name), hr)
    cv2.imwrite(os.path.join(lr_dest, save_name), lr)


def process_image(f_name, src, hr_dest, lr_dest, output_size):
    try:
        base_name = os.path.splitext(f_name)[0]
        img = cv2.imread(os.path.join(src, f_name))
        if img.shape[0] == 800:
            hr = center_crop_image(img)
            lr_images = subsample_image(hr, output_size)
            for index, lr in enumerate(lr_images):
                lr = resize_image(lr)
                save_images(hr, lr, hr_dest, lr_dest, base_name, index)
    except Exception as e:
        print(f'Error processing {f_name}: {str(e)}')


def process_images_in_parallel(src, hr_dest, lr_dest, output_size):
    file_names = os.listdir(src)
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(
            process_image,
            file_names,
            itertools.repeat(src),
            itertools.repeat(hr_dest),
            itertools.repeat(lr_dest),
            itertools.repeat(output_size)
        ), total=len(file_names)))


def main(src, hr_dest, lr_dest, output_size):
    setup_directories(hr_dest, lr_dest)
    process_images_in_parallel(src, hr_dest, lr_dest, output_size)


if __name__ == '__main__':
    SRC = '../datasets/raw/'
    OUTPUT_SIZE = (800, 100)

    hr_dest = f'../datasets/{OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]}/hr/'
    lr_dest = f'../datasets/{OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]}/lr/'
    
    main(SRC, hr_dest, lr_dest, OUTPUT_SIZE)
