import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import concurrent.futures
from tqdm import tqdm



def create_directories(dirs):
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)


def prepare_images_dataframe(hr_dir, lr_dir):
    images = pd.DataFrame({'file_name': os.listdir(hr_dir)})
    images['mrn'] = images['file_name'].str.split('-', n=1).str[0]
    images['hr'] = os.path.join(hr_dir, images['file_name'])
    images['lr'] = os.path.join(lr_dir, images['file_name'])
    return images


def split_by_mrn(images, seed=1337, test_size=0.2):
    unique_mrns = images['mrn'].unique()
    train_mrns, test_mrns = train_test_split(unique_mrns, test_size=test_size, random_state=seed)
    train = images[images['mrn'].isin(train_mrns)]
    test = images[images['mrn'].isin(test_mrns)]
    return train, test


def copy_files(row, src_col, dest_dir):
    try:
        src_file = row[src_col]
        dest_file = os.path.join(dest_dir, row['file_name'])
        shutil.copy(src_file, dest_file)
    except Exception as e:
        print(f'Error copying file {row["file_name"]} from {src_file} to {dest_dir}: {e}')


def parallel_copy(df, src_col, dest_dir):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        with tqdm(total=len(df), desc=f'Copying {src_col} files to {dest_dir}') as pbar:
            futures = [executor.submit(copy_files, row, src_col, dest_dir) for _, row in df.iterrows()]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f'File copy error: {e}')
                pbar.update(1)


def main(dataset_dirs):
    create_directories(dataset_dirs)

    images = prepare_images_dataframe(dataset_dirs['hr_dir'], dataset_dirs['lr_dir'])
    train, test = split_by_mrn(images)
    train.to_csv(os.path.join(dataset_dirs['base_dir'], 'train.csv'), index=False)
    test.to_csv(os.path.join(dataset_dirs['base_dir'], 'test.csv'), index=False)

    copy_tasks = [
        (train, 'lr', dataset_dirs['train_a']),
        (train, 'hr', dataset_dirs['train_b']),
        (test, 'lr', dataset_dirs['test_a']),
        (test, 'hr', dataset_dirs['test_b'])
    ]

    for df, src_col, dest_dir in copy_tasks:
        parallel_copy(df, src_col, dest_dir)


if __name__ == '__main__':
    OUTPUT_SIZE = (800, 100)
    dataset_dirs = {
        'base_dir': f'../datasets/{OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]}',
        'hr_dir': f'../datasets/{OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]}/hr',
        'lr_dir': f'../datasets/{OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]}/lr',
        'train_a': f'../datasets/{OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]}/train_A',
        'train_b': f'../datasets/{OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]}/train_B',
        'test_a': f'../datasets/{OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]}/test_A',
        'test_b': f'../datasets/{OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]}/test_B'
    }
    main(dataset_dirs)
