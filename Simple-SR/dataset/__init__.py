from easydict import EasyDict as edict
import importlib
import os

from utils.common import scandir


dataset_root = os.path.dirname(os.path.abspath(__file__))
dataset_filenames = [
        os.path.splitext(os.path.basename(v))[0] for v in scandir(dataset_root)
        if v.endswith('_dataset.py')
]
_dataset_modules = [
        importlib.import_module(f'dataset.{file_name}')
        for file_name in dataset_filenames
]


class DATASET:
    LEGAL = ['OCT']

    OCT = edict()

    OCT.TRAIN = edict()
    OCT.TRAIN.HRx4 = '../../../datasets/beby/hr_train'
    OCT.TRAIN.LRx4 = '../../../datasets/beby/lr_train'

    
    OCT.VAL = edict()
    OCT.VAL.HRx4 = '../../../datasets/beby/hr_val'
    OCT.VAL.LRx4 = '../../../datasets/beby/lr_val'




def get_dataset(config):
    dataset_type = config.TYPE
    dataset_cls = None
    for module in _dataset_modules:
        dataset_cls = getattr(module, dataset_type, None)
        if dataset_cls is not None:
            break
    if dataset_cls is None:
        raise ValueError(f'Dataset {dataset_type} is not found.')

    hr_paths = []
    lr_paths = []
    D = DATASET()

    for dataset, split in zip(config.DATASETS, config.SPLITS):
        if dataset not in D.LEGAL or split not in eval('D.%s' % dataset):
            raise ValueError('Illegal dataset.')
        hr_paths.append(eval('D.%s.%s.HRx%d' % (dataset, split, config.SCALE)))
        lr_paths.append(eval('D.%s.%s.LRx%d' % (dataset, split, config.SCALE)))

    return dataset_cls(hr_paths, lr_paths, config)

