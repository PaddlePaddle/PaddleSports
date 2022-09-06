
import numpy as np
import random
import paddle.io as io
from copy import deepcopy
from functools import partial

# from .prefetch_dataloader import PrefetchDataLoader
# from utils.logger import get_root_logger
# from .realesrgan_dataset import RealESRGANDataset
# from .realesrgan_paired_dataset import RealESRGANPairedDataset


# def build_dataset(dataset_opt):
#     """Build dataset from options.
#
#     Args:
#         dataset_opt (dict): Configuration for dataset. It must constain:
#             name (str): Dataset name.
#             type (str): Dataset type.
#     """
#     dataset_opt = deepcopy(dataset_opt)
#     if dataset_opt['type']=='RealESRGANDataset':
#         dataset = RealESRGANDataset(dataset_opt)
#     else:
#         dataset = RealESRGANPairedDataset(dataset_opt)
#
#     logger = get_root_logger()
#     logger.info(f'Dataset [{dataset.__class__.__name__}] - {dataset_opt["name"]} ' 'is built.')
#     return dataset


def build_dataloader(dataset, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=None):
    """Build dataloader.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options. It contains the following keys:
            phase (str): 'train' or 'val'.
            num_worker_per_gpu (int): Number of workers for each GPU.
            batch_size_per_gpu (int): Training batch size for each GPU.
        num_gpu (int): Number of GPUs. Used only in the train phase.
            Default: 1.
        dist (bool): Whether in distributed training. Used only in the train
            phase. Default: False.
        sampler (torch.utils.data.sampler): Data sampler. Default: None.
        seed (int | None): Seed. Default: None
    """
    phase = dataset_opt['phase']
    # rank, _ = get_dist_info()
    rank = 0
    if phase == 'train':
        if dist:  # distributed training
            batch_size = dataset_opt['batch_size_per_gpu']
            num_workers = dataset_opt['num_worker_per_gpu']
        else:  # non-distributed training
            multiplier = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
            num_workers = dataset_opt['num_worker_per_gpu'] * multiplier
        dataloader_args = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True)

        dataloader_args['worker_init_fn'] = partial(
            worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None
    elif phase in ['val', 'test']:  # validation
        dataloader_args = dict(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    else:
        raise ValueError(f'Wrong dataset phase: {phase}. ' "Supported ones are 'train', 'val' and 'test'.")

    # prefetch_mode = dataset_opt.get('prefetch_mode')
    # if prefetch_mode == 'cpu':  # CPUPrefetcher
    #     num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1)
    #     logger = get_root_logger()
    #     logger.info(f'Use {prefetch_mode} prefetch dataloader: ' f'num_prefetch_queue = {num_prefetch_queue}')
    #     return PrefetchDataLoader(num_prefetch_queue=num_prefetch_queue, **dataloader_args)
    # else:
    return io.DataLoader(**dataloader_args)


def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
