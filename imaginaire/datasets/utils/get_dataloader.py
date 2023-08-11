'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import importlib

import torch
import torch.distributed as dist

from imaginaire.utils.distributed import master_only_print as print

from imaginaire.datasets.utils.sampler import DistributedSamplerPreemptable
from imaginaire.datasets.utils.dataloader import MultiEpochsDataLoader


def _get_train_dataset_objects(cfg, subset_indices=None):
    r"""Return dataset objects for the training set.
    Args:
        cfg (obj): Global configuration file.
        subset_indices (sequence): Indices of the subset to use.

    Returns:
        train_dataset (obj): PyTorch training dataset object.
    """
    dataset_module = importlib.import_module(cfg.data.type)
    train_dataset = dataset_module.Dataset(cfg, is_inference=False)
    if subset_indices is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, subset_indices)
    print('Train dataset length:', len(train_dataset))
    return train_dataset


def _get_val_dataset_objects(cfg, subset_indices=None):
    r"""Return dataset objects for the validation set.
    Args:
        cfg (obj): Global configuration file.
        subset_indices (sequence): Indices of the subset to use.
    Returns:
        val_dataset (obj): PyTorch validation dataset object.
    """
    dataset_module = importlib.import_module(cfg.data.type)
    if hasattr(cfg.data.val, 'type'):
        for key in ['type', 'input_types', 'input_image']:
            setattr(cfg.data, key, getattr(cfg.data.val, key))
        dataset_module = importlib.import_module(cfg.data.type)
    val_dataset = dataset_module.Dataset(cfg, is_inference=True)

    if subset_indices is not None:
        val_dataset = torch.utils.data.Subset(val_dataset, subset_indices)
    print('Val dataset length:', len(val_dataset))
    return val_dataset


def _get_test_dataset_object(cfg, subset_indices=None):
    r"""Return dataset object for the test set

    Args:
        cfg (obj): Global configuration file.
        subset_indices (sequence): Indices of the subset to use.
    Returns:
        (obj): PyTorch dataset object.
    """
    dataset_module = importlib.import_module(cfg.test_data.type)
    test_dataset = dataset_module.Dataset(cfg, is_inference=True, is_test=True)
    if subset_indices is not None:
        test_dataset = torch.utils.data.Subset(test_dataset, subset_indices)
    return test_dataset


def _get_data_loader(cfg, dataset, batch_size, not_distributed=False,
                     shuffle=True, drop_last=True, seed=0, use_multi_epoch_loader=False,
                     preemptable=False):
    r"""Return data loader .

    Args:
        cfg (obj): Global configuration file.
        dataset (obj): PyTorch dataset object.
        batch_size (int): Batch size.
        not_distributed (bool): Do not use distributed samplers.
        shuffle (bool): Whether to shuffle the data
        drop_last (bool): Whether to drop the last batch is the number of samples is smaller than the batch size
        seed (int): random seed.
        preemptable (bool): Whether to handle preemptions.
    Return:
        (obj): Data loader.
    """
    not_distributed = not_distributed or not dist.is_initialized()
    if not_distributed:
        sampler = None
    else:
        if preemptable:
            sampler = DistributedSamplerPreemptable(dataset, shuffle=shuffle, seed=seed)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle, seed=seed)
    num_workers = getattr(cfg.data, 'num_workers', 8)
    persistent_workers = getattr(cfg.data, 'persistent_workers', False)
    data_loader = (MultiEpochsDataLoader if use_multi_epoch_loader else torch.utils.data.DataLoader)(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and (sampler is None),
        sampler=sampler,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=drop_last,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )
    return data_loader


def get_train_dataloader(
        cfg, shuffle=True, drop_last=True, subset_indices=None, seed=0, preemptable=False):
    r"""Return dataset objects for the training and validation sets.
    Args:
        cfg (obj): Global configuration file.
        shuffle (bool): Whether to shuffle the data
        drop_last (bool): Whether to drop the last batch is the number of samples is smaller than the batch size
        subset_indices (sequence): Indices of the subset to use.
        seed (int): random seed.
        preemptable (bool): Flag for preemption handling
    Returns:
        train_data_loader (obj): Train data loader.
    """
    train_dataset = _get_train_dataset_objects(cfg, subset_indices=subset_indices)
    train_data_loader = _get_data_loader(
        cfg, train_dataset, cfg.data.train.batch_size, not_distributed=False,
        shuffle=shuffle, drop_last=drop_last, seed=seed,
        use_multi_epoch_loader=cfg.data.use_multi_epoch_loader,
        preemptable=preemptable
    )
    return train_data_loader


def get_val_dataloader(cfg, subset_indices=None, seed=0):
    r"""Return dataset objects for the training and validation sets.
    Args:
        cfg (obj): Global configuration file.
        subset_indices (sequence): Indices of the subset to use.
        seed (int): random seed.
    Returns:
        val_data_loader (obj): Val data loader.
    """
    val_dataset = _get_val_dataset_objects(cfg, subset_indices=subset_indices)
    not_distributed = getattr(cfg.data, 'val_data_loader_not_distributed', False)
    # We often use a folder of images to represent a video. As doing evaluation, we like the images to preserve the
    # original order. As a result, we do not want to distribute images from the same video to different GPUs.
    not_distributed = 'video' in cfg.data.type or not_distributed
    drop_last = getattr(cfg.data.val, 'drop_last', False)
    # Validation loader need not have preemption handling.
    val_data_loader = _get_data_loader(
        cfg, val_dataset, cfg.data.val.batch_size, not_distributed=not_distributed,
        shuffle=False, drop_last=drop_last, seed=seed,
        preemptable=False
    )
    return val_data_loader


def get_test_dataloader(cfg, subset_indices=None):
    r"""Return dataset objects for testing

    Args:
        cfg (obj): Global configuration file.
        subset_indices (sequence): Indices of the subset to use.
    Returns:
        (obj): Test data loader. It may not contain the ground truth.
    """
    test_dataset = _get_test_dataset_object(cfg, subset_indices=subset_indices)
    not_distributed = getattr(
        cfg.test_data, 'val_data_loader_not_distributed', False)
    not_distributed = 'video' in cfg.test_data.type or not_distributed
    test_data_loader = _get_data_loader(
        cfg, test_dataset, cfg.test_data.test.batch_size, not_distributed=not_distributed,
        shuffle=False)
    return test_data_loader
