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

import torch

from imaginaire.utils.distributed import dist_all_gather_tensor


def collate_test_data_batches(data_batches):
    """Aggregate the list of test data from all devices and process the results.
    Args:
        data_batches (list): List of (hierarchical) dictionaries, where leaf entries are tensors.
    Returns:
        data_gather (dict): (hierarchical) dictionaries, where leaf entries are concatenated tensors.
    """
    data_gather = dict()
    for key in data_batches[0].keys():
        data_list = [data[key] for data in data_batches]
        if isinstance(data_batches[0][key], dict):
            data_gather[key] = collate_test_data_batches(data_list)
        elif isinstance(data_batches[0][key], torch.Tensor):
            data_gather[key] = torch.cat(data_list, dim=0)
            data_gather[key] = torch.cat(dist_all_gather_tensor(data_gather[key].contiguous()), dim=0)
        else:
            raise TypeError
    return data_gather


def get_unique_test_data(data_gather, idx):
    """Aggregate the list of test data from all devices and process the results.
    Args:
        data_gather (dict): (hierarchical) dictionaries, where leaf entries are tensors.
        idx (tensor): sample indices.
    Returns:
        data_all (dict): (hierarchical) dictionaries, where leaf entries are tensors ordered by idx.
    """
    data_all = dict()
    for key, value in data_gather.items():
        if isinstance(value, dict):
            data_all[key] = get_unique_test_data(value, idx)
        elif isinstance(value, torch.Tensor):
            data_all[key] = []
            for i in range(max(idx) + 1):
                # If multiple occurrences of the same idx, just choose the first one. If no occurrence, just ignore.
                matches = (idx == i).nonzero(as_tuple=True)[0]
                if matches.numel() != 0:
                    data_all[key].append(value[matches[0]])
            data_all[key] = torch.stack(data_all[key], dim=0)
        else:
            raise TypeError
    return data_all


def trim_test_samples(data, max_samples=None):
    for key, value in data.items():
        if isinstance(value, dict):
            data[key] = trim_test_samples(value, max_samples=max_samples)
        elif isinstance(value, torch.Tensor):
            if max_samples is not None:
                data[key] = value[:max_samples]
        else:
            raise TypeError
