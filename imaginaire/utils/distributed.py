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

import functools
import ctypes

import torch
import torch.distributed as dist
from contextlib import contextmanager


def init_dist(local_rank, backend='nccl', **kwargs):
    r"""Initialize distributed training"""
    if dist.is_available():
        if dist.is_initialized():
            return torch.cuda.current_device()
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method='env://', **kwargs)

    # Increase the L2 fetch granularity for faster speed.
    _libcudart = ctypes.CDLL('libcudart.so')
    # Set device limit on the current device
    # cudaLimitMaxL2FetchGranularity = 0x05
    pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    # assert pValue.contents.value == 128


def get_rank():
    r"""Get rank of the thread."""
    rank = 0
    if dist.is_available():
        if dist.is_initialized():
            rank = dist.get_rank()
    return rank


def get_world_size():
    r"""Get world size. How many GPUs are available in this job."""
    world_size = 1
    if dist.is_available():
        if dist.is_initialized():
            world_size = dist.get_world_size()
    return world_size


def broadcast_object_list(message, src=0):
    r"""Broadcast object list from the master to the others"""
    # Send logdir from master to all workers.
    if dist.is_available():
        if dist.is_initialized():
            torch.distributed.broadcast_object_list(message, src=src)
    return message


def master_only(func):
    r"""Apply this function only to the master GPU."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        r"""Simple function wrapper for the master function"""
        if get_rank() == 0:
            return func(*args, **kwargs)
        else:
            return None
    return wrapper


def is_master():
    r"""check if current process is the master"""
    return get_rank() == 0


def is_dist():
    return dist.is_initialized()


def barrier():
    if is_dist():
        dist.barrier()


@contextmanager
def master_first():
    if not is_master():
        barrier()
    yield
    if dist.is_initialized() and is_master():
        barrier()


def is_local_master():
    return torch.cuda.current_device() == 0


@master_only
def master_only_print(*args):
    r"""master-only print"""
    print(*args)


def dist_reduce_tensor(tensor, rank=0, reduce='mean'):
    r""" Reduce to rank 0 """
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    with torch.no_grad():
        dist.reduce(tensor, dst=rank)
        if get_rank() == rank:
            if reduce == 'mean':
                tensor /= world_size
            elif reduce == 'sum':
                pass
            else:
                raise NotImplementedError
    return tensor


def dist_all_reduce_tensor(tensor, reduce='mean'):
    r""" Reduce to all ranks """
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    with torch.no_grad():
        dist.all_reduce(tensor)
        if reduce == 'mean':
            tensor /= world_size
        elif reduce == 'sum':
            pass
        else:
            raise NotImplementedError
    return tensor


def dist_all_gather_tensor(tensor):
    r""" gather to all ranks """
    world_size = get_world_size()
    if world_size < 2:
        return [tensor]
    tensor_list = [
        torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    with torch.no_grad():
        dist.all_gather(tensor_list, tensor)
    return tensor_list
