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

import math
import os
# pynvml is a python bindings to the NVIDIA Management Library
# https://developer.nvidia.com/nvidia-management-library-nvml
# An API for monitoring and managing various states of the NVIDIA GPU devices.
# It provides direct access to the queries and commands exposed via nvidia-smi.

import pynvml

pynvml.nvmlInit()


def system_get_driver_version():
    r"""Get Driver Version"""
    return pynvml.nvmlSystemGetDriverVersion()


def device_get_count():
    r"""Get number of devices"""
    return pynvml.nvmlDeviceGetCount()


class Device(object):
    r"""Device used for nvml."""
    _nvml_affinity_elements = math.ceil(os.cpu_count() / 64)

    def __init__(self, device_idx):
        super().__init__()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

    def get_name(self):
        r"""Get obect name"""
        return pynvml.nvmlDeviceGetName(self.handle)

    def get_cpu_affinity(self):
        r"""Get CPU affinity"""
        affinity_string = ''
        for j in pynvml.nvmlDeviceGetCpuAffinity(self.handle, Device._nvml_affinity_elements):
            # assume nvml returns list of 64 bit ints
            affinity_string = '{:064b}'.format(j) + affinity_string
        affinity_list = [int(x) for x in affinity_string]
        affinity_list.reverse()  # so core 0 is in 0th element of list

        return [i for i, e in enumerate(affinity_list) if e != 0]


def set_affinity(gpu_id=None):
    r"""Set GPU affinity

    Args:
        gpu_id (int): Which gpu device.
    """
    if gpu_id is None:
        gpu_id = int(os.getenv('LOCAL_RANK', 0))

    try:
        dev = Device(gpu_id)
        # os.sched_setaffinity() method in Python is used to set the CPU affinity mask of a process indicated
        # by the specified process id.
        # A process’s CPU affinity mask determines the set of CPUs on which it is eligible to run.
        # Syntax: os.sched_setaffinity(pid, mask)
        # pid=0 means the current process
        os.sched_setaffinity(0, dev.get_cpu_affinity())
        # list of ints
        # representing the logical cores this process is now affinitied with
        return os.sched_getaffinity(0)

    except pynvml.NVMLError:
        print("(Setting affinity with NVML failed, skipping...)")
