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

import torch.backends.cudnn as cudnn

from imaginaire.utils.distributed import master_only_print as print


def init_cudnn(deterministic, benchmark):
    r"""Initialize the cudnn module. The two things to consider is whether to
    use cudnn benchmark and whether to use cudnn deterministic. If cudnn
    benchmark is set, then the cudnn deterministic is automatically false.

    Args:
        deterministic (bool): Whether to use cudnn deterministic.
        benchmark (bool): Whether to use cudnn benchmark.
    """
    cudnn.deterministic = deterministic
    cudnn.benchmark = benchmark
    print('cudnn benchmark: {}'.format(benchmark))
    print('cudnn deterministic: {}'.format(deterministic))
