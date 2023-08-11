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

import random
import numpy as np
import torch

from imaginaire.utils.distributed import get_rank
from imaginaire.utils.distributed import master_only_print as print


def set_random_seed(seed, by_rank=False):
    r"""Set random seeds for everything, including random, numpy, torch.manual_seed, torch.cuda_manual_seed.
    torch.cuda.manual_seed_all is not necessary (included in torch.manual_seed)

    Args:
        seed (int): Random seed.
        by_rank (bool): if true, each gpu will use a different random seed.
    """
    if by_rank:
        seed += get_rank()
    print(f"Using random seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)         # sets seed on the current CPU & all GPUs
    torch.cuda.manual_seed(seed)    # sets seed on current GPU
    # torch.cuda.manual_seed_all(seed)  # included in torch.manual_seed
