import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn


def set_random_seed(seed):
    r"""Set random seeds for everything, including random, numpy, torch.manual_seed, torch.cuda_manual_seed.
    torch.cuda.manual_seed_all is not necessary (included in torch.manual_seed)

    Args:
        seed (int): Random seed.
        by_rank (bool): if true, each gpu will use a different random seed.
    """
    print(f"Using random seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # sets seed on the current CPU & all GPUs
    torch.cuda.manual_seed(seed)  # sets seed on current GPU
    # torch.cuda.manual_seed_all(seed)  # included in torch.manual_seed


def setup_optimizer(cfg, model):
    params = cfg.optim.params
    optim = torch.optim.AdamW(model.parameters(), **params)
    return optim


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
    print("cudnn benchmark: {}".format(benchmark))
    print("cudnn deterministic: {}".format(deterministic))
