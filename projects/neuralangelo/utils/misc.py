"""
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
"""

from functools import partial
import numpy as np
import torch
import torch.nn.functional as torch_F
from torch.optim import lr_scheduler

flip_mat = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


def get_scheduler(cfg_opt, opt):
    """Return the scheduler object.

    Args:
        cfg_opt (obj): Config for the specific optimization module (gen/dis).
        opt (obj): PyTorch optimizer object.

    Returns:
        (obj): Scheduler
    """
    if cfg_opt.sched.type == "two_steps_with_warmup":
        warm_up_end = cfg_opt.sched.warm_up_end
        two_steps = cfg_opt.sched.two_steps
        gamma = cfg_opt.sched.gamma

        def sch(x):
            if x < warm_up_end:
                return x / warm_up_end
            else:
                if x > two_steps[1]:
                    return 1.0 / gamma**2
                elif x > two_steps[0]:
                    return 1.0 / gamma
                else:
                    return 1.0

        scheduler = lr_scheduler.LambdaLR(opt, lambda x: sch(x))
    elif cfg_opt.sched.type == "cos_with_warmup":
        alpha = cfg_opt.sched.alpha
        max_iter = cfg_opt.sched.max_iter
        warm_up_end = cfg_opt.sched.warm_up_end

        def sch(x):
            if x < warm_up_end:
                return x / warm_up_end
            else:
                progress = (x - warm_up_end) / (max_iter - warm_up_end)
                learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (
                    1 - alpha
                ) + alpha
                return learning_factor

        scheduler = lr_scheduler.LambdaLR(opt, lambda x: sch(x))
    else:
        raise
    return scheduler


def eikonal_loss(gradients, outside=None):
    gradient_error = (gradients.norm(dim=-1) - 1.0) ** 2  # [B,R,N]
    gradient_error = gradient_error.nan_to_num(
        nan=0.0, posinf=0.0, neginf=0.0
    )  # [B,R,N]
    if outside is not None:
        return (gradient_error * (~outside).float()).mean()
    else:
        return gradient_error.mean()


def curvature_loss(hessian, outside=None):
    laplacian = hessian.sum(dim=-1).abs()  # [B,R,N]
    laplacian = laplacian.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)  # [B,R,N]
    if outside is not None:
        return (laplacian * (~outside).float()).mean()
    else:
        return laplacian.mean()


def get_activation(activ, **kwargs):
    func = dict(
        identity=lambda x: x,
        relu=torch_F.relu,
        relu_=torch_F.relu_,
        abs=torch.abs,
        abs_=torch.abs_,
        sigmoid=torch.sigmoid,
        sigmoid_=torch.sigmoid_,
        exp=torch.exp,
        exp_=torch.exp_,
        softplus=torch_F.softplus,
        silu=torch_F.silu,
        silu_=partial(torch_F.silu, inplace=True),
    )[activ]
    return partial(func, **kwargs)


def to_full_image(image, image_size=None, from_vec=True):
    # if from_vec is True: [B,HW,...,K] --> [B,K,H,W,...]
    # if from_vec is False: [B,H,W,...,K] --> [B,K,H,W,...]
    if from_vec:
        assert image_size is not None
        image = image.unflatten(dim=1, sizes=image_size)
    image = image.moveaxis(-1, 1)
    return image


def calculate_model_size(model):
    r"""Calculate number of parameters in a PyTorch network.

    Args:
        model (obj): PyTorch network.

    Returns:
        (int): Number of parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
        if isinstance(data_list[0], dict):
            data_gather[key] = collate_test_data_batches(data_list)
        elif isinstance(data_list[0], torch.Tensor):
            data_gather[key] = torch.cat(data_list, dim=0)
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
