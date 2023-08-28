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

import collections
import functools
import os
import signal
import time
from collections import OrderedDict

import torch
import torch.nn.functional as F
import wandb

from imaginaire.utils.distributed import is_master, master_only

string_classes = (str, bytes)

from imaginaire.utils.termcolor import alert, PP  # noqa


def santize_args(name, locals_fn):
    args = {k: v for k, v in locals_fn.items()}
    if 'kwargs' in args and args['kwargs']:
        unused = PP(args['kwargs'])
        alert(f'{name}: Unused kwargs\n{unused}')

    keys_to_remove = ['self', 'kwargs']
    for k in keys_to_remove:
        args.pop(k, None)
    alert(f'{name}: Used args\n{PP(args)}', 'green')
    return args


def split_labels(labels, label_lengths):
    r"""Split concatenated labels into their parts.

    Args:
        labels (torch.Tensor): Labels obtained through concatenation.
        label_lengths (OrderedDict): Containing order of labels & their lengths.

    Returns:

    """
    assert isinstance(label_lengths, OrderedDict)
    start = 0
    outputs = {}
    for data_type, length in label_lengths.items():
        end = start + length
        if labels.dim() == 5:
            outputs[data_type] = labels[:, :, start:end]
        elif labels.dim() == 4:
            outputs[data_type] = labels[:, start:end]
        elif labels.dim() == 3:
            outputs[data_type] = labels[start:end]
        start = end
    return outputs


def requires_grad(model, require=True):
    r""" Set a model to require gradient or not.

    Args:
        model (nn.Module): Neural network model.
        require (bool): Whether the network requires gradient or not.

    Returns:

    """
    for p in model.parameters():
        p.requires_grad = require


def to_device(data, device):
    r"""Move all tensors inside data to device.

    Args:
        data (dict, list, or tensor): Input data.
        device (str): 'cpu' or 'cuda'.
    """
    if isinstance(device, str):
        device = torch.device(device)
    assert isinstance(device, torch.device)

    if isinstance(data, torch.Tensor):
        data = data.to(device)
        return data
    elif isinstance(data, collections.abc.Mapping):
        return type(data)({key: to_device(data[key], device) for key in data})
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
        return type(data)([to_device(d, device) for d in data])
    else:
        return data


def to_cuda(data):
    r"""Move all tensors inside data to gpu.

    Args:
        data (dict, list, or tensor): Input data.
    """
    return to_device(data, 'cuda')


def to_cpu(data):
    r"""Move all tensors inside data to cpu.

    Args:
        data (dict, list, or tensor): Input data.
    """
    return to_device(data, 'cpu')


def to_half(data):
    r"""Move all floats to half.

    Args:
        data (dict, list or tensor): Input data.
    """
    if isinstance(data, torch.Tensor) and torch.is_floating_point(data):
        data = data.half()
        return data
    elif isinstance(data, collections.abc.Mapping):
        return type(data)({key: to_half(data[key]) for key in data})
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
        return type(data)([to_half(d) for d in data])
    else:
        return data


def to_float(data):
    r"""Move all halfs to float.

    Args:
        data (dict, list or tensor): Input data.
    """
    if isinstance(data, torch.Tensor) and torch.is_floating_point(data):
        data = data.float()
        return data
    elif isinstance(data, collections.abc.Mapping):
        return type(data)({key: to_float(data[key]) for key in data})
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
        return type(data)([to_float(d) for d in data])
    else:
        return data


def slice_tensor(data, start, end):
    r"""Slice all tensors from start to end.
    Args:
        data (dict, list or tensor): Input data.
    """
    if isinstance(data, torch.Tensor):
        data = data[start:end]
        return data
    elif isinstance(data, collections.abc.Mapping):
        return type(data)({key: slice_tensor(data[key], start, end) for key in data})
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
        return type(data)([slice_tensor(d, start, end) for d in data])
    else:
        return data


def get_and_setattr(cfg, name, default):
    r"""Get attribute with default choice. If attribute does not exist, set it
    using the default value.

    Args:
        cfg (obj) : Config options.
        name (str) : Attribute name.
        default (obj) : Default attribute.

    Returns:
        (obj) : Desired attribute.
    """
    if not hasattr(cfg, name) or name not in cfg.__dict__:
        setattr(cfg, name, default)
    return getattr(cfg, name)


def get_nested_attr(cfg, attr_name, default):
    r"""Iteratively try to get the attribute from cfg. If not found, return
    default.

    Args:
        cfg (obj): Config file.
        attr_name (str): Attribute name (e.g. XXX.YYY.ZZZ).
        default (obj): Default return value for the attribute.

    Returns:
        (obj): Attribute value.
    """
    names = attr_name.split('.')
    atr = cfg
    for name in names:
        if not hasattr(atr, name):
            return default
        atr = getattr(atr, name)
    return atr


def gradient_norm(model):
    r"""Return the gradient norm of model.

    Args:
        model (PyTorch module): Your network.

    """
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** (1. / 2)


def random_shift(x, offset=0.05, mode='bilinear', padding_mode='reflection'):
    r"""Randomly shift the input tensor.

    Args:
        x (4D tensor): The input batch of images.
        offset (int): The maximum offset ratio that is between [0, 1].
        The maximum shift is offset * image_size for each direction.
        mode (str): The resample mode for 'F.grid_sample'.
        padding_mode (str): The padding mode for 'F.grid_sample'.

    Returns:
        x (4D tensor) : The randomly shifted image.
    """
    assert x.dim() == 4, "Input must be a 4D tensor."
    batch_size = x.size(0)
    theta = torch.eye(2, 3, device=x.device).unsqueeze(0).repeat(
        batch_size, 1, 1)
    theta[:, :, 2] = 2 * offset * torch.rand(batch_size, 2) - offset
    grid = F.affine_grid(theta, x.size())
    x = F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode, align_corners=False)
    return x


# def truncated_gaussian(threshold, size, seed=None, device=None):
#     r"""Apply the truncated gaussian trick to trade diversity for quality
#
#     Args:
#         threshold (float): Truncation threshold.
#         size (list of integer): Tensor size.
#         seed (int): Random seed.
#         device:
#     """
#     state = None if seed is None else np.random.RandomState(seed)
#     values = truncnorm.rvs(-threshold, threshold,
#                            size=size, random_state=state)
#     return torch.tensor(values, device=device).float()


def apply_imagenet_normalization(input):
    r"""Normalize using ImageNet mean and std.

    Args:
        input (4D tensor NxCxHxW): The input images, assuming to be [-1, 1].

    Returns:
        Normalized inputs using the ImageNet normalization.
    """
    # normalize the input back to [0, 1]
    normalized_input = (input + 1) / 2
    # normalize the input using the ImageNet mean and std
    mean = normalized_input.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = normalized_input.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    output = (normalized_input - mean) / std
    return output


def alarm_handler(timeout_period, signum, frame):
    # What to do when the process gets stuck. For now, we simply end the process.
    error_message = f"Timeout error! More than {timeout_period} seconds have passed since the last iteration. Most " \
                    f"likely the process has been stuck due to node failure or PBSS error."
    ngc_job_id = os.environ.get('NGC_JOB_ID', None)
    if ngc_job_id is not None:
        error_message += f" Failed NGC job ID: {ngc_job_id}."
    # Let's reserve `wandb.alert` for this purpose.
    wandb.alert(title="Timeout error!", text=error_message, level=wandb.AlertLevel.ERROR)
    exit()


class Timer(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.time_iteration = 0
        self.time_epoch = 0
        if is_master():
            # noinspection PyTypeChecker
            signal.signal(signal.SIGALRM, functools.partial(alarm_handler, self.cfg.timeout_period))

    def reset(self):
        self.accu_forw_iter_time = 0
        self.accu_loss_iter_time = 0
        self.accu_back_iter_time = 0
        self.accu_step_iter_time = 0
        self.accu_avg_iter_time = 0

    def _time_before_forward(self):
        r"""Record time before applying forward."""
        if self.cfg.speed_benchmark:
            torch.cuda.synchronize()
            self.forw_time = time.time()

    def _time_before_loss(self):
        r"""Record time before computing loss."""
        if self.cfg.speed_benchmark:
            torch.cuda.synchronize()
            self.loss_time = time.time()

    def _time_before_backward(self):
        r"""Record time before applying backward."""
        if self.cfg.speed_benchmark:
            torch.cuda.synchronize()
            self.back_time = time.time()

    def _time_before_step(self):
        r"""Record time before updating the weights"""
        if self.cfg.speed_benchmark:
            torch.cuda.synchronize()
            self.step_time = time.time()

    def _time_before_model_avg(self):
        r"""Record time before applying model average."""
        if self.cfg.speed_benchmark:
            torch.cuda.synchronize()
            self.avg_time = time.time()

    def _time_before_leave_gen(self):
        r"""Record forward, backward, loss, and model average time for the network update."""
        if self.cfg.speed_benchmark:
            torch.cuda.synchronize()
            end_time = time.time()
            self.accu_forw_iter_time += self.loss_time - self.forw_time
            self.accu_loss_iter_time += self.back_time - self.loss_time
            self.accu_back_iter_time += self.step_time - self.back_time
            self.accu_step_iter_time += self.avg_time - self.step_time
            self.accu_avg_iter_time += end_time - self.avg_time

    def _print_speed_benchmark(self, avg_time):
        """Prints the profiling results and resets the timers."""
        print('{:6f}'.format(avg_time))
        print('\tModel FWD time {:6f}'.format(self.accu_forw_iter_time / self.cfg.logging_iter))
        print('\tModel LOS time {:6f}'.format(self.accu_loss_iter_time / self.cfg.logging_iter))
        print('\tModel BCK time {:6f}'.format(self.accu_back_iter_time / self.cfg.logging_iter))
        print('\tModel STP time {:6f}'.format(self.accu_step_iter_time / self.cfg.logging_iter))
        print('\tModel AVG time {:6f}'.format(self.accu_avg_iter_time / self.cfg.logging_iter))
        self.accu_forw_iter_time = 0
        self.accu_loss_iter_time = 0
        self.accu_back_iter_time = 0
        self.accu_step_iter_time = 0
        self.accu_avg_iter_time = 0

    def checkpoint_tic(self):
        # reset timer
        self.checkpoint_start_time = time.time()

    def checkpoint_toc(self):
        # return time by minutes
        return (time.time() - self.checkpoint_start_time) / 60

    @master_only
    def reset_timeout_counter(self):
        signal.alarm(self.cfg.timeout_period)
