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

import copy

import torch
from torch import nn
from imaginaire.utils.misc import requires_grad


def reset_batch_norm(m):
    r"""Reset batch norm statistics

    Args:
        m: Pytorch module
    """
    if hasattr(m, 'reset_running_stats'):
        m.reset_running_stats()


def calibrate_batch_norm_momentum(m):
    r"""Calibrate batch norm momentum

    Args:
        m: Pytorch module
    """
    if hasattr(m, 'reset_running_stats'):
        # if m._get_name() == 'SyncBatchNorm':
        if 'BatchNorm' in m._get_name():
            m.momentum = 1.0 / float(m.num_batches_tracked + 1)


class ModelAverage(nn.Module):
    r"""In this model average implementation, the spectral layers are
    absorbed in the model parameter by default. If such options are
    turned on, be careful with how you do the training. Remember to
    re-estimate the batch norm parameters before using the model.

    Args:
        module (torch nn module): Torch network.
        beta (float): Moving average weights. How much we weight the past.
        start_iteration (int): From which iteration, we start the update.
    """
    def __init__(self, module, beta=0.9999, start_iteration=0):
        super(ModelAverage, self).__init__()

        self.module = module
        # A shallow copy creates a new object which stores the reference of
        # the original elements.
        # A deep copy creates a new object and recursively adds the copies of
        # nested objects present in the original elements.
        self._averaged_model = copy.deepcopy(self.module).to('cuda')
        self.stream = torch.cuda.Stream()

        self.beta = beta

        self.start_iteration = start_iteration
        # This buffer is to track how many iterations has the model been
        # trained for. We will ignore the first $(start_iterations) and start
        # the averaging after.
        self.register_buffer('num_updates_tracked',
                             torch.tensor(0, dtype=torch.long))
        self.num_updates_tracked = self.num_updates_tracked.to('cuda')
        self.averaged_model.eval()

        # Averaged model does not require grad.
        requires_grad(self.averaged_model, False)

    @property
    def averaged_model(self):
        self.stream.synchronize()
        return self._averaged_model

    def forward(self, *inputs, **kwargs):
        r"""PyTorch module forward function overload."""
        return self.module(*inputs, **kwargs)

    @torch.no_grad()
    def update_average(self):
        r"""Update the moving average."""
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.num_updates_tracked += 1
            if self.num_updates_tracked <= self.start_iteration:
                beta = 0.
            else:
                beta = self.beta
            source_dict = self.module.state_dict()
            target_dict = self._averaged_model.state_dict()
            source_list = []
            target_list = []
            for key in target_dict:
                if 'num_batches_tracked' in key:
                    continue
                source_list.append(source_dict[key].data)
                target_list.append(target_dict[key].data.float())

            torch._foreach_mul_(target_list, beta)
            torch._foreach_add_(target_list, source_list, alpha=1 - beta)

    def __repr__(self):
        r"""Returns a string that holds a printable representation of an
        object"""
        return self.module.__repr__()
