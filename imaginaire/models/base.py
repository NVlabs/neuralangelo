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


class Model(torch.nn.Module):

    def __init__(self, cfg_model, cfg_data):
        super().__init__()

    def get_param_groups(self, cfg_optim):
        """Allow the network to use different hyperparameters (e.g., learning rate) for different parameters.
        Returns:
            PyTorch parameter group (list or generator). See the PyTorch documentation for details.
        """
        return self.parameters()

    def device(self):
        """Return device on which model resides."""
        return next(self.parameters()).device
