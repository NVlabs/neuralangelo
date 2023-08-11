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
from torch.nn import init


def weights_init(init_type, gain, bias=None):
    r"""Initialize weights in the network.

    Args:
        init_type (str): The name of the initialization scheme.
        gain (float): The parameter that is required for the initialization
            scheme.
        bias (object): If not ``None``, specifies the initialization parameter
            for bias.

    Returns:
        (obj): init function to be applied.
    """

    def init_func(m):
        r"""Init function

        Args:
            m: module to be weight initialized.
        """
        class_name = m.__class__.__name__
        if hasattr(m, 'weight') and (
                class_name.find('Conv') != -1 or
                class_name.find('Linear') != -1 or
                class_name.find('Embedding') != -1):
            lr_mul = getattr(m, 'lr_mul', 1.)
            gain_final = gain / lr_mul
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain_final)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain_final)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain_final)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                with torch.no_grad():
                    m.weight.data *= gain_final
            elif init_type == 'kaiming_linear':
                init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_in', nonlinearity='linear'
                )
                with torch.no_grad():
                    m.weight.data *= gain_final
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain_final)
            elif init_type == 'none':
                pass
            else:
                raise NotImplementedError(
                    'initialization method [%s] is '
                    'not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            if init_type == 'none':
                pass
            elif bias is not None:
                bias_type = getattr(bias, 'type', 'normal')
                if bias_type == 'normal':
                    bias_gain = getattr(bias, 'gain', 0.5)
                    init.normal_(m.bias.data, 0.0, bias_gain)
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is '
                        'not implemented' % bias_type)
            else:
                init.constant_(m.bias.data, 0.0)
    return init_func


def weights_rescale():
    def init_func(m):
        if hasattr(m, 'init_gain'):
            for name, p in m.named_parameters():
                if 'output_scale' not in name:
                    p.data.mul_(m.init_gain)
    return init_func
