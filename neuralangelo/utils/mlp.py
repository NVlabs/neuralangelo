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

import numpy as np
import torch
import torch.nn.functional as torch_F


class MLPforNeuralSDF(torch.nn.Module):

    def __init__(self, layer_dims, skip_connection=[], activ=None, use_layernorm=False, use_weightnorm=False,
                 geometric_init=False, out_bias=0., invert=False):
        """Initialize a multi-layer perceptron with skip connection.
        Args:
            layer_dims: A list of integers representing the number of channels in each layer.
            skip_connection: A list of integers representing the index of layers to add skip connection.
        """
        super().__init__()
        self.skip_connection = skip_connection
        self.use_layernorm = use_layernorm
        self.linears = torch.nn.ModuleList()
        if use_layernorm:
            self.layer_norm = torch.nn.ModuleList()
        # Hidden layers
        layer_dim_pairs = list(zip(layer_dims[:-1], layer_dims[1:]))
        for li, (k_in, k_out) in enumerate(layer_dim_pairs):
            if li in self.skip_connection:
                k_in += layer_dims[0]
            linear = torch.nn.Linear(k_in, k_out)
            if geometric_init:
                self._geometric_init(linear, k_in, k_out, first=(li == 0),
                                     skip_dim=(layer_dims[0] if li in self.skip_connection else 0))
            if use_weightnorm:
                linear = torch.nn.utils.weight_norm(linear)
            self.linears.append(linear)
            if use_layernorm and li != len(layer_dim_pairs) - 1:
                self.layer_norm.append(torch.nn.LayerNorm(k_out))
            if li == len(layer_dim_pairs) - 1:
                self.linears[-1].bias.data.fill_(0.0)
        # SDF prediction layer
        self.linear_sdf = torch.nn.Linear(k_in, 1)
        if geometric_init:
            self._geometric_init_sdf(self.linear_sdf, k_in, out_bias=out_bias, invert=invert)
        self.activ = activ or torch_F.relu_

    def forward(self, input, with_sdf=True, with_feat=True):
        feat = input
        for li, linear in enumerate(self.linears):
            if li in self.skip_connection:
                feat = torch.cat([feat, input], dim=-1)
            if li != len(self.linears) - 1 or with_feat:
                feat_pre = linear(feat)
                if self.use_layernorm:
                    feat_pre = self.layer_norm[li](feat_pre)
                feat_activ = self.activ(feat_pre)
            if li == len(self.linears) - 1:
                out = [self.linear_sdf(feat) if with_sdf else None,
                       feat_activ if with_feat else None]
            feat = feat_activ
        return out

    def _geometric_init(self, linear, k_in, k_out, first=False, skip_dim=0):
        torch.nn.init.constant_(linear.bias, 0.0)
        torch.nn.init.normal_(linear.weight, 0.0, np.sqrt(2 / k_out))
        if first:
            torch.nn.init.constant_(linear.weight[:, 3:], 0.0)  # positional encodings
        if skip_dim:
            torch.nn.init.constant_(linear.weight[:, -skip_dim:], 0.0)  # skip connections

    def _geometric_init_sdf(self, linear, k_in, out_bias=0., invert=False):
        torch.nn.init.normal_(linear.weight, mean=np.sqrt(np.pi / k_in), std=0.0001)
        torch.nn.init.constant_(linear.bias, -out_bias)
        if invert:
            linear.weight.data *= -1
            linear.bias.data *= -1
