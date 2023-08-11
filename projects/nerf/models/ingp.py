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
import tinycudann as tcnn

from projects.nerf.models import nerf


class Model(nerf.Model):

    def __init__(self, cfg_model, cfg_data):
        super().__init__(cfg_model, cfg_data)
        self.fine_sampling = False
        self.density_reg = cfg_model.density_noise_reg
        # Define models.
        self.nerf = InstantNGP(cfg_model)


class InstantNGP(nerf.NeRF):

    def __init__(self, cfg_model):
        self.voxel = cfg_model.voxel
        super().__init__(cfg_model)

    def set_input_dims(self, cfg_model):
        # Define the input encoding dimensions.
        self.input_3D_dim = 3 + cfg_model.voxel.dim * cfg_model.voxel.levels.num
        self.input_view_dim = 3 if cfg_model.view_dep else None

    def build_model(self, cfg_model):
        super().build_model(cfg_model)
        # Build the tcnn hash grid.
        l_min, l_max = self.voxel.levels.min, self.voxel.levels.max
        r_min, r_max = 2 ** l_min, 2 ** l_max
        num_levels = self.voxel.levels.num
        growth_rate = np.exp((np.log(r_max) - np.log(r_min)) / (num_levels - 1))
        config = dict(
            otype="HashGrid",
            n_levels=cfg_model.voxel.levels.num,
            n_features_per_level=cfg_model.voxel.dim,
            log2_hashmap_size=cfg_model.voxel.dict_size,
            base_resolution=2 ** cfg_model.voxel.levels.min,
            per_level_scale=growth_rate,
        )
        self.tiny_cuda_encoding = tcnn.Encoding(3, config)
        # Compute resolutions of all levels.
        self.resolutions = []
        for lv in range(0, num_levels):
            size = np.floor(r_min * growth_rate ** lv).astype(int) + 1
            self.resolutions.append(size)

    def forward(self, points_3D, ray_unit, density_reg=None):
        return super().forward(points_3D, ray_unit, density_reg)

    def _encode_3D(self, points_3D):
        # Tri-linear interpolate the corresponding embeddings from the dictionary.
        vol_min, vol_max = self.voxel.range
        points_3D_normalized = (points_3D - vol_min) / (vol_max - vol_min)  # Normalize to [0,1].
        tcnn_input = points_3D_normalized.view(-1, 3)
        tcnn_output = self.tiny_cuda_encoding(tcnn_input)
        points_enc = tcnn_output.view(*points_3D_normalized.shape[:-1], tcnn_output.shape[-1])
        points_enc = torch.cat([points_enc, points_3D], dim=-1)  # [B,R,N,LD+3]
        return points_enc

    def _encode_view(self, ray_unit):
        return ray_unit
