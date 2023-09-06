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
from functools import partial
import numpy as np
import tinycudann as tcnn

from projects.neuralangelo.utils.spherical_harmonics import get_spherical_harmonics
from projects.neuralangelo.utils.mlp import MLPforNeuralSDF
from projects.neuralangelo.utils.misc import get_activation
from projects.nerf.utils import nerf_util


class NeuralSDF(torch.nn.Module):

    def __init__(self, cfg_sdf):
        super().__init__()
        self.cfg_sdf = cfg_sdf
        encoding_dim = self.build_encoding(cfg_sdf.encoding)
        input_dim = 3 + encoding_dim
        self.build_mlp(cfg_sdf.mlp, input_dim=input_dim)

    def build_encoding(self, cfg_encoding):
        if cfg_encoding.type == "fourier":
            encoding_dim = 6 * cfg_encoding.levels
        elif cfg_encoding.type == "hashgrid":
            # Build the multi-resolution hash grid.
            l_min, l_max = cfg_encoding.hashgrid.min_logres, cfg_encoding.hashgrid.max_logres
            r_min, r_max = 2 ** l_min, 2 ** l_max
            num_levels = cfg_encoding.levels
            self.growth_rate = np.exp((np.log(r_max) - np.log(r_min)) / (num_levels - 1))
            config = dict(
                otype="HashGrid",
                n_levels=cfg_encoding.levels,
                n_features_per_level=cfg_encoding.hashgrid.dim,
                log2_hashmap_size=cfg_encoding.hashgrid.dict_size,
                base_resolution=2 ** cfg_encoding.hashgrid.min_logres,
                per_level_scale=self.growth_rate,
            )
            self.tcnn_encoding = tcnn.Encoding(3, config)
            self.resolutions = []
            for lv in range(0, num_levels):
                size = np.floor(r_min * self.growth_rate ** lv).astype(int) + 1
                self.resolutions.append(size)
            encoding_dim = cfg_encoding.hashgrid.dim * cfg_encoding.levels
        else:
            raise NotImplementedError("Unknown encoding type")
        return encoding_dim

    def build_mlp(self, cfg_mlp, input_dim=3):
        # SDF + point-wise feature
        layer_dims = [input_dim] + [cfg_mlp.hidden_dim] * cfg_mlp.num_layers + [cfg_mlp.hidden_dim]
        activ = get_activation(cfg_mlp.activ, **cfg_mlp.activ_params)
        self.mlp = MLPforNeuralSDF(layer_dims, skip_connection=cfg_mlp.skip, activ=activ,
                                   use_weightnorm=cfg_mlp.weight_norm, geometric_init=cfg_mlp.geometric_init,
                                   out_bias=cfg_mlp.out_bias, invert=cfg_mlp.inside_out)

    def forward(self, points_3D, with_sdf=True, with_feat=True):
        points_enc = self.encode(points_3D)  # [...,3+LD]
        sdf, feat = self.mlp(points_enc, with_sdf=with_sdf, with_feat=with_feat)
        return sdf, feat  # [...,1],[...,K]

    def sdf(self, points_3D):
        return self.forward(points_3D, with_sdf=True, with_feat=False)[0]

    def encode(self, points_3D):
        if self.cfg_sdf.encoding.type == "fourier":
            points_enc = nerf_util.positional_encoding(points_3D, num_freq_bases=self.cfg_sdf.encoding.levels)
            feat_dim = 6
        elif self.cfg_sdf.encoding.type == "hashgrid":
            # Tri-linear interpolate the corresponding embeddings from the dictionary.
            vol_min, vol_max = self.cfg_sdf.encoding.hashgrid.range
            points_3D_normalized = (points_3D - vol_min) / (vol_max - vol_min)  # Normalize to [0,1].
            tcnn_input = points_3D_normalized.view(-1, 3)
            tcnn_output = self.tcnn_encoding(tcnn_input)
            points_enc = tcnn_output.view(*points_3D_normalized.shape[:-1], tcnn_output.shape[-1])
            feat_dim = self.cfg_sdf.encoding.hashgrid.dim
        else:
            raise NotImplementedError("Unknown encoding type")
        # Coarse-to-fine.
        if self.cfg_sdf.encoding.coarse2fine.enabled:
            mask = self._get_coarse2fine_mask(points_enc, feat_dim=feat_dim)
            points_enc = points_enc * mask
        points_enc = torch.cat([points_3D, points_enc], dim=-1)  # [B,R,N,3+LD]
        return points_enc

    def set_active_levels(self, current_iter=None):
        anneal_levels = max((current_iter - self.warm_up_end) // self.cfg_sdf.encoding.coarse2fine.step, 1)
        self.anneal_levels = min(self.cfg_sdf.encoding.levels, anneal_levels)
        self.active_levels = max(self.cfg_sdf.encoding.coarse2fine.init_active_level, self.anneal_levels)

    def set_normal_epsilon(self):
        if self.cfg_sdf.encoding.coarse2fine.enabled:
            epsilon_res = self.resolutions[self.anneal_levels - 1]
        else:
            epsilon_res = self.resolutions[-1]
        self.normal_eps = 1. / epsilon_res

    @torch.no_grad()
    def _get_coarse2fine_mask(self, points_enc, feat_dim):
        mask = torch.zeros_like(points_enc)
        mask[..., :(self.active_levels * feat_dim)] = 1
        return mask

    def compute_gradients(self, x, training=False, sdf=None):
        # Note: hessian is not fully hessian but diagonal elements
        if self.cfg_sdf.gradient.mode == "analytical":
            requires_grad = x.requires_grad
            with torch.enable_grad():
                # 1st-order gradient
                x.requires_grad_(True)
                sdf = self.sdf(x)
                gradient = torch.autograd.grad(sdf.sum(), x, create_graph=True)[0]
                # 2nd-order gradient (hessian)
                if training:
                    hessian = torch.autograd.grad(gradient.sum(), x, create_graph=True)[0]
                else:
                    hessian = None
                    gradient = gradient.detach()
            x.requires_grad_(requires_grad)
        elif self.cfg_sdf.gradient.mode == "numerical":
            if self.cfg_sdf.gradient.taps == 6:
                eps = self.normal_eps
                # 1st-order gradient
                eps_x = torch.tensor([eps, 0., 0.], dtype=x.dtype, device=x.device)  # [3]
                eps_y = torch.tensor([0., eps, 0.], dtype=x.dtype, device=x.device)  # [3]
                eps_z = torch.tensor([0., 0., eps], dtype=x.dtype, device=x.device)  # [3]
                sdf_x_pos = self.sdf(x + eps_x)  # [...,1]
                sdf_x_neg = self.sdf(x - eps_x)  # [...,1]
                sdf_y_pos = self.sdf(x + eps_y)  # [...,1]
                sdf_y_neg = self.sdf(x - eps_y)  # [...,1]
                sdf_z_pos = self.sdf(x + eps_z)  # [...,1]
                sdf_z_neg = self.sdf(x - eps_z)  # [...,1]
                gradient_x = (sdf_x_pos - sdf_x_neg) / (2 * eps)
                gradient_y = (sdf_y_pos - sdf_y_neg) / (2 * eps)
                gradient_z = (sdf_z_pos - sdf_z_neg) / (2 * eps)
                gradient = torch.cat([gradient_x, gradient_y, gradient_z], dim=-1)  # [...,3]
                # 2nd-order gradient (hessian)
                if training:
                    assert sdf is not None  # computed when feed-forwarding through the network
                    hessian_xx = (sdf_x_pos + sdf_x_neg - 2 * sdf) / (eps ** 2)  # [...,1]
                    hessian_yy = (sdf_y_pos + sdf_y_neg - 2 * sdf) / (eps ** 2)  # [...,1]
                    hessian_zz = (sdf_z_pos + sdf_z_neg - 2 * sdf) / (eps ** 2)  # [...,1]
                    hessian = torch.cat([hessian_xx, hessian_yy, hessian_zz], dim=-1)  # [...,3]
                else:
                    hessian = None
            elif self.cfg_sdf.gradient.taps == 4:
                eps = self.normal_eps / np.sqrt(3)
                k1 = torch.tensor([1, -1, -1], dtype=x.dtype, device=x.device)  # [3]
                k2 = torch.tensor([-1, -1, 1], dtype=x.dtype, device=x.device)  # [3]
                k3 = torch.tensor([-1, 1, -1], dtype=x.dtype, device=x.device)  # [3]
                k4 = torch.tensor([1, 1, 1], dtype=x.dtype, device=x.device)  # [3]
                sdf1 = self.sdf(x + k1 * eps)  # [...,1]
                sdf2 = self.sdf(x + k2 * eps)  # [...,1]
                sdf3 = self.sdf(x + k3 * eps)  # [...,1]
                sdf4 = self.sdf(x + k4 * eps)  # [...,1]
                gradient = (k1*sdf1 + k2*sdf2 + k3*sdf3 + k4*sdf4) / (4.0 * eps)
                if training:
                    assert sdf is not None  # computed when feed-forwarding through the network
                    # the result of 4 taps is directly trace, but we assume they are individual components
                    # so we use the same signature as 6 taps
                    hessian_xx = ((sdf1 + sdf2 + sdf3 + sdf4) / 2.0 - 2 * sdf) / eps ** 2   # [N,1]
                    hessian = torch.cat([hessian_xx, hessian_xx, hessian_xx], dim=-1) / 3.0
                else:
                    hessian = None
            else:
                raise ValueError("Only support 4 or 6 taps.")
        return gradient, hessian


class NeuralRGB(torch.nn.Module):

    def __init__(self, cfg_rgb, feat_dim, appear_embed):
        super().__init__()
        self.cfg_rgb = cfg_rgb
        self.cfg_appear_embed = appear_embed
        encoding_view_dim = self.build_encoding(cfg_rgb.encoding_view)
        input_base_dim = 6 if cfg_rgb.mode == "idr" else 3
        input_dim = input_base_dim + encoding_view_dim + feat_dim + (appear_embed.dim if appear_embed.enabled else 0)
        self.build_mlp(cfg_rgb.mlp, input_dim=input_dim)

    def build_encoding(self, cfg_encoding_view):
        if cfg_encoding_view.type == "fourier":
            encoding_view_dim = 6 * cfg_encoding_view.levels
        elif cfg_encoding_view.type == "spherical":
            self.spherical_harmonic_encoding = partial(get_spherical_harmonics, levels=cfg_encoding_view.levels)
            encoding_view_dim = (cfg_encoding_view.levels + 1) ** 2
        else:
            raise NotImplementedError("Unknown encoding type")
        return encoding_view_dim

    def build_mlp(self, cfg_mlp, input_dim=3):
        # RGB prediction
        layer_dims = [input_dim] + [cfg_mlp.hidden_dim] * cfg_mlp.num_layers + [3]
        activ = get_activation(cfg_mlp.activ, **cfg_mlp.activ_params)
        self.mlp = nerf_util.MLPwithSkipConnection(layer_dims, skip_connection=cfg_mlp.skip, activ=activ,
                                                   use_weightnorm=cfg_mlp.weight_norm)

    def forward(self, points_3D, normals, rays_unit, feats, app):
        view_enc = self.encode_view(rays_unit)  # [...,LD]
        input_list = [points_3D, view_enc, normals, feats]
        if app is not None:
            input_list.append(app)
        if self.cfg_rgb.mode == "no_view_dir":
            input_list.remove(view_enc)
        if self.cfg_rgb.mode == "no_normal":
            input_list.remove(normals)
        input_vec = torch.cat(input_list, dim=-1)
        rgb = self.mlp(input_vec).sigmoid_()
        return rgb  # [...,3]

    def encode_view(self, rays_unit):
        if self.cfg_rgb.encoding_view.type == "fourier":
            view_enc = nerf_util.positional_encoding(rays_unit, num_freq_bases=self.cfg_rgb.encoding_view.levels)
        elif self.cfg_rgb.encoding_view.type == "spherical":
            view_enc = self.spherical_harmonic_encoding(rays_unit)
        else:
            raise NotImplementedError("Unknown encoding type")
        return view_enc


class BackgroundNeRF(torch.nn.Module):

    def __init__(self, cfg_background, appear_embed):
        super().__init__()
        self.cfg_background = cfg_background
        self.cfg_appear_embed = appear_embed
        encoding_dim, encoding_view_dim = self.build_encoding(cfg_background.encoding, cfg_background.encoding_view)
        input_dim = 4 + encoding_dim
        input_view_dim = cfg_background.mlp.hidden_dim + encoding_view_dim + \
            (appear_embed.dim if appear_embed.enabled else 0)
        self.build_mlp(cfg_background.mlp, input_dim=input_dim, input_view_dim=input_view_dim)

    def build_encoding(self, cfg_encoding, cfg_encoding_view):
        # Positional encoding.
        if cfg_encoding.type == "fourier":
            encoding_dim = 8 * cfg_encoding.levels
        else:
            raise NotImplementedError("Unknown encoding type")
        # View encoding.
        if cfg_encoding_view.type == "fourier":
            encoding_view_dim = 6 * cfg_encoding_view.levels
        elif cfg_encoding_view.type == "spherical":
            self.spherical_harmonic_encoding = partial(get_spherical_harmonics, levels=cfg_encoding_view.levels)
            encoding_view_dim = (cfg_encoding_view.levels + 1) ** 2
        else:
            raise NotImplementedError("Unknown encoding type")
        return encoding_dim, encoding_view_dim

    def build_mlp(self, cfg_mlp, input_dim=3, input_view_dim=3):
        activ = get_activation(cfg_mlp.activ, **cfg_mlp.activ_params)
        # Point-wise feature.
        layer_dims = [input_dim] + [cfg_mlp.hidden_dim] * (cfg_mlp.num_layers - 1) + [cfg_mlp.hidden_dim + 1]
        self.mlp_feat = nerf_util.MLPwithSkipConnection(layer_dims, skip_connection=cfg_mlp.skip, activ=activ)
        self.activ_density = get_activation(cfg_mlp.activ_density, **cfg_mlp.activ_density_params)
        # RGB prediction.
        layer_dims_rgb = [input_view_dim] + [cfg_mlp.hidden_dim_rgb] * (cfg_mlp.num_layers_rgb - 1) + [3]
        self.mlp_rgb = nerf_util.MLPwithSkipConnection(layer_dims_rgb, skip_connection=cfg_mlp.skip_rgb, activ=activ)

    def forward(self, points_3D, rays_unit, app_outside):
        points_enc = self.encode(points_3D)  # [...,4+LD]
        # Volume density prediction.
        out = self.mlp_feat(points_enc)
        density, feat = self.activ_density(out[..., 0]), self.mlp_feat.activ(out[..., 1:])  # [...],[...,K]
        # RGB color prediction.
        if self.cfg_background.view_dep:
            view_enc = self.encode_view(rays_unit)  # [...,LD]
            input_list = [feat, view_enc]
            if app_outside is not None:
                input_list.append(app_outside)
            input_vec = torch.cat(input_list, dim=-1)
            rgb = self.mlp_rgb(input_vec).sigmoid_()  # [...,3]
        else:
            raise NotImplementedError
        return rgb, density

    def encode(self, points_3D):
        # Reparametrize the 3D points.
        # TODO: revive this.
        if True:
            points_3D_norm = points_3D.norm(dim=-1, keepdim=True)  # [B,R,N,1]
            points = torch.cat([points_3D / points_3D_norm, 1.0 / points_3D_norm], dim=-1)  # [B,R,N,4]
        else:
            points = points_3D
        # Positional encoding.
        if self.cfg_background.encoding.type == "fourier":
            points_enc = nerf_util.positional_encoding(points, num_freq_bases=self.cfg_background.encoding.levels)
        else:
            raise NotImplementedError("Unknown encoding type")
        # TODO: 1/x?
        points_enc = torch.cat([points, points_enc], dim=-1)  # [B,R,N,4+LD]
        return points_enc

    def encode_view(self, rays_unit):
        if self.cfg_background.encoding_view.type == "fourier":
            view_enc = nerf_util.positional_encoding(rays_unit, num_freq_bases=self.cfg_background.encoding_view.levels)
        elif self.cfg_background.encoding_view.type == "spherical":
            view_enc = self.spherical_harmonic_encoding(rays_unit)
        else:
            raise NotImplementedError("Unknown encoding type")
        return view_enc
