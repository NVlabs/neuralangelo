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
import torch.nn.functional as torch_F
import copy
from functools import partial
from collections import defaultdict

from imaginaire.models.base import Model as BaseModel
from projects.nerf.utils import camera, render, nerf_util


class Model(BaseModel):

    def __init__(self, cfg_model, cfg_data):
        super().__init__(cfg_model, cfg_data)
        self.num_rays = cfg_model.rand_rays
        self.image_size = cfg_data.image_size
        self.fine_sampling = cfg_model.fine_sampling
        self.stratified = cfg_model.sample_stratified
        self.density_reg = cfg_model.density_noise_reg
        self.opaque_background = cfg_model.opaque_background
        self.bgcolor = getattr(cfg_data, "bgcolor", 1.)
        # Define models.
        self.nerf = NeRF(cfg_model)
        if self.fine_sampling:
            self.nerf_fine = NeRF(cfg_model)
        # Define functions.
        self.ray_generator = partial(nerf_util.ray_generator,
                                     image_size=cfg_data.image_size,
                                     camera_ndc=cfg_model.camera_ndc,
                                     num_rays=cfg_model.rand_rays)
        self.sample_dists = partial(nerf_util.sample_dists,
                                    dist_range=cfg_model.dist.range,
                                    intvs=cfg_model.sample_intvs)
        self.sample_dists_from_pdf = partial(nerf_util.sample_dists_from_pdf,
                                             intvs_fine=cfg_model.sample_intvs_fine)
        self.reparametrize_dist = partial(nerf_util.reparametrize_dist,
                                          param_type=cfg_model.dist.param)
        self.get_inverse_depth = partial(nerf_util.get_inverse_depth,
                                         camera_ndc=cfg_model.camera_ndc)
        self.to_full_image = lambda vec, image_size=cfg_data.image_size: \
            vec.moveaxis(1, -1).unflatten(dim=-1, sizes=image_size)

    def forward(self, data):
        # Randomly sample and render the pixels.
        ray_idx = self._sample_random_rays(data)
        output = self.render_pixels(data["pose"], data["intr"], full_image=False, ray_idx=ray_idx,
                                    stratified=self.stratified, density_reg=self.density_reg)
        output.update(ray_idx=ray_idx)  # [B,R]
        return output

    def _sample_random_rays(self, data):
        batch_size = len(data["pose"])
        num_pixels = self.image_size[0] * self.image_size[1]
        ray_idx = torch.rand(batch_size, num_pixels, device=data["pose"].device).argsort(dim=1)[:, :self.num_rays]
        return ray_idx  # [B,R]

    @torch.no_grad()
    def inference(self, data):
        self.eval()
        # Render the full images.
        output = self.render_image(data["pose"], data["intr"], stratified=False)  # [B,N,C]
        # Get full rendered RGB and depth images.
        inv_depth = self.get_inverse_depth(output["depth"], opacity=output["opacity"])
        output.update(
            rgb_map=self.to_full_image(output["rgb"]),  # [B,3,H,W]
            inv_depth_map=self.to_full_image(inv_depth),  # [B,1,H,W]
        )
        if self.fine_sampling:
            inv_depth_fine = self.get_inverse_depth(output["depth_fine"], opacity=output["opacity_fine"])
            output.update(
                rgb_map_fine=self.to_full_image(output["rgb_fine"]),  # [B,3,H,W]
                inv_depth_map_fine=self.to_full_image(inv_depth_fine),  # [B,1,H,W]
            )
        return output

    def render_image(self, pose, intr, stratified=False):
        """ Render the rays given the camera intrinsics and poses.
        Args:
            pose (tensor [batch,3,4]): Camera poses ([R,t]).
            intr (tensor [batch,3,3]): Camera intrinsics.
            stratified (bool): Whether to stratify the depth sampling.
        Returns:
            output: A dictionary containing the outputs.
        """
        output = defaultdict(list)
        for center, ray, _ in self.ray_generator(pose, intr, full_image=True):
            ray_unit = torch_F.normalize(ray, dim=-1)  # [B,R,3]
            output_batch = self.render_rays(center, ray_unit, stratified=stratified)
            if not self.training:
                depth = output_batch["dist"] / ray.norm(dim=-1, keepdim=True)
                output_batch.update(depth=depth)
                if self.fine_sampling:
                    depth_fine = output_batch["dist_fine"] / ray.norm(dim=-1, keepdim=True)
                    output_batch.update(depth_fine=depth_fine)
            for key, value in output_batch.items():
                output[key].append(value.detach())
        # Concat each item (list) in output into one tensor. Concatenate along the ray dimension (1)
        for key, value in output.items():
            output[key] = torch.cat(value, dim=1)
        return output

    def render_pixels(self, pose, intr, full_image=False, ray_idx=None, stratified=False, density_reg=None):
        center, ray = camera.get_center_and_ray(pose, intr, self.image_size)  # [B,HW,3]
        center = nerf_util.slice_by_ray_idx(center, ray_idx)  # [B,R,3]
        ray = nerf_util.slice_by_ray_idx(ray, ray_idx)  # [B,R,3]
        ray_unit = torch_F.normalize(ray, dim=-1)  # [B,R,3]
        output = self.render_rays(center, ray_unit, stratified=stratified, density_reg=density_reg)
        return output

    def render_rays(self, center, ray_unit, sample_idx=None, stratified=False, density_reg=None):
        with torch.no_grad():
            dists = self.sample_dists(ray_unit.shape[:2], stratified=stratified)  # [B,R,N,1]
            dists = self.reparametrize_dist(dists)  # [B,R,N,1]
        points = camera.get_3D_points_from_dist(center, ray_unit, dists)  # [B,R,N,3]
        rays_unit = ray_unit[..., None, :].expand_as(points)  # [B,R,N,3]
        rgbs, densities = self.nerf.forward(points, rays_unit, density_reg=density_reg)
        weights = render.volume_rendering_weights_dist(densities, dists)  # [B,R,N,1]
        opacity = render.composite(1., weights)  # [B,R,1]
        rgb = render.composite(rgbs, weights)  # [B,R,3]
        if self.opaque_background:
            rgb = rgb + self.bgcolor * (1 - opacity)
        dist = render.composite(dists, weights)  # [B,R,1]
        # Collect output.
        output = dict(
            rgb=rgb,  # [B,R,3]
            dist=dist,  # [B,R,1]
            opacity=opacity,  # [B,R,1]
        )
        if self.fine_sampling:
            with torch.no_grad():
                # Resample depth according to coarse empirical distribution.
                dists_mid = 0.5 * (dists[..., :-1, :] + dists[..., 1:, :])  # [B,R,N-1,1]
                dists_fine = self.sample_dists_from_pdf(dists_mid, weights=weights[..., 1:-1, 0])  # [B,R,Nf,1]
                dists = torch.cat([dists, dists_fine], dim=2)  # [B,R,N+Nf,1]
                dists = dists.sort(dim=2).values
            points = camera.get_3D_points_from_dist(center, ray_unit, dists)  # [B,R,N,3]
            rays_unit = ray_unit[..., None, :].expand_as(points)  # [B,R,N,3]
            rgbs, densities = self.nerf_fine.forward(points, rays_unit, density_reg=density_reg)
            weights = render.volume_rendering_weights_dist(densities, dists)  # [B,R,N,1]
            opacity = render.composite(1., weights)  # [B,R,1]
            rgb = render.composite(rgbs, weights)  # [B,R,3]
            if self.opaque_background:
                rgb = rgb + self.bgcolor * (1 - opacity)
            dist = render.composite(dists, weights)  # [B,R,1]
            # Collect output.
            output.update(
                rgb_fine=rgb,  # [B,R,3]
                dist_fine=dist,  # [B,R,1]
                opacity_fine=opacity,  # [B,R,1]
            )
        return output


class NeRF(torch.nn.Module):

    def __init__(self, cfg_model):
        super().__init__()
        self.view_dep = cfg_model.view_dep
        self.posenc = cfg_model.posenc
        # Define learnable parameters.
        self.set_input_dims(cfg_model)
        self.build_model(cfg_model)

    def set_input_dims(self, cfg_model):
        # Define the input encoding dimensions.
        self.input_3D_dim = 3 + 6 * cfg_model.posenc.L_3D if cfg_model.posenc.L_3D else 3
        if cfg_model.view_dep:
            self.input_view_dim = 3 + 6 * cfg_model.posenc.L_view if cfg_model.posenc.L_view else 3
        else:
            self.input_view_dim = None

    def build_model(self, cfg_model):
        # Point-wise feature.
        layers_feat = copy.copy(cfg_model.mlp.layers_feat)
        layers_feat[0] = self.input_3D_dim
        layers_feat[-1] += 1  # For predicting the volume density.
        self.mlp_feat = nerf_util.MLPwithSkipConnection(layers_feat, skip_connection=cfg_model.mlp.skip)
        # RGB prediction.
        layers_rgb = copy.copy(cfg_model.mlp.layers_rgb)
        layers_rgb[0] = cfg_model.mlp.layers_feat[-1] + (self.input_view_dim if cfg_model.view_dep else 0)
        self.mlp_rgb = nerf_util.MLPwithSkipConnection(layers_rgb)
        self.density_activ = dict(
            relu=torch_F.relu,
            relu_=torch_F.relu_,
            abs=torch.abs,
            abs_=torch.abs_,
            sigmoid=torch.sigmoid,
            sigmoid_=torch.sigmoid_,
            exp=torch.exp,
            exp_=torch.exp_,
            softplus=torch_F.softplus,
            identity=lambda x: x,
        )[cfg_model.density_activ]

    def forward(self, points_3D, ray_unit, density_reg=None):
        """ Forward pass the NeRF (MLP).
        Args:
            points_3D (tensor [batch,...,3]): 3D points in world space.
            ray_unit (tensor [batch,...,3]): Unit ray direction in world space.
            density_reg (float or None): Density regularization.
        Returns:
            rgb (tensor [batch,...,3]): Predicted RGB values in [0,1].
            density (tensor [batch,...,3]): Predicted volume density values.
        """
        density, feat = self.get_density(points_3D, density_reg=density_reg)
        rgb = self.get_color(feat, ray_unit)
        return rgb, density

    def get_density(self, points_3D, density_reg=None):
        points_enc = self._encode_3D(points_3D)
        out = self.mlp_feat(points_enc)
        density, feat = out[..., 0], out[..., 1:].relu_()
        if density_reg is not None:
            density = density + torch.randn_like(density) * density_reg
        density = self.density_activ(density)
        return density, feat

    def get_color(self, feat, ray_unit=None):
        if self.view_dep:
            ray_enc = self._encode_view(ray_unit)
            feat = torch.cat([feat, ray_enc], dim=-1)
        rgb = self.mlp_rgb(feat).sigmoid_()  # [B,...,3]
        return rgb

    def _encode_3D(self, points_3D):
        if self.posenc.L_3D:
            points_enc = nerf_util.positional_encoding(points_3D, num_freq_bases=self.posenc.L_3D)
            points_enc = torch.cat([points_3D, points_enc], dim=-1)  # [B,...,6L+3]
        else:
            points_enc = points_3D
        return points_enc

    def _encode_view(self, ray_unit):
        if self.posenc.L_view:
            ray_enc = nerf_util.positional_encoding(ray_unit, num_freq_bases=self.posenc.L_view)
            ray_enc = torch.cat([ray_unit, ray_enc], dim=-1)  # [B,...,6L+3]
        else:
            ray_enc = ray_unit
        return ray_enc
