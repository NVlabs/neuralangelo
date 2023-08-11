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

from projects.nerf.utils import camera


def sample_dists(ray_size, dist_range, intvs, stratified, device="cuda"):
    """Sample points on ray shooting from pixels using distance.
    Args:
        ray_size (int [2]): Integers for [batch size, number of rays].
        range (float [2]): Range of distance (depth) [min, max] to be sampled on rays.
        intvs: (int): Number of points sampled on a ray.
        stratified: (bool): Use stratified sampling or constant 0.5 sampling.
    Returns:
        dists (tensor [batch_size, num_ray, intvs, 1]): Sampled distance for all rays in a batch.
    """
    batch_size, num_rays = ray_size
    dist_min, dist_max = dist_range
    if stratified:
        rands = torch.rand(batch_size, num_rays, intvs, 1, device=device)
    else:
        rands = torch.empty(batch_size, num_rays, intvs, 1, device=device).fill_(0.5)
    rands += torch.arange(intvs, dtype=torch.float, device=device)[None, None, :, None]  # [B,R,N,1]
    dists = rands / intvs * (dist_max - dist_min) + dist_min  # [B,R,N,1]
    return dists


def sample_dists_from_pdf(bin, weights, intvs_fine):
    """Sample points on ray shooting from pixels using the weights from the coarse NeRF.
    Args:
        bin (tensor [batch_size, num_rays, intvs]): bins of distance values from the coarse NeRF.
        weights (tensor [batch_size, num_rays, intvs]): weights from the coarse NeRF.
        intvs_fine: (int): Number of fine-grained points sampled on a ray.
    Returns:
        dists (tensor [batch_size, num_ray, intvs, 1]): Sampled distance for all rays in a batch.
    """
    pdf = torch_F.normalize(weights, p=1, dim=-1)
    # Get CDF from PDF (along last dimension).
    cdf = pdf.cumsum(dim=-1)  # [B,R,N]
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # [B,R,N+1]
    # Take uniform samples.
    grid = torch.linspace(0, 1, intvs_fine + 1, device=pdf.device)  # [Nf+1]
    unif = 0.5 * (grid[:-1] + grid[1:]).repeat(*cdf.shape[:-1], 1)  # [B,R,Nf]
    idx = torch.searchsorted(cdf, unif, right=True)  # [B,R,Nf] \in {1...N}
    # Inverse transform sampling from CDF.
    low = (idx - 1).clamp(min=0)  # [B,R,Nf]
    high = idx.clamp(max=cdf.shape[-1] - 1)  # [B,R,Nf]
    dist_min = bin[..., 0].gather(dim=2, index=low)  # [B,R,Nf]
    dist_max = bin[..., 0].gather(dim=2, index=high)  # [B,R,Nf]
    cdf_low = cdf.gather(dim=2, index=low)  # [B,R,Nf]
    cdf_high = cdf.gather(dim=2, index=high)  # [B,R,Nf]
    # Linear interpolation.
    t = (unif - cdf_low) / (cdf_high - cdf_low + 1e-8)  # [B,R,Nf]
    dists = dist_min + t * (dist_max - dist_min)  # [B,R,Nf]
    return dists[..., None]  # [B,R,Nf,1]


def reparametrize_dist(dist, param_type="metric"):
    """Reparametrize the sampled distance values according to param_type.
    Args:
        dist (tensor): Sampled distance values.
        param_type (str): Reparametrization type.
    Returns:
        dist_new (tensor): Reparametrized distance values.
    """
    return dict(
        metric=dist,
        ndc=dist,
        inverse=1 / (dist + 1e-8),
    )[param_type]


def ray_generator(pose, intr, image_size, num_rays, full_image=False, camera_ndc=False,
                  ray_indices=None):
    """Yield sampled rays for coordinate-based model to predict NeRF.
    Args:
        pose (tensor [bs,3,4]): Camera poses ([R,t]).
        intr (tensor [bs,3,3]): Camera intrinsics.
        image_size: (tensor [bs,2]): Image size [height, width].
        num_rays (int): Number of rays to sample (random rays unless full_image=True).
        full_image (bool): Sample rays from the full image.
        camera_ndc (bool): Use normalized device coordinate for camera.
    Returns:
        center_slice (tensor [bs, ray, 3]): Sampled 3-D center in the world coordinate.
        ray_slice (tensor [bs, ray, 3]): Sampled 3-D ray in the world coordinate.
        ray_idx (tensor [bs, ray]): Sampled indices to index sampled pixels on images.
    """
    # Create a grid of centers and rays on an image.
    batch_size = pose.shape[0]
    # We used to randomly sample ray indices here. Now, we assume they are pre-generated and passed in.
    if ray_indices is None:
        num_pixels = image_size[0] * image_size[1]
        if full_image:
            # Sample rays from the full image.
            ray_indices = torch.arange(0, num_pixels, device=pose.device).repeat(batch_size, 1)  # [B,HW]
        else:
            # Sample rays randomly. The below is equivalent to batched torch.randperm().
            ray_indices = torch.rand(batch_size, num_pixels, device=pose.device).argsort(dim=1)[:, :num_rays]  # [B,R]
    center, ray = camera.get_center_and_ray(pose, intr, image_size)  # [B,HW,3]
    # Convert center/ray representations to NDC if necessary.
    if camera_ndc == "new":
        center, ray = camera.convert_NDC2(center, ray, intr=intr)
    elif camera_ndc:
        center, ray = camera.convert_NDC(center, ray, intr=intr)
    # Yield num_rays of sampled rays in each iteration (when random, the loop will only iterate once).
    for c in range(0, ray_indices.shape[1], num_rays):
        ray_idx = ray_indices[:, c:c + num_rays]  # [B,R]
        batch_idx = torch.arange(batch_size, device=pose.device).repeat(ray_idx.shape[1], 1).t()  # [B,R]
        center_slice = center[batch_idx, ray_idx]  # [B,R,3]
        ray_slice = ray[batch_idx, ray_idx]  # [B,R,3]
        yield center_slice, ray_slice, ray_idx


def slice_by_ray_idx(var, ray_idx):
    batch_size, num_rays = ray_idx.shape[:2]
    batch_idx = torch.arange(batch_size, device=ray_idx.device).repeat(num_rays, 1).t()  # [B,R]
    var_slice = var[batch_idx, ray_idx]  # [B,R,...]
    return var_slice


def positional_encoding(input, num_freq_bases):
    """Encode input into position codes.
    Args:
        input (tensor [bs, ..., N]): A batch of data with N dimension.
        num_freq_bases: (int): The number of frequency base of the code.
    Returns:
        input_enc (tensor [bs, ..., 2*N*num_freq_bases]): Positional codes for input.
    """
    freq = 2 ** torch.arange(num_freq_bases, dtype=torch.float32, device=input.device) * np.pi  # [L].
    spectrum = input[..., None] * freq  # [B,...,N,L].
    sin, cos = spectrum.sin(), spectrum.cos()  # [B,...,N,L].
    input_enc = torch.stack([sin, cos], dim=-2)  # [B,...,N,2,L].
    input_enc = input_enc.view(*input.shape[:-1], -1)  # [B,...,2NL].
    return input_enc


def get_inverse_depth(depth, opacity=None, camera_ndc=False, eps=1e-10):
    # Compute inverse depth for visualization.
    if opacity is not None:
        return (1 - depth) / opacity if camera_ndc else 1 / (depth / opacity + eps)
    else:
        return (1 - depth) if camera_ndc else 1 / (depth + eps)


class MLPwithSkipConnection(torch.nn.Module):

    def __init__(self, layer_dims, skip_connection=[], activ=None, use_layernorm=False, use_weightnorm=False):
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
        layer_dim_pairs = list(zip(layer_dims[:-1], layer_dims[1:]))
        for li, (k_in, k_out) in enumerate(layer_dim_pairs):
            if li in self.skip_connection:
                k_in += layer_dims[0]
            linear = torch.nn.Linear(k_in, k_out)
            if use_weightnorm:
                linear = torch.nn.utils.weight_norm(linear)
            self.linears.append(linear)
            if use_layernorm and li != len(layer_dim_pairs) - 1:
                self.layer_norm.append(torch.nn.LayerNorm(k_out))
            if li == len(layer_dim_pairs) - 1:
                self.linears[-1].bias.data.fill_(0.0)
        self.activ = activ or torch_F.relu_

    def forward(self, input):
        feat = input
        for li, linear in enumerate(self.linears):
            if li in self.skip_connection:
                feat = torch.cat([feat, input], dim=-1)
            feat = linear(feat)
            if li != len(self.linears) - 1:
                if self.use_layernorm:
                    feat = self.layer_norm[li](feat)
                feat = self.activ(feat)
        return feat


def intersect_with_sphere(center, ray_unit, radius=1.0):
    ctc = (center * center).sum(dim=-1, keepdim=True)  # [...,1]
    ctv = (center * ray_unit).sum(dim=-1, keepdim=True)  # [...,1]
    b2_minus_4ac = ctv ** 2 - (ctc - radius ** 2)
    dist_near = -ctv - b2_minus_4ac.sqrt()
    dist_far = -ctv + b2_minus_4ac.sqrt()
    return dist_near, dist_far


def get_pixel_radii(intr):
    # fx and fy should be very close.
    focal = (intr[..., 0, 0] + intr[..., 1, 1]) / 2
    radii = 1. / focal / np.sqrt(3)
    return radii


def contract(x, r_in=1, r_out=2, eps=1e-8):
    """ Contract function in mip-NeRF 360 (eq 10).
    Args:
        x (tensor [...,3]): The input points.
    Returns:
        x_warp (tensor [...,3]): The warped points.
    """
    x_norm = x.norm(dim=-1, keepdim=True)  # [...,1]
    scale = r_out - r_in * (r_out - r_in) / (x_norm + eps)  # [...,1]
    x_contract = scale * torch_F.normalize(x, dim=-1)  # [...,3]
    # No effect if within r_in.
    inside = x_norm <= r_in
    x_warp = torch.where(inside, x, x_contract)  # [...,3]
    return x_warp


def contract_jacobian(x, r_in=1, r_out=2, eps=1e-8):
    """ Jacobian of the contract function in mip-NeRF 360.
    Args:
        x (tensor [...,3]): The input points.
    Returns:
        jacobian (tensor [...,3,3]): The Jacobian at the input points.
    """
    x_norm = x.norm(dim=-1)[..., None, None]  # [...,1,1]
    x_norm_sq = (x ** 2).sum(dim=-1)[..., None, None]  # [...,1,1] should be numerically more stable
    b = r_in * (r_out - r_in)
    scale = r_out - b / (x_norm + eps)  # [...,1,1]
    x_outer_prod = x[..., None] * x[..., None, :]  # [...,3,3]
    eye = torch.eye(3, device=x.device).repeat(*x_norm.shape)  # [...,3,3]
    term1 = b * x_outer_prod / (x_norm_sq ** 2 + eps)  # [...,3,3]
    term2 = scale * (eye - x_outer_prod / x_norm_sq + eps) / (x_norm + eps)  # [...,3,3]
    jacobian_contract = term1 + term2  # [...,3,3]
    # No effect if within r_in.
    inside = x_norm <= r_in
    jacobian = torch.where(inside, eye, jacobian_contract)  # [...,3]
    return jacobian


def contract_mip(mean, cov, r_in=1, r_out=2, diag=False):
    """ Contraction function on mip-NeRF 360 Gaussians.
    Args:
        mean (tensor [...,3]): The mean values.
        cov (tensor [...,3,3]): The covariance values.
        r_in (float): The radius of unaffected region.
        r_out (float): The radius of contracted region. r_in < r_out.
    Returns:
        mean_warp (tensor [...,3]): The contracted mean values.
        cov_warp (tensor [...,3,3]): The contracted covariance values.
    """
    mean_warp = contract(mean, r_in=r_in, r_out=r_out)  # [...,3]
    jacobian = contract_jacobian(mean, r_in=r_in, r_out=r_out)  # [...,3,3]
    if diag:
        cov_warp = (jacobian * cov[..., None, :]) @ jacobian.transpose(-2, -1)
    else:
        cov_warp = jacobian @ cov @ jacobian.transpose(-2, -1)
    return mean_warp, cov_warp
