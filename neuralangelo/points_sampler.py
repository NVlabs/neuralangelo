import torch
import nerfacc
import torch.nn.functional as F
from loguru import logger
from neuralangelo.utils.rays import RayBundle, RaySample


class OccGridSampler:
    def __init__(self, cfg, aabb: torch.Tensor):
        """
        aabb: 2x3
        """
        self.cfg = cfg
        self.estimator = nerfacc.OccGridEstimator(
            roi_aabb=aabb.view(-1),
            resolution=cfg.resolution,
            levels=cfg.levels,
        )

        if not self.cfg.grid_prune:
            self.estimator.occs.fill_(True)
            self.estimator.binaries.fill_(True)

        self.render_step_size = (
            1.732 * 2 * self.cfg.radius / self.cfg.num_samples_per_ray
        )

    def get_alpha(self, sdf, normal, dirs, dists):
        inv_std = self.variance(sdf)

        true_cos = (dirs * normal).sum(-1, keepdim=True)
        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio)
            + F.relu(-true_cos) * self.cos_anneal_ratio
        )  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_std)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_std)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
        return alpha

    @staticmethod
    def validate_empty_rays(ray_indices, t_start, t_end):
        if ray_indices.nelement() == 0:
            logger.warning("Empty rays_indices!")
            ray_indices = torch.LongTensor([0]).to(ray_indices)
            t_start = torch.Tensor([0]).to(ray_indices)
            t_end = torch.Tensor([0]).to(ray_indices)
        return ray_indices, t_start, t_end

    @torch.no_grad()
    def sample(
        self,
        ray_bundle: RayBundle,
        alpha_fn,
    ):
        rays_o_flatten = ray_bundle.origins.reshape(-1, 3).contiguous()
        rays_d_flatten = ray_bundle.directions.reshape(-1, 3).contiguous()
        t_min = None
        t_max = None
        if ray_bundle.nears is not None and ray_bundle.fars is not None:
            t_min = ray_bundle.nears.contiguous().reshape(-1)
            t_max = ray_bundle.fars.contiguous().reshape(-1)
        # camera_indices = None
        # if ray_bundle.camera_indices is not None:
        #     camera_indices = ray_bundle.camera_indices.contiguous()

        ray_indices, starts, ends = self.occupancy_grid.sampling(
            rays_o=rays_o_flatten,
            rays_d=rays_d_flatten,
            alpha_fn=alpha_fn,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            t_min=t_min,
            t_max=t_max,
            render_step_size=self.render_step_size,
            stratified=self.training,
            cone_angle=self.cfg.cone_angle,
            alpha_thre=self.cfg.alpha_thre,
        )
        ray_indices, starts, ends = self.validate_empty_rays(ray_indices, starts, ends)
        ray_indices = ray_indices.long()
        t_starts = starts[..., None]
        t_ends = ends[..., None]
        t_origins = rays_o_flatten[ray_indices]
        t_dirs = rays_d_flatten[ray_indices]
        t_positions = (t_starts + t_ends) / 2.0
        points = t_origins + t_dirs * t_positions
        t_intervals = t_ends - t_starts
        return RaySample(
            points=points,
            positions=t_positions,
            intervals=t_intervals,
            directions=t_dirs,
        )
