import torch
import torch.nn.functional as torch_F
from typing import Dict
from neuralangelo.utils import nerf_util, camera
from neuralangelo.utils.rays import RayBundle


class RaySampler(object):
    def __init__(self, cfg):
        """
        cfg = cfg.data
        """
        self.cfg = cfg
        self.train_imsize = self.cfg.train.image_size
        self.max_train_raynum = self.train_imsize[0] * self.train_imsize[1]
        self.val_imsize = self.cfg.val.image_size

    @torch.no_grad()
    def sample(self, data: Dict):
        # for training
        # [B,HW,3]
        center, ray = camera.get_center_and_ray(
            data["pose"], data["intr"], self.train_imsize
        )
        center = nerf_util.slice_by_ray_idx(center, data["ray_idx"])  # [B,R,3]
        ray = nerf_util.slice_by_ray_idx(ray, data["ray_idx"])  # [B,R,3]
        ray_unit = torch_F.normalize(ray, dim=-1)  # [B,R,3]
        return RayBundle(
            origins=center,
            directions=ray_unit,
            camera_indices=data["idx"],
        )

    @torch.no_grad()
    def ray_generator(self, data: Dict):
        # for inference
        for center, ray, ray_idx in nerf_util.ray_generator(
            pose=data["pose"],
            intr=data["intr"],
            image_size=self.val_imsize,
            full_image=True,
            camera_ndc=False,
            num_rays=self.cfg.rand_rays,
        ):
            yield RayBundle(
                origins=center,
                directions=ray,
                rays_index=ray_idx,
                camera_indices=data["idx"],
            )
