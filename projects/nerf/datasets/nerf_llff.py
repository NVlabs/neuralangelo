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
import torchvision.transforms.functional as torchvision_F
from PIL import Image, ImageFile

from projects.nerf.datasets import base
from projects.nerf.utils import camera

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(base.Dataset):

    def __init__(self, cfg, is_inference=False, is_test=False):
        super().__init__(cfg, is_inference=is_inference, is_test=is_test)
        cfg_data = cfg.test_data if self.split == "test" else cfg.data
        data_info = cfg_data[self.split]
        self.root = cfg_data.root
        self.preload = cfg_data.preload
        self.raw_H, self.raw_W = 3024, 4032
        self.H, self.W = cfg_data.image_size
        list_fname = f"{cfg_data.root}/images.list"
        image_fnames = open(list_fname).read().splitlines()
        poses_raw, bounds = self.parse_cameras_and_bounds(cfg_data)
        self.list = list(zip(image_fnames, poses_raw, bounds))
        # Manually split train/val subsets.
        num_val_split = int(len(self) * cfg_data.val_ratio)
        self.list = self.list[:-num_val_split] if self.split == "train" else self.list[-num_val_split:]
        # Consider only a subset of data.
        if data_info.subset:
            self.list = self.list[:data_info.subset]
        # Preload dataset if possible.
        if cfg_data.preload:
            self.images = self.preload_threading(self.get_image, cfg_data.num_workers)
            self.cameras = self.preload_threading(self.get_camera, cfg_data.num_workers, data_str="cameras")

    def parse_cameras_and_bounds(self, cfg_data):
        fname = f"{cfg_data.root}/poses_bounds.npy"
        data = torch.tensor(np.load(fname), dtype=torch.float32)
        # Parse cameras (intrinsics and poses).
        cam_data = data[:, :-2].view([-1, 3, 5])  # [N,3,5]
        poses_raw = cam_data[..., :4]  # [N,3,4]
        poses_raw[..., 0], poses_raw[..., 1] = poses_raw[..., 1], -poses_raw[..., 0]
        raw_H, raw_W, self.focal = cam_data[0, :, -1]
        assert self.raw_H == raw_H and self.raw_W == raw_W
        # Parse depth bounds.
        bounds = data[:, -2:]  # [N,2]
        scale = 1. / (bounds.min() * 0.75)  # Not sure how this was determined?
        poses_raw[..., 3] *= scale
        bounds *= scale
        # Roughly center camera poses.
        poses_raw = self.center_camera_poses(poses_raw)
        return poses_raw, bounds

    def center_camera_poses(self, poses):
        # Compute average pose.
        center = poses[..., 3].mean(dim=0)
        v1 = torch_F.normalize(poses[..., 1].mean(dim=0), dim=0)
        v2 = torch_F.normalize(poses[..., 2].mean(dim=0), dim=0)
        v0 = v1.cross(v2)
        pose_avg = torch.stack([v0, v1, v2, center], dim=-1)[None]  # [1,3,4]
        # Apply inverse of averaged pose.
        poses = camera.pose.compose([poses, camera.pose.invert(pose_avg)])
        return poses

    def __getitem__(self, idx):
        """Process raw data and return processed data in a dictionary.

        Args:
            idx: The index of the sample of the dataset.
        Returns: A dictionary containing the data.
                 idx (scalar): The index of the sample of the dataset.
                 image (3xHxW tensor): Image with pixel values in [0,1] for supervision.
                 intr (3x3 tensor): The camera intrinsics of `image`.
                 pose (3x4 tensor): The camera extrinsics [R,t] of `image`.
        """
        # Keep track of sample index for convenience.
        sample = dict(idx=idx)
        # Get the images.
        image = self.images[idx] if self.preload else self.get_image(idx)
        image = self.preprocess_image(image)
        # Get the cameras (intrinsics and pose).
        intr, pose = self.cameras[idx] if self.preload else self.get_camera(idx)
        intr, pose = self.preprocess_camera(intr, pose)
        # Update the data sample.
        sample.update(
            image=image,
            intr=intr,
            pose=pose,
        )
        return sample

    def get_image(self, idx):
        image_fname = f"{self.root}/images/{self.list[idx][0]}"
        image = Image.open(image_fname)
        image.load()
        return image

    def preprocess_image(self, image):
        # Resize the image and convert to Pytorch.
        image = image.resize((self.W, self.H))
        image = torchvision_F.to_tensor(image)
        return image

    def get_camera(self, idx):
        # Camera intrinsics.
        intr = torch.tensor([[self.focal, 0, self.raw_W / 2],
                             [0, self.focal, self.raw_H / 2],
                             [0, 0, 1]]).float()
        # Camera pose.
        pose_raw = self.list[idx][1]
        pose = self.parse_raw_camera(pose_raw)
        return intr, pose

    def preprocess_camera(self, intr, pose):
        # Adjust the intrinsics according to the resized image.
        intr = intr.clone()
        intr[0] *= self.W / self.raw_W
        intr[1] *= self.H / self.raw_H
        return intr, pose

    def parse_raw_camera(self, pose_raw):
        pose_flip = camera.pose(R=torch.diag(torch.tensor([1, -1, -1])))
        pose = camera.pose.compose([pose_flip, pose_raw[:3]])
        pose = camera.pose.invert(pose)
        pose = camera.pose.compose([pose_flip, pose])
        return pose
