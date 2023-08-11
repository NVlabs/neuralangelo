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

import json
import numpy as np
import torch
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
        self.bgcolor = cfg_data.bgcolor
        self.raw_H, self.raw_W = 800, 800
        self.H, self.W = cfg_data.image_size
        meta_fname = f"{cfg_data.root}/transforms_{self.split}.json"
        with open(meta_fname) as file:
            self.meta = json.load(file)
        self.focal = 0.5 * self.raw_W / np.tan(0.5 * self.meta["camera_angle_x"])
        self.list = self.meta["frames"]
        # Consider only a subset of data.
        if data_info.subset:
            self.list = self.list[:data_info.subset]
        # Preload dataset if possible.
        if cfg_data.preload:
            self.images = self.preload_threading(self.get_image, cfg_data.num_workers)
            self.cameras = self.preload_threading(self.get_camera, cfg_data.num_workers, data_str="cameras")

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
        fpath = self.list[idx]["file_path"][2:]
        image_fname = f"{self.root}/{fpath}.png"
        image = Image.open(image_fname)
        image.load()
        return image

    def preprocess_image(self, image):
        # Resize the image.
        image = image.resize((self.W, self.H))
        image = torchvision_F.to_tensor(image)
        # Background masking.
        rgb, mask = image[:3], image[3:]
        if self.bgcolor is not None:
            rgb = rgb * mask + self.bgcolor * (1 - mask)
        return rgb

    def get_camera(self, idx):
        # Camera intrinsics.
        intr = torch.tensor([[self.focal, 0, self.raw_W / 2],
                             [0, self.focal, self.raw_H / 2],
                             [0, 0, 1]]).float()
        # Camera pose.
        pose_raw = torch.tensor(self.list[idx]["transform_matrix"], dtype=torch.float32)
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
        return pose
