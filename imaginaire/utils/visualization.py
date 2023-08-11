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

import cv2
import numpy as np
import PIL
import wandb
from PIL import Image
import torch
import torchvision
import os

from matplotlib import pyplot as plt
from torchvision.transforms import functional as torchvision_F


def save_tensor_image(
        filename, image, minus1to1_normalized=False):
    r"""Convert a 3 dimensional torch tensor to a PIL image with the desired
    width and height.

    Args:
        filename (str): Image filename to be saved to.
        image (3 x W1 x H1 tensor): Image tensor
        minus1to1_normalized (bool): True if the tensor values are in [-1,
        1]. Otherwise, we assume the values are in [0, 1].

    Returns:
        (PIL image): The resulting PIL image.
    """
    if len(image.size()) != 3:
        raise ValueError('Image tensor dimension does not equal = 3.')
    if image.size(0) != 3:
        raise ValueError('Image has more than 3 channels.')
    if minus1to1_normalized:
        # Normalize back to [0, 1]
        image = (image + 1) * 0.5
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)
    image_grid = torchvision.utils.make_grid(
        image, nrow=1, padding=0, normalize=False)
    torchvision.utils.save_image(image_grid, filename, nrow=1)
    return


def tensor2pilimage(image, width=None, height=None, minus1to1_normalized=False):
    r"""Convert a 3 dimensional torch tensor to a PIL image with the desired
    width and height.

    Args:
        image (3 x W1 x H1 tensor): Image tensor
        width (int): Desired width for the result PIL image.
        height (int): Desired height for the result PIL image.
        minus1to1_normalized (bool): True if the tensor values are in [-1,
        1]. Otherwise, we assume the values are in [0, 1].

    Returns:
        (PIL image): The resulting PIL image.
    """
    if len(image.size()) != 3:
        raise ValueError('Image tensor dimension does not equal = 3.')
    if image.size(0) != 3:
        raise ValueError('Image has more than 3 channels.')
    if minus1to1_normalized:
        # Normalize back to [0, 1]
        image = (image + 1) * 0.5
    image = image.detach().cpu().squeeze().numpy()
    image = np.transpose(image, (1, 2, 0)) * 255
    output_img = Image.fromarray(np.uint8(image))
    if width is not None and height is not None:
        output_img = output_img.resize((width, height), Image.BICUBIC)
    return output_img


def tensor2im(image_tensor, imtype=np.uint8, normalize=True,
              three_channel_output=True):
    r"""Convert tensor to image.

    Args:
        image_tensor (torch.tensor or list of torch.tensor): If tensor then
            (NxCxHxW) or (NxTxCxHxW) or (CxHxW).
        imtype (np.dtype): Type of output image.
        normalize (bool): Is the input image normalized or not?
            three_channel_output (bool): Should single channel images be made 3
            channel in output?

    Returns:
        (numpy.ndarray, list if case 1, 2 above).
    """
    if image_tensor is None:
        return None
    if isinstance(image_tensor, list):
        return [tensor2im(x, imtype, normalize) for x in image_tensor]
    if image_tensor.dim() == 5 or image_tensor.dim() == 4:
        return [tensor2im(image_tensor[idx], imtype, normalize)
                for idx in range(image_tensor.size(0))]

    if image_tensor.dim() == 3:
        image_numpy = image_tensor.detach().cpu().float().numpy()
        if normalize:
            image_numpy = (np.transpose(
                image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        else:
            image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
        image_numpy = np.clip(image_numpy, 0, 255)
        if image_numpy.shape[2] == 1 and three_channel_output:
            image_numpy = np.repeat(image_numpy, 3, axis=2)
        elif image_numpy.shape[2] > 3:
            image_numpy = image_numpy[:, :, :3]
        return image_numpy.astype(imtype)


def tensor2label(segmap, n_label=None, imtype=np.uint8,
                 colorize=True, output_normalized_tensor=False):
    r"""Convert segmentation mask tensor to color image.
    Args:
        segmap (tensor) of
        If tensor then (NxCxHxW) or (NxTxCxHxW) or (CxHxW).
        n_label (int): If None, then segmap.size(0).
        imtype (np.dtype): Type of output image.
        colorize (bool): Put colors in.

    Returns:
        (numpy.ndarray or normalized torch image).
    """
    if segmap is None:
        return None
    if isinstance(segmap, list):
        return [tensor2label(x, n_label,
                             imtype, colorize,
                             output_normalized_tensor) for x in segmap]
    if segmap.dim() == 5 or segmap.dim() == 4:
        return [tensor2label(segmap[idx], n_label,
                             imtype, colorize,
                             output_normalized_tensor)
                for idx in range(segmap.size(0))]

    segmap = segmap.float()
    if not output_normalized_tensor:
        segmap = segmap.cpu()
    if n_label is None:
        n_label = segmap.size(0)
    if n_label > 1:
        segmap = segmap.max(0, keepdim=True)[1]

    if output_normalized_tensor:
        if n_label == 0:
            segmap = Colorize(256)(segmap).to('cuda')
        else:
            segmap = Colorize(n_label)(segmap).to('cuda')
        return 2 * (segmap.float() / 255) - 1
    else:
        if colorize:
            if n_label == 0:
                segmap = Colorize(256)(segmap)
            else:
                segmap = Colorize(n_label)(segmap)
            segmap = np.transpose(segmap.numpy(), (1, 2, 0))
        else:
            segmap = segmap.cpu().numpy()
        return segmap.astype(imtype)


def tensor2flow(tensor, imtype=np.uint8):
    r"""Convert flow tensor to color image.

    Args:
        tensor (tensor) of
        If tensor then (NxCxHxW) or (NxTxCxHxW) or (CxHxW).
        imtype (np.dtype): Type of output image.

    Returns:
        (numpy.ndarray or normalized torch image).
    """
    if tensor is None:
        return None
    if isinstance(tensor, list):
        tensor = [t for t in tensor if t is not None]
        if not tensor:
            return None
        return [tensor2flow(t, imtype) for t in tensor]
    if tensor.dim() == 5 or tensor.dim() == 4:
        return [tensor2flow(tensor[b]) for b in range(tensor.size(0))]

    tensor = tensor.detach().cpu().float().numpy()
    tensor = np.transpose(tensor, (1, 2, 0))

    hsv = np.zeros((tensor.shape[0], tensor.shape[1], 3), dtype=imtype)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(tensor[..., 0], tensor[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def plot_keypoints(image, keypoints, normalize=True):
    r"""Plot keypoints on image.

    Args:
       image (PIL.Image, or numpy.ndarray, or torch.Tensor): Input image.
       keypoints (np.ndarray or torch.Tensor, Nx2): Keypoint locations.
       normalize (bool): Whether to normalize the image or not.
    """
    if isinstance(image, PIL.Image.Image):
        image = np.array(image)
    if isinstance(image, torch.Tensor):
        image = tensor2im(image, normalize=normalize)
    if isinstance(image, np.ndarray):
        assert image.ndim == 3
        assert image.shape[-1] == 1 or image.shape[-1] == 3
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.cpu().numpy()
    assert keypoints.ndim == 2 and keypoints.shape[1] == 2

    cv2_image = np.ascontiguousarray(image[:, :, ::-1])  # RGB to BGR.
    for idx in range(keypoints.shape[0]):
        keypoint = np.round(keypoints[idx]).astype(np.int)
        cv2_image = cv2.circle(cv2_image, tuple(keypoint),
                               5, (0, 255, 0), -1)
    image = np.ascontiguousarray(cv2_image[:, :, ::-1])
    return image


def labelcolormap(N):
    r"""Create colors for segmentation label ids.

    Args:
        N (int): Number of labels.
    """
    if N == 35:  # GTA/cityscape train
        cmap = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
                         (111, 74, 0), (81, 0, 81), (128, 64, 128),
                         (244, 35, 232), (250, 170, 160), (230, 150, 140),
                         (70, 70, 70), (102, 102, 156), (190, 153, 153),
                         (180, 165, 180), (150, 100, 100), (150, 120, 90),
                         (153, 153, 153), (153, 153, 153), (250, 170, 30),
                         (220, 220, 0), (107, 142, 35), (152, 251, 152),
                         (70, 130, 180), (220, 20, 60), (255, 0, 0),
                         (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 0, 90),
                         (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32),
                         (0, 0, 142)],
                        dtype=np.uint8)
    elif N == 20:  # GTA/cityscape eval
        cmap = np.array([(128, 64, 128), (244, 35, 232), (70, 70, 70),
                         (102, 102, 156), (190, 153, 153), (153, 153, 153),
                         (250, 170, 30), (220, 220, 0), (107, 142, 35),
                         (152, 251, 152), (220, 20, 60), (255, 0, 0),
                         (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100),
                         (0, 0, 230), (119, 11, 32), (70, 130, 180), (0, 0, 0)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros([N, 3]).astype(np.uint8)
        for i in range(N):
            r, g, b = np.zeros(3)
            for j in range(8):
                r = r + (1 << (7 - j)) * ((i & (1 << (3 * j))) >> (3 * j))
                g = g + (1 << (7 - j)) * \
                    ((i & (1 << (3 * j + 1))) >> (3 * j + 1))
                b = b + (1 << (7 - j)) * \
                    ((i & (1 << (3 * j + 2))) >> (3 * j + 2))
            cmap[i, :] = np.array([r, g, b])
    return cmap


class Colorize(object):
    """Class to colorize segmentation maps."""

    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, seg_map):
        r"""

        Args:
            seg_map (tensor): Input Segmentation maps to be colorized.
        """
        size = seg_map.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
        for label in range(0, len(self.cmap)):
            mask = (label == seg_map[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]
        return color_image


def plot_keypoints_on_black(resize_h, resize_w, crop_h, crop_w, is_flipped,
                            cfgdata, keypoints):
    r"""Plot keypoints on black image.

    Args:
        resize_h (int): Height to be resized to.
        resize_w (int): Width to be resized to.
        crop_h (int): Height of the cropping.
        crop_w (int): Width of the cropping.
        is_flipped (bool): If image is a flipped version.
        cfgdata (obj): Data configuration object.
        keypoints (np.ndarray): Keypoint locations. Shape of
            (Nx2) or (TxNx2).

    Returns:
        (list of np.ndarray): List of images (output_h, output_w, 3).
    """
    if keypoints.ndim == 2 and keypoints.shape[1] == 2:
        keypoints = keypoints[np.newaxis, ...]

    outputs = []
    for t_idx in range(keypoints.shape[0]):
        cv2_image = np.zeros((crop_h, crop_w, 3)).astype(np.uint8)
        for idx in range(keypoints[t_idx].shape[0]):
            keypoint = np.round(keypoints[t_idx][idx]).astype(np.int)
            cv2_image = cv2.circle(cv2_image, tuple(keypoint),
                                   5, (0, 255, 0), -1)
        image = np.ascontiguousarray(cv2_image[:, :, ::-1])  # BGR to RGB.
        outputs.append(image)

    return outputs


def wandb_image(images, from_range=(0, 1)):
    images = preprocess_image(images, from_range=from_range)
    image_grid = torchvision.utils.make_grid(images, nrow=1, pad_value=1)
    image_grid = torchvision_F.to_pil_image(image_grid)
    wandb_image = wandb.Image(image_grid)
    return wandb_image


def preprocess_image(images, from_range=(0, 1), cmap="gray"):
    min, max = from_range
    images = (images - min) / (max - min)
    images = images.detach().cpu().float().clamp_(min=0, max=1)
    if images.shape[1] == 1:
        images = get_heatmap(images[:, 0], cmap=cmap)
    return images


def get_heatmap(gray, cmap):  # [N,H,W]
    color = plt.get_cmap(cmap)(gray.numpy())
    color = torch.from_numpy(color[..., :3]).permute(0, 3, 1, 2).float()  # [N,3,H,W]
    return color
