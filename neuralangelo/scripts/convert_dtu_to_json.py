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
import json
from argparse import ArgumentParser
import os
import cv2
from PIL import Image, ImageFile
from glob import glob
import math
import sys
from pathlib import Path


dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[2]
sys.path.append(dir_path.__str__())
from projects.neuralangelo.scripts.convert_data_to_json import _cv_to_gl  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_K_Rt_from_P(filename, P=None):
    # This function is borrowed from IDR: https://github.com/lioryariv/idr
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def dtu_to_json(args):
    assert args.dtu_path, "Provide path to DTU dataset"
    scene_list = os.listdir(args.dtu_path)

    for scene in scene_list:
        scene_path = os.path.join(args.dtu_path, scene)
        if not os.path.isdir(scene_path) or 'scan' not in scene:
            continue

        out = {
            "k1": 0.0,  # take undistorted images only
            "k2": 0.0,
            "k3": 0.0,
            "k4": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "is_fisheye": False,
            "frames": []
        }

        camera_param = dict(np.load(os.path.join(scene_path, 'cameras_sphere.npz')))
        images_lis = sorted(glob(os.path.join(scene_path, 'image/*.png')))
        for idx, image in enumerate(images_lis):
            image = os.path.basename(image)

            world_mat = camera_param['world_mat_%d' % idx]
            scale_mat = camera_param['scale_mat_%d' % idx]

            # scale and decompose
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsic_param, c2w = load_K_Rt_from_P(None, P)
            c2w_gl = _cv_to_gl(c2w)

            frame = {"file_path": 'image/' + image, "transform_matrix": c2w_gl.tolist()}
            out["frames"].append(frame)

        fl_x = intrinsic_param[0][0]
        fl_y = intrinsic_param[1][1]
        cx = intrinsic_param[0][2]
        cy = intrinsic_param[1][2]
        sk_x = intrinsic_param[0][1]
        sk_y = intrinsic_param[1][0]
        w, h = Image.open(os.path.join(scene_path, 'image', image)).size

        angle_x = math.atan(w / (fl_x * 2)) * 2
        angle_y = math.atan(h / (fl_y * 2)) * 2

        scale_mat = scale_mat.astype(float)

        out.update({
            "camera_angle_x": angle_x,
            "camera_angle_y": angle_y,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "cx": cx,
            "cy": cy,
            "sk_x": sk_x,
            "sk_y": sk_y,
            "w": int(w),
            "h": int(h),
            "aabb_scale": np.exp2(np.rint(np.log2(scale_mat[0, 0]))),  # power of two, for INGP resolution computation
            "sphere_center": [0., 0., 0.],
            "sphere_radius": 1.,
        })

        file_path = os.path.join(scene_path, 'transforms.json')
        with open(file_path, "w") as outputfile:
            json.dump(out, outputfile, indent=2)
        print('Writing data to json file: ', file_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dtu_path', type=str, default=None)

    args = parser.parse_args()

    dtu_to_json(args)
