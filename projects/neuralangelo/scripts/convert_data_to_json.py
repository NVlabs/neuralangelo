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
from argparse import ArgumentParser
import os
import sys
from pathlib import Path
import json
import math

dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[2]
sys.path.append(dir_path.__str__())

from third_party.colmap.scripts.python.read_write_model import read_model, qvec2rotmat  # NOQA


def find_closest_point(p1, d1, p2, d2):
    # Calculate the direction vectors of the lines
    d1_norm = d1 / np.linalg.norm(d1)
    d2_norm = d2 / np.linalg.norm(d2)

    # Create the coefficient matrix A and the constant vector b
    A = np.vstack((d1_norm, -d2_norm)).T
    b = p2 - p1

    # Solve the linear system to find the parameters t1 and t2
    t1, t2 = np.linalg.lstsq(A, b, rcond=None)[0]

    # Calculate the closest point on each line
    closest_point1 = p1 + d1_norm * t1
    closest_point2 = p2 + d2_norm * t2

    # Calculate the average of the two closest points
    closest_point = 0.5 * (closest_point1 + closest_point2)

    return closest_point


def bound_by_pose(images):
    poses = []
    for img in images.values():
        rotation = qvec2rotmat(img.qvec)
        translation = img.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([0, 0, 0, 1])[None]], 0)
        c2w = np.linalg.inv(w2c)
        poses.append(c2w)

    center = np.array([0.0, 0.0, 0.0])
    for f in poses:
        src_frame = f[0:3, :]
        for g in poses:
            tgt_frame = g[0:3, :]
            p = find_closest_point(src_frame[:, 3], src_frame[:, 2], tgt_frame[:, 3], tgt_frame[:, 2])
            center += p
    center /= len(poses)**2

    radius = 0.0
    for f in poses:
        radius += np.linalg.norm(f[0:3, 3])
    radius /= len(poses)
    bounding_box = [
        [center[0] - radius, center[0] + radius],
        [center[1] - radius, center[1] + radius],
        [center[2] - radius, center[2] + radius],
    ]
    return center, radius, bounding_box


def bound_by_points(points3D):
    xyzs = np.stack([point.xyz for point in points3D.values()])
    center = xyzs.mean(axis=0)
    std = xyzs.std(axis=0)
    radius = float(std.max() * 3)  # use 3*std to define the region, equivalent to 99% percentile
    bounding_box = [
        [center[0] - std[0] * 3, center[0] + std[0] * 3],
        [center[1] - std[1] * 3, center[1] + std[1] * 3],
        [center[2] - std[2] * 3, center[2] + std[2] * 3],
    ]
    return center, radius, bounding_box


def _cv_to_gl(cv):
    # convert to GL convention used in iNGP
    gl = cv * np.array([1, -1, -1, 1])
    return gl


def export_to_json(cameras, images, bounding_box, center, radius, file_path):
    intrinsic_param = np.array([camera.params for camera in cameras.values()])
    fl_x = intrinsic_param[0][0]  # TODO: only supports single camera for now
    fl_y = intrinsic_param[0][1]
    cx = intrinsic_param[0][2]
    cy = intrinsic_param[0][3]
    image_width = np.array([camera.width for camera in cameras.values()])
    image_height = np.array([camera.height for camera in cameras.values()])
    w = image_width[0]
    h = image_height[0]

    angle_x = math.atan(w / (fl_x * 2)) * 2
    angle_y = math.atan(h / (fl_y * 2)) * 2

    out = {
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "sk_x": 0.0,  # TODO: check if colmap has skew
        "sk_y": 0.0,
        "k1": 0.0,  # take undistorted images only
        "k2": 0.0,
        "k3": 0.0,
        "k4": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "is_fisheye": False,  # TODO: not supporting fish eye camera
        "cx": cx,
        "cy": cy,
        "w": int(w),
        "h": int(h),
        "aabb_scale": np.exp2(np.rint(np.log2(radius))),  # power of two, for INGP resolution computation
        "aabb_range": bounding_box,
        "sphere_center": center,
        "sphere_radius": radius,
        "frames": []
    }

    # read poses
    for img in sorted(images.values()):
        rotation = qvec2rotmat(img.qvec)
        translation = img.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([0, 0, 0, 1])[None]], 0)
        c2w = np.linalg.inv(w2c)
        c2w = _cv_to_gl(c2w)  # convert to GL convention used in iNGP

        frame = {"file_path": "images/" + img.name, "transform_matrix": c2w.tolist()}
        out["frames"].append(frame)

    with open(file_path, "w") as outputfile:
        json.dump(out, outputfile, indent=2)

    return


def auto_bound(args):
    cameras, images, points3D = read_model(os.path.join(args.data_dir, "sparse"), ext=".bin")

    # define bounding regions based on scene type
    if args.scene_type == "outdoor":
        center_points, radius_points, bounding_box_points = bound_by_points(points3D)
        center_pose, radius_pose, bounding_box_pose = bound_by_pose(images)
        # check the differences
        diff = np.linalg.norm(center_points - center_pose)
        squared_radii = radius_points + radius_pose
        diff = np.inf
        if diff / squared_radii < 0.05:
            # scene center is close to pose center with 95% overlap, use pose
            center, radius, bounding_box = center_pose, radius_pose, bounding_box_pose
        else:
            # different results, use sfm points
            center, radius, bounding_box = center_points, radius_points, bounding_box_points
    elif args.scene_type == "indoor":
        # use sfm points as a proxy to define bounding regions
        center, radius, bounding_box = bound_by_points(points3D)
    elif args.scene_type == "object":
        # use poses as a proxy to define bounding regions
        center, radius, bounding_box = bound_by_pose(images)
    else:
        raise TypeError("Unknown scene type")

    # export json file
    export_to_json(cameras, images, bounding_box, list(center), radius, os.path.join(args.data_dir, "transforms.json"))
    print("Writing data to json file: ", os.path.join(args.data_dir, "transforms.json"))
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None, help="Path to data")
    parser.add_argument("--scene_type", type=str, default="outdoor", choices=["outdoor", "indoor", "object"],
                        help="Select scene type. Outdoor for building-scale reconstruction; "
                        "indoor for room-scale reconstruction; object for object-centric scene reconstruction.")
    args = parser.parse_args()
    auto_bound(args)
