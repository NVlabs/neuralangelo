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
import matplotlib.pyplot as plt
import k3d

from projects.nerf.utils import camera


def get_camera_mesh(pose, depth=1):
    vertices = torch.tensor([[-0.5, -0.5, 1],
                             [0.5, -0.5, 1],
                             [0.5, 0.5, 1],
                             [-0.5, 0.5, 1],
                             [0, 0, 0]]) * depth  # [6,3]
    faces = torch.tensor([[0, 1, 2],
                          [0, 2, 3],
                          [0, 1, 4],
                          [1, 2, 4],
                          [2, 3, 4],
                          [3, 0, 4]])  # [6,3]
    vertices = camera.cam2world(vertices[None], pose)  # [N,6,3]
    wireframe = vertices[:, [0, 1, 2, 3, 0, 4, 1, 2, 4, 3]]  # [N,10,3]
    return vertices, faces, wireframe


def merge_meshes(vertices, faces):
    mesh_N, vertex_N = vertices.shape[:2]
    faces_merged = torch.cat([faces + i * vertex_N for i in range(mesh_N)], dim=0)
    vertices_merged = vertices.view(-1, vertices.shape[-1])
    return vertices_merged, faces_merged


def merge_wireframes(wireframe):
    wf_first, wf_last, wf_dummy = wireframe[:, :1], wireframe[:, -1:], wireframe[:, :1] * np.nan
    wireframe_merged = torch.cat([wf_first, wireframe, wf_last, wf_dummy], dim=1)
    return wireframe_merged


def get_xyz_indicators(pose, length=0.1):
    xyz = torch.eye(4, 3)[None] * length
    xyz = camera.cam2world(xyz, pose)
    return xyz


def merge_xyz_indicators(xyz):  # [N,4,3]
    xyz = xyz[:, [[-1, 0], [-1, 1], [-1, 2]]]  # [N,3,2,3]
    xyz_0, xyz_1 = xyz.unbind(dim=2)  # [N,3,3]
    xyz_dummy = xyz_0 * np.nan
    xyz_merged = torch.stack([xyz_0, xyz_0, xyz_1, xyz_1, xyz_dummy], dim=2)  # [N,3,5,3]
    return xyz_merged


def k3d_visualize_pose(poses, vis_depth=0.5, xyz_length=0.1, center_size=0.1, xyz_width=0.02):
    # poses has shape [N,3,4] potentially in sequential order
    N = len(poses)
    centers_cam = torch.zeros(N, 1, 3)
    centers_world = camera.cam2world(centers_cam, poses)
    centers_world = centers_world[:, 0]
    # Get the camera wireframes.
    vertices, faces, wireframe = get_camera_mesh(poses, depth=vis_depth)
    xyz = get_xyz_indicators(poses, length=xyz_length)
    vertices_merged, faces_merged = merge_meshes(vertices, faces)
    wireframe_merged = merge_wireframes(wireframe)
    xyz_merged = merge_xyz_indicators(xyz)
    # Set the color map for the camera trajectory and the xyz indicators.
    color_map = plt.get_cmap("gist_rainbow")
    center_color = []
    vertices_merged_color = []
    wireframe_color = []
    xyz_color = []
    x_hex, y_hex, z_hex = int(255) << 16, int(255) << 8, int(255)
    for i in range(N):
        # Set the camera pose colors (with a smooth gradient color map).
        r, g, b, _ = color_map(i / (N - 1))
        r, g, b = r * 0.8, g * 0.8, b * 0.8
        pose_rgb_hex = (int(r * 255) << 16) + (int(g * 255) << 8) + int(b * 255)
        center_color += [pose_rgb_hex]
        vertices_merged_color += [pose_rgb_hex] * 5
        wireframe_color += [pose_rgb_hex] * 13
        # Set the xyz indicator colors.
        xyz_color += [x_hex] * 5 + [y_hex] * 5 + [z_hex] * 5
    # Plot in K3D.
    plot = k3d.plot(name="poses",
                    height=800,
                    camera_rotate_speed=5.0,
                    camera_zoom_speed=3.0,
                    camera_pan_speed=1.0,
                    )
    plot += k3d.points(centers_world,
                       colors=center_color,
                       point_size=center_size,
                       shader="3d",
                       )
    plot += k3d.mesh(vertices_merged, faces_merged,
                     colors=vertices_merged_color,
                     side="double",
                     opacity=0.05,
                     )
    plot += k3d.line(wireframe_merged,
                     colors=wireframe_color,
                     shader="simple",
                     )
    plot += k3d.line(xyz_merged,
                     colors=xyz_color,
                     shader="thick",
                     width=xyz_width,
                     )
    return plot
