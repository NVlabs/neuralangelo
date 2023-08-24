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
import plotly.graph_objs as go
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


def merge_wireframes_k3d(wireframe):
    wf_first, wf_last, wf_dummy = wireframe[:, :1], wireframe[:, -1:], wireframe[:, :1] * np.nan
    wireframe_merged = torch.cat([wf_first, wireframe, wf_last, wf_dummy], dim=1)
    return wireframe_merged


def merge_wireframes_plotly(wireframe):
    wf_dummy = wireframe[:, :1] * np.nan
    wireframe_merged = torch.cat([wireframe, wf_dummy], dim=1).view(-1, 3)
    return wireframe_merged


def get_xyz_indicators(pose, length=0.1):
    xyz = torch.eye(4, 3)[None] * length
    xyz = camera.cam2world(xyz, pose)
    return xyz


def merge_xyz_indicators_k3d(xyz):  # [N,4,3]
    xyz = xyz[:, [[-1, 0], [-1, 1], [-1, 2]]]  # [N,3,2,3]
    xyz_0, xyz_1 = xyz.unbind(dim=2)  # [N,3,3]
    xyz_dummy = xyz_0 * np.nan
    xyz_merged = torch.stack([xyz_0, xyz_0, xyz_1, xyz_1, xyz_dummy], dim=2)  # [N,3,5,3]
    return xyz_merged


def merge_xyz_indicators_plotly(xyz):  # [N,4,3]
    xyz = xyz[:, [[-1, 0], [-1, 1], [-1, 2]]]  # [N,3,2,3]
    xyz_0, xyz_1 = xyz.unbind(dim=2)  # [N,3,3]
    xyz_dummy = xyz_0 * np.nan
    xyz_merged = torch.stack([xyz_0, xyz_1, xyz_dummy], dim=2)  # [N,3,3,3]
    xyz_merged = xyz_merged.view(-1, 3)
    return xyz_merged


def k3d_visualize_pose(poses, vis_depth=0.5, xyz_length=0.1, center_size=0.1, xyz_width=0.02, mesh_opacity=0.05):
    # poses has shape [N,3,4] potentially in sequential order
    N = len(poses)
    centers_cam = torch.zeros(N, 1, 3)
    centers_world = camera.cam2world(centers_cam, poses)
    centers_world = centers_world[:, 0]
    # Get the camera wireframes.
    vertices, faces, wireframe = get_camera_mesh(poses, depth=vis_depth)
    xyz = get_xyz_indicators(poses, length=xyz_length)
    vertices_merged, faces_merged = merge_meshes(vertices, faces)
    wireframe_merged = merge_wireframes_k3d(wireframe)
    xyz_merged = merge_xyz_indicators_k3d(xyz)
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
    k3d_objects = [
        k3d.points(centers_world, colors=center_color, point_size=center_size, shader="3d"),
        k3d.mesh(vertices_merged, faces_merged, colors=vertices_merged_color, side="double", opacity=mesh_opacity),
        k3d.line(wireframe_merged, colors=wireframe_color, shader="simple"),
        k3d.line(xyz_merged, colors=xyz_color, shader="thick", width=xyz_width),
    ]
    return k3d_objects


def plotly_visualize_pose(poses, vis_depth=0.5, xyz_length=0.5, center_size=2, xyz_width=5, mesh_opacity=0.05):
    # poses has shape [N,3,4] potentially in sequential order
    N = len(poses)
    centers_cam = torch.zeros(N, 1, 3)
    centers_world = camera.cam2world(centers_cam, poses)
    centers_world = centers_world[:, 0]
    # Get the camera wireframes.
    vertices, faces, wireframe = get_camera_mesh(poses, depth=vis_depth)
    xyz = get_xyz_indicators(poses, length=xyz_length)
    vertices_merged, faces_merged = merge_meshes(vertices, faces)
    wireframe_merged = merge_wireframes_plotly(wireframe)
    xyz_merged = merge_xyz_indicators_plotly(xyz)
    # Break up (x,y,z) coordinates.
    wireframe_x, wireframe_y, wireframe_z = wireframe_merged.unbind(dim=-1)
    xyz_x, xyz_y, xyz_z = xyz_merged.unbind(dim=-1)
    centers_x, centers_y, centers_z = centers_world.unbind(dim=-1)
    vertices_x, vertices_y, vertices_z = vertices_merged.unbind(dim=-1)
    # Set the color map for the camera trajectory and the xyz indicators.
    color_map = plt.get_cmap("gist_rainbow")
    center_color = []
    faces_merged_color = []
    wireframe_color = []
    xyz_color = []
    x_color, y_color, z_color = *np.eye(3).T,
    for i in range(N):
        # Set the camera pose colors (with a smooth gradient color map).
        r, g, b, _ = color_map(i / (N - 1))
        rgb = np.array([r, g, b]) * 0.8
        wireframe_color += [rgb] * 11
        center_color += [rgb]
        faces_merged_color += [rgb] * 6
        xyz_color += [x_color] * 3 + [y_color] * 3 + [z_color] * 3
    # Plot in plotly.
    plotly_traces = [
        go.Scatter3d(x=wireframe_x, y=wireframe_y, z=wireframe_z, mode="lines",
                     line=dict(color=wireframe_color, width=1)),
        go.Scatter3d(x=xyz_x, y=xyz_y, z=xyz_z, mode="lines", line=dict(color=xyz_color, width=xyz_width)),
        go.Scatter3d(x=centers_x, y=centers_y, z=centers_z, mode="markers",
                     marker=dict(color=center_color, size=center_size, opacity=1)),
        go.Mesh3d(x=vertices_x, y=vertices_y, z=vertices_z,
                  i=[f[0] for f in faces_merged], j=[f[1] for f in faces_merged], k=[f[2] for f in faces_merged],
                  facecolor=faces_merged_color, opacity=mesh_opacity),
    ]
    return plotly_traces
