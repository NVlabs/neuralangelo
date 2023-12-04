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

import os
import numpy as np
import json
import sys
from pathlib import Path
from argparse import ArgumentParser
import trimesh

dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[2]
sys.path.append(dir_path.__str__())

from projects.neuralangelo.scripts.convert_data_to_json import export_to_json  # NOQA

from third_party.colmap.scripts.python.database import COLMAPDatabase  # NOQA
from third_party.colmap.scripts.python.read_write_model import read_model, rotmat2qvec  # NOQA


def create_init_files(pinhole_dict_file, db_file, out_dir):
    # Partially adapted from https://github.com/Kai-46/nerfplusplus/blob/master/colmap_runner/run_colmap_posed.py

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # create template
    with open(pinhole_dict_file) as fp:
        pinhole_dict = json.load(fp)

    template = {}
    cameras_line_template = '{camera_id} RADIAL {width} {height} {f} {cx} {cy} {k1} {k2}\n'
    images_line_template = '{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_name}\n\n'

    for img_name in pinhole_dict:
        # w, h, fx, fy, cx, cy, qvec, t
        params = pinhole_dict[img_name]
        w = params[0]
        h = params[1]
        fx = params[2]
        # fy = params[3]
        cx = params[4]
        cy = params[5]
        qvec = params[6:10]
        tvec = params[10:13]

        cam_line = cameras_line_template.format(
            camera_id="{camera_id}", width=w, height=h, f=fx, cx=cx, cy=cy, k1=0, k2=0)
        img_line = images_line_template.format(image_id="{image_id}", qw=qvec[0], qx=qvec[1], qy=qvec[2], qz=qvec[3],
                                               tx=tvec[0], ty=tvec[1], tz=tvec[2], camera_id="{camera_id}",
                                               image_name=img_name)
        template[img_name] = (cam_line, img_line)

    # read database
    db = COLMAPDatabase.connect(db_file)
    table_images = db.execute("SELECT * FROM images")
    img_name2id_dict = {}
    for row in table_images:
        img_name2id_dict[row[1]] = row[0]

    cameras_txt_lines = [template[img_name][0].format(camera_id=1)]
    images_txt_lines = []
    for img_name, img_id in img_name2id_dict.items():
        image_line = template[img_name][1].format(image_id=img_id, camera_id=1)
        images_txt_lines.append(image_line)

    with open(os.path.join(out_dir, 'cameras.txt'), 'w') as fp:
        fp.writelines(cameras_txt_lines)

    with open(os.path.join(out_dir, 'images.txt'), 'w') as fp:
        fp.writelines(images_txt_lines)
        fp.write('\n')

    # create an empty points3D.txt
    fp = open(os.path.join(out_dir, 'points3D.txt'), 'w')
    fp.close()


def convert_cam_dict_to_pinhole_dict(cam_dict, pinhole_dict_file):
    # Partially adapted from https://github.com/Kai-46/nerfplusplus/blob/master/colmap_runner/run_colmap_posed.py

    print('Writing pinhole_dict to: ', pinhole_dict_file)
    h = 1080
    w = 1920

    pinhole_dict = {}
    for img_name in cam_dict:
        W2C = cam_dict[img_name]

        # params
        fx = 0.6 * w
        fy = 0.6 * w
        cx = w / 2.0
        cy = h / 2.0

        qvec = rotmat2qvec(W2C[:3, :3])
        tvec = W2C[:3, 3]

        params = [w, h, fx, fy, cx, cy,
                  qvec[0], qvec[1], qvec[2], qvec[3],
                  tvec[0], tvec[1], tvec[2]]
        pinhole_dict[img_name] = params

    with open(pinhole_dict_file, 'w') as fp:
        json.dump(pinhole_dict, fp, indent=2, sort_keys=True)


def load_COLMAP_poses(cam_file, img_dir, tf='w2c'):
    # load img_dir namges
    names = sorted(os.listdir(img_dir))

    with open(cam_file) as f:
        lines = f.readlines()

    # C2W
    poses = {}
    for idx, line in enumerate(lines):
        if idx % 5 == 0:  # header
            img_idx, valid, _ = line.split(' ')
            if valid != '-1':
                poses[int(img_idx)] = np.eye(4)
                poses[int(img_idx)]
        else:
            if int(img_idx) in poses:
                num = np.array([float(n) for n in line.split(' ')])
                poses[int(img_idx)][idx % 5-1, :] = num

    if tf == 'c2w':
        return poses
    else:
        # convert to W2C (follow nerf convention)
        poses_w2c = {}
        for k, v in poses.items():
            poses_w2c[names[k]] = np.linalg.inv(v)
        return poses_w2c


def load_transformation(trans_file):
    with open(trans_file) as f:
        lines = f.readlines()

    trans = np.eye(4)
    for idx, line in enumerate(lines):
        num = np.array([float(n) for n in line.split(' ')])
        trans[idx, :] = num

    return trans


def align_gt_with_cam(pts, trans):
    trans_inv = np.linalg.inv(trans)
    pts_aligned = pts @ trans_inv[:3, :3].transpose(-1, -2) + trans_inv[:3, -1]
    return pts_aligned


def compute_bound(pts):
    bounding_box = np.array([pts.min(axis=0), pts.max(axis=0)])
    center = bounding_box.mean(axis=0)
    radius = np.max(np.linalg.norm(pts - center, axis=-1)) * 1.01
    return center, radius, bounding_box.T.tolist()


def init_colmap(args):
    assert args.tnt_path, "Provide path to Tanks and Temples dataset"
    scene_list = os.listdir(args.tnt_path)

    for scene in scene_list:
        scene_path = os.path.join(args.tnt_path, scene)

        if not os.path.exists(f"{scene_path}/images_raw"):
            raise Exception(f"'images_raw` folder cannot be found in {scene_path}."
                            "Please check the expected folder structure in DATA_PREPROCESSING.md")

        # extract features
        os.system(f"colmap feature_extractor --database_path {scene_path}/database.db \
                --image_path {scene_path}/images_raw \
                --ImageReader.camera_model=RADIAL \
                --SiftExtraction.use_gpu=true \
                --SiftExtraction.num_threads=32 \
                --ImageReader.single_camera=true"
                  )

        # match features
        os.system(f"colmap sequential_matcher \
                --database_path {scene_path}/database.db \
                --SiftMatching.use_gpu=true"
                  )

        # read poses
        poses = load_COLMAP_poses(os.path.join(scene_path, f'{scene}_COLMAP_SfM.log'),
                                  os.path.join(scene_path, 'images_raw'))

        # convert to colmap files
        pinhole_dict_file = os.path.join(scene_path, 'pinhole_dict.json')
        convert_cam_dict_to_pinhole_dict(poses, pinhole_dict_file)

        db_file = os.path.join(scene_path, 'database.db')
        sfm_dir = os.path.join(scene_path, 'sparse')
        create_init_files(pinhole_dict_file, db_file, sfm_dir)

        # bundle adjustment
        os.system(f"colmap point_triangulator \
                --database_path {scene_path}/database.db \
                --image_path {scene_path}/images_raw \
                --input_path {scene_path}/sparse \
                --output_path {scene_path}/sparse \
                --Mapper.tri_ignore_two_view_tracks=true"
                  )
        os.system(f"colmap bundle_adjuster \
                --input_path {scene_path}/sparse \
                --output_path {scene_path}/sparse \
                --BundleAdjustment.refine_extrinsics=false"
                  )

        # undistortion
        os.system(f"colmap image_undistorter \
            --image_path {scene_path}/images_raw \
            --input_path {scene_path}/sparse \
            --output_path {scene_path} \
            --output_type COLMAP \
            --max_image_size 1500"
                  )

        # read for bounding information
        trans = load_transformation(os.path.join(scene_path, f'{scene}_trans.txt'))
        pts = trimesh.load(os.path.join(scene_path, f'{scene}.ply'))
        pts = pts.vertices
        pts_aligned = align_gt_with_cam(pts, trans)
        center, radius, bounding_box = compute_bound(pts_aligned[::100])

        # colmap to json
        cameras, images, points3D = read_model(os.path.join(args.tnt_path, scene, 'sparse'), ext='.bin')
        export_to_json(cameras, images, bounding_box, list(center), radius, os.path.join(scene_path, 'transforms.json'))
        print('Writing data to json file: ', os.path.join(scene_path, 'transforms.json'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tnt_path', type=str, default=None, help='Path to tanks and temples dataset')

    args = parser.parse_args()

    init_colmap(args)
