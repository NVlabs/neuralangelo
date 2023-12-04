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
import sys
from argparse import ArgumentParser
from pathlib import Path
import yaml
from addict import Dict
from PIL import Image, ImageFile

dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[2]
sys.path.append(dir_path.__str__())

ImageFile.LOAD_TRUNCATED_IMAGES = True


def generate_config(args):
    cfg = Dict()
    cfg._parent_ = "projects/neuralangelo/configs/base.yaml"
    num_images = len(os.listdir(os.path.join(args.data_dir, "images")))
    # model cfg
    if args.auto_exposure_wb:
        cfg.data.num_images = num_images
        cfg.model.appear_embed.enabled = True
        cfg.model.appear_embed.dim = 8
        if num_images < 4:  # default is 4
            cfg.data.val.subset = num_images
    else:
        cfg.model.appear_embed.enabled = False
    if args.scene_type == "outdoor":
        cfg.model.object.sdf.mlp.inside_out = False
        cfg.model.object.sdf.encoding.coarse2fine.init_active_level = 8
    elif args.scene_type == "indoor":
        cfg.model.object.sdf.mlp.inside_out = True
        cfg.model.object.sdf.encoding.coarse2fine.init_active_level = 8
        cfg.model.background.enabled = False
        cfg.model.render.num_samples.background = 0
    elif args.scene_type == "object":
        cfg.model.object.sdf.mlp.inside_out = False
        cfg.model.object.sdf.encoding.coarse2fine.init_active_level = 4
    else:
        raise TypeError("Unknown scene type")
    # data config
    cfg.data.type = "projects.neuralangelo.data"
    cfg.data.root = args.data_dir
    img = Image.open(os.path.join(args.data_dir, "images", os.listdir(os.path.join(args.data_dir, "images"))[0]))
    w, h = img.size
    cfg.data.train.image_size = [h, w]
    short_size = args.val_short_size
    cfg.data.val.image_size = [short_size, int(w/h*short_size)] if w > h else [int(h/w*short_size), short_size]
    cfg.data.readjust.center = [0., 0., 0.]
    cfg.data.readjust.scale = 1.
    # export cfg
    cfg_fname = os.path.join(dir_path, "projects/neuralangelo/configs", f"custom/{args.sequence_name}.yaml")
    with open(cfg_fname, "w") as file:
        yaml.safe_dump(cfg.to_dict(), file, default_flow_style=False, indent=4)
    print("Config generated to file:", cfg_fname)
    return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sequence_name", type=str, default="recon", help="Name of sequence")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to data")
    parser.add_argument("--auto_exposure_wb", action="store_true",
                        help="Video capture with auto-exposure or white-balance")
    parser.add_argument("--scene_type", type=str, default="outdoor", choices=["outdoor", "indoor", "object"],
                        help="Select scene type. Outdoor for building-scale reconstruction; "
                             "indoor for room-scale reconstruction; object for object-centric scene reconstruction.")
    parser.add_argument("--val_short_size", type=int, default=300,
                        help="Set the short side of validation images (for saving compute when rendering val images)")
    args = parser.parse_args()
    generate_config(args)
