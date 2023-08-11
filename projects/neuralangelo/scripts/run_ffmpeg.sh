# -----------------------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# -----------------------------------------------------------------------------

# usage: run_ffmpeg.sh <full_fname.mp4> <skip_frames>

image_path=${1%.*}_skip${2}/raw_images
mkdir -p ${image_path}
ffmpeg -i $1 -vf "select=not(mod(n\,$2))" -vsync vfr -q:v 2 ${image_path}/%06d.jpg
