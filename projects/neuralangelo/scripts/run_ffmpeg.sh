# -----------------------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# -----------------------------------------------------------------------------

# usage: run_ffmpeg.sh <sequence_name> <full_video_path> <downsample_rate>

data_path=datasets/${1}_ds${3}
image_path=${data_path}/images_raw
mkdir -p ${image_path}
ffmpeg -i ${2} -vf "select=not(mod(n\,$3))" -vsync vfr -q:v 2 ${image_path}/%06d.jpg
