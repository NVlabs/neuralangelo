# -----------------------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# -----------------------------------------------------------------------------

# usage: run_colmap.sh <project_path>

mkdir -p ${1}/sparse
mkdir -p ${1}/dense

colmap feature_extractor \
    --database_path ${1}/database.db \
    --image_path ${1}/raw_images \
    --ImageReader.camera_model=RADIAL \
    --SiftExtraction.use_gpu=true \
    --SiftExtraction.num_threads=32 \
    --ImageReader.single_camera=true # assuming single camera

colmap sequential_matcher \
    --database_path ${1}/database.db \
    --SiftMatching.use_gpu=true

colmap mapper \
    --database_path ${1}/database.db \
    --image_path ${1}/raw_images \
    --output_path ${1}/sparse

colmap image_undistorter \
    --image_path ${1}/raw_images \
    --input_path ${1}/sparse/0 \
    --output_path ${1}/dense \
    --output_type COLMAP \
    --max_image_size 2000

rm -rf ${1}/sparse
