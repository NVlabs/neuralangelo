# -----------------------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# -----------------------------------------------------------------------------

# usage: preprocess.sh <sequence_name> <full_video_path> <downsample_rate> <scene_type>

data_path=datasets/${1}_ds${3}
bash projects/neuralangelo/scripts/run_ffmpeg.sh ${1} ${2} ${3}
bash projects/neuralangelo/scripts/run_colmap.sh ${data_path}
python3 projects/neuralangelo/scripts/convert_data_to_json.py --data_dir ${data_path} --scene_type ${4}
python3 projects/neuralangelo/scripts/generate_config.py --sequence_name ${1} --data_dir ${data_path} --scene_type ${4}
