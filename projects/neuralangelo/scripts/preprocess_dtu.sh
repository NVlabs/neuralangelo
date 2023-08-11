# -----------------------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# -----------------------------------------------------------------------------

# usage: dtu_download.sh <path_to_dtu>

echo "Download DTU data"
mkdir -p "${1}"
gdown 1zgD-uTLjO8hXcjLqelU444rwS9s9-Syg
unzip -q data_DTU.zip -d ${1}
rm data_DTU.zip
echo "Generate json files"
python3 projects/neuralangelo/scripts/convert_dtu_to_json.py --dtu_path ${1}
