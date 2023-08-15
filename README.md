# Neuralangelo
This is the official implementation of **Neuralangelo: High-Fidelity Neural Surface Reconstruction**.

[Zhaoshuo Li](https://mli0603.github.io/),
[Thomas Müller](https://tom94.net/),
[Alex Evans](https://research.nvidia.com/person/alex-evans),
[Russell H. Taylor](https://www.cs.jhu.edu/~rht/),
[Mathias Unberath](https://mathiasunberath.github.io/),
[Ming-Yu Liu](https://mingyuliu.net/),
[Chen-Hsuan Lin](https://chenhsuanlin.bitbucket.io/)  
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023

### [Project page](https://research.nvidia.com/labs/dir/neuralangelo/) | [Paper](https://arxiv.org/abs/2306.03092/)

<img src="assets/teaser.gif">

The code is built upon the Imaginaire library from the Deep Imagination Research Group at NVIDIA.  
For business inquiries, please submit the [NVIDIA research licensing form](https://www.nvidia.com/en-us/research/inquiries/).

--------------------------------------

## Installation
We offer two ways to setup the environment:
1. We provide prebuilt Docker images, where
    - `docker.io/chenhsuanlin/colmap:3.8` is for running COLMAP and the data preprocessing scripts. This includes the prebuilt COLMAP library (CUDA-supported).
    - `docker.io/chenhsuanlin/neuralangelo:23.04-py3` is for running the main Neuralangelo pipeline.

    The corresponding Dockerfiles can be found in the `docker` directory.
2. The conda environment for Neuralangelo. Install the dependencies and activate the environment `neuralangelo` with
    ```bash
    conda env create --file neuralangelo.yaml
    conda activate neuralangelo
    ```
For COLMAP, alternative installation options are also available on the [COLMAP website](https://colmap.github.io/).

--------------------------------------

## Data preparation
Please refer to [Data Preparation](DATA_PROCESSING.md) for step-by-step instructions.  
We assume known camera poses for each extracted frame from the video.
The code uses the same json format as [Instant NGP](https://github.com/NVlabs/instant-ngp).

--------------------------------------

## Run Neuralangelo!
```bash
EXPERIMENT=toy_example
GROUP=example_group
NAME=example_name
CONFIG=projects/neuralangelo/configs/custom/${EXPERIMENT}.yaml
GPUS=1  # use >1 for multi-GPU training!
torchrun --nproc_per_node=${GPUS} train.py \
    --logdir=logs/${GROUP}/${NAME} \
    --config=${CONFIG} \
    --show_pbar
```
Some useful notes:
- This codebase supports logging with [Weights & Biases](https://wandb.ai/site). You should have a W&B account for this.
    - Add `--wandb` to the command line argument to enable W&B logging.
    - Add `--wandb_name` to specify the W&B project name.
    - More detailed control can be found in the `init_wandb()` function in `imaginaire/trainers/base.py`.
- Configs can be overridden through the command line (e.g. `--optim.params.lr=1e-2`).
- Set `--checkpoint={CHECKPOINT_PATH}` to initialize with a certain checkpoint; set `--resume=True` to resume training.
- If appearance embeddings are enabled, make sure `data.num_images` is set to the number of training images.

--------------------------------------

## Isosurface extraction
Use the following command to run isosurface mesh extraction:
```bash
CHECKPOINT=logs/${GROUP}/${NAME}/xxx.pt
OUTPUT_MESH=xxx.ply
CONFIG=projects/neuralangelo/configs/custom/${EXPERIMENT}.yaml
RESOLUTION=2048
BLOCK_RES=128
GPUS=1  # use >1 for multi-GPU mesh extraction
torchrun --nproc_per_node=${GPUS} projects/neuralangelo/scripts/extract_mesh.py \
    --logdir=logs/${GROUP}/${NAME} \
    --config=${CONFIG} \
    --checkpoint=${CHECKPOINT} \
    --output_file=${OUTPUT_MESH} \
    --resolution=${RESOLUTION} \
    --block_res=${BLOCK_RES}
```

--------------------------------------

## Citation
If you find our code useful for your research, please cite
```
@inproceedings{li2023neuralangelo,
  title={Neuralangelo: High-Fidelity Neural Surface Reconstruction},
  author={Li, Zhaoshuo and M\"uller, Thomas and Evans, Alex and Taylor, Russell H and Unberath, Mathias and Liu, Ming-Yu and Lin, Chen-Hsuan},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition ({CVPR})},
  year={2023}
}
```
