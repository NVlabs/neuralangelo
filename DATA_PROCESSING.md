# Data Preparation

The following sections provide a step-by-step guide on how to convert a video to a json file that Neuralangelo parses.

## Prerequisites
Initialize the COLMAP submodule:
```bash
git submodule update --init --recursive
```

## Self-captured video sequence
To capture your own data, we recommend using a high shutter speed to avoid motion blur (which is very common when using a phone camera). A [toy example](https://drive.google.com/file/d/1VJeWYNJEBK0MFIzHjI8xZzwf3_I3eLwH/view?usp=drive_link) is provided for testing the workflow. There are two steps:
1. [preprocessing](#preprocessing) the data and running COLMAP,
2. [inspecting](#inspect-and-adjust-colmap-results) and refining the bounding sphere of interest for running Neuralangelo.

### Preprocessing
You can run the following command to preprocess your data:

```bash
EXPERIMENT_NAME=toy_example
PATH_TO_VIDEO=toy_example.MOV
SKIP_FRAME_RATE=4  # If video motion is small, set this to large values (e.g., 24). If video motion is large, set this to small values (e.g., 4).
SCENE_TYPE=object  # {outdoor,indoor,object}
bash projects/neuralangelo/scripts/preprocess.sh ${EXPERIMENT_NAME} ${PATH_TO_VIDEO} ${SKIP_FRAME_RATE} ${SCENE_TYPE}
```

Alternatively, you can follow the steps below if you want more fine-grained control.

1. Convert video to images

    ```bash
    PATH_TO_VIDEO=toy_example.MOV
    SKIP_FRAME_RATE=4  # If video motion is small, set this to large values (e.g., 24). If video motion is large, set this to small values (e.g., 4).
    bash projects/neuralangelo/scripts/run_ffmpeg.sh ${PATH_TO_VIDEO} ${SKIP_FRAME_RATE}
    ```
    `PATH_TO_VIDEO`: path to video  
    `SKIP_FRAME_RATE`: downsampling rate (recommended 10 for 24 fps captured videos)

2. Run COLMAP

    ```bash
    PATH_TO_IMAGES=datasets/toy_example_skip4
    bash projects/neuralangelo/scripts/run_colmap.sh ${PATH_TO_IMAGES}
    ```
    `PATH_TO_IMAGES`: path to extracted images

    After COLMAP finishes, the folder structure will look like following:
    ```bash
    PATH_TO_IMAGES
    |__ database.db (COLMAP databse)
    |__ raw_images (raw input images)
    |__ dense
    |____ images (undistorted images)
    |____ sparse (COLMAP correspondences, intrinsics and sparse point cloud)
    |____ stereo (COLMAP files for MVS)
    ```
    `dense/images` will be the input for surface reconstruction.

3. Generate json file for data loading

    In this step, we define the bounding region for reconstruction and convert the COLMAP data to json format following Instant NGP. We strongly recommend you go to step 5 to validate the quality of the automatic bounding region extraction for improved performance.

    ```bash
    PATH_TO_IMAGES=datasets/toy_example_skip4
    SCENE_TYPE=object  # {outdoor,indoor,object}
    python3 projects/neuralangelo/scripts/convert_data_to_json.py --data_dir ${PATH_TO_IMAGES}/dense --scene_type ${SCENE_TYPE}
    ```
    `PATH_TO_IMAGES`: path to extracted images

4. Config files

    Use the following to configure and generate your config files
    ```bash
    EXPERIMENT_NAME=toy_example
    SCENE_TYPE=object  # {outdoor,indoor,object}
    python3 projects/neuralangelo/scripts/generate_config.py --experiment_name ${EXPERIMENT_NAME} --data_dir ${PATH_TO_IMAGES}/dense --scene_type ${SCENE_TYPE}
    ```
    The config file will be generated as `projects/neuralangelo/configs/custom/{EXPERIMENT_NAME}.yaml`.

    To find more arguments and how they work:
    ```bash
    python3 projects/neuralangelo/scripts/generate_config.py -h
    ```
    You can also manually adjust the parameters in the yaml file directly.

### Inspect and adjust COLMAP results

For certain cases, the camera poses estimated by COLMAP could be erroneous. In addition, the automated estimation of the bounding sphere could be inaccurate (which ideally should include the scene/object of interest). It is highly recommended that the bounding sphere is adjusted. 
We offer some tools to to inspect and adjust the pre-processing results. Below are some options:

- Blender: Download [Blender](https://www.blender.org/download/) and follow the instructions in our [add-on repo](https://github.com/mli0603/BlenderNeuralangelo). The add-on will save your adjustment of the bounding sphere.
- This [Jupyter notebook](projects/neuralangelo/scripts/visualize_colmap.ipynb) (using K3D) can be helpful for visualizing the COLMAP results. You can adjust the bounding sphere by manually specifying the refining sphere center and size in the `data.readjust` config.

If a continous camera recording trajectory turns into broken trajectories, it is because COLMAP fails due to ambiguous matches. You can try to change `exhaustive_matcher` to `sequential_matcher` in [run_colmap.sh file](https://github.com/NVlabs/neuralangelo/blob/main/projects/neuralangelo/scripts/run_colmap.sh#L24).

## DTU dataset
- Please use respecting the license terms of the dataset.

You can run the following command to download [the DTU dataset](https://roboimagedata.compute.dtu.dk/?page_id=36) that is preprocessed by NeuS authors and generate json files:
```bash
PATH_TO_DTU=datasets/dtu  # Modify this to be the DTU dataset root directory.
bash projects/neuralangelo/scripts/preprocess_dtu.sh ${PATH_TO_DTU}
```

## Tanks and Temples dataset
- Please use respecting the license terms of the dataset.

Download the data from [Tanks and Temples](https://tanksandtemples.org/download/) website.
You will also need to download additional [COLMAP/camera/alignment](https://drive.google.com/file/d/1jAr3IDvhVmmYeDWi0D_JfgiHcl70rzVE/view?resourcekey=) and the images of each scene.  
The file structure should look like (you may need to move around the downloaded images):
```
tanks_and_temples
├─ Barn
│  ├─ Barn_COLMAP_SfM.log (camera poses)
│  ├─ Barn.json (cropfiles)
│  ├─ Barn.ply (ground-truth point cloud)
│  ├─ Barn_trans.txt (colmap-to-ground-truth transformation)
│  └─ images (folder of images)
│     ├─ 000001.png
│     ├─ 000002.png
│     ...
├─ Caterpillar
│  ├─ ...
...
```
Run the following command to generate json files:
```bash
PATH_TO_TNT=datasets/tanks_and_temples  # Modify this to be the Tanks and Temples root directory.
bash projects/neuralangelo/scripts/preprocess_tnt.sh ${PATH_TO_TNT}
```
