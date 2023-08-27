# Data Preparation

*Note: please use respecting the license terms of each dataset. Each user is responsible for checking the content of datasets and the applicable licenses and determining if suitable for the intended use.*

The following sections provide a guide on how to preprocess input videos for Neuralangelo.

## Prerequisites
Initialize the COLMAP submodule:
```bash
git submodule update --init --recursive
```

## Self-captured video sequence
To capture your own data, we recommend using a high shutter speed to avoid motion blur (which is very common when using a phone camera). We provide a synthetic [Lego sequence](https://drive.google.com/file/d/1yWoZ4Hk3FgmV3pd34ZbW7jEqgqyJgzHy/view?usp=drive_link) (from the original [NeRF](https://github.com/bmild/nerf)) as a toy example video for testing the workflow. There are two steps:
1. [preprocessing](#preprocessing) the data and running COLMAP,
2. [inspecting](#inspect-and-adjust-colmap-results) and refining the bounding sphere of interest for running Neuralangelo.

### Preprocessing
First, set some environment variables:
```bash
SEQUENCE=lego
PATH_TO_VIDEO=lego.mp4
DOWNSAMPLE_RATE=2
SCENE_TYPE=object
```
where
- `SEQUENCE`: your custom name for the video sequence.
- `PATH_TO_VIDEO`: absolute/relative path to your video.
- `DOWNSAMPLE_RATE`: temporal downsampling rate of video sequence (for extracting video frames).
- `SCENE_TYPE`: can be one of ` {outdoor,indoor,object}`.

To preprocess your data, you can choose to either

- Run the following end-to-end script:
    ```bash
    bash projects/neuralangelo/scripts/preprocess.sh ${SEQUENCE} ${PATH_TO_VIDEO} ${DOWNSAMPLE_RATE} ${SCENE_TYPE}
    ```

- Or you can follow the steps below if you want more fine-grained control:

    1. Extract images from the input video

        ```bash
        bash projects/neuralangelo/scripts/run_ffmpeg.sh ${SEQUENCE} ${PATH_TO_VIDEO} ${DOWNSAMPLE_RATE}
        ```
        This will create a directory `datasets/{SEQUENCE}_ds{DOWNSAMPLE_RATE}` (set as `DATA_PATH` onwards), which stores all the processed data.
        The extracted images will be stored in `{DATA_PATH}/images_raw`.

    2. Run COLMAP

        ```bash
        DATA_PATH=datasets/${SEQUENCE}_ds${DOWNSAMPLE_RATE}
        bash projects/neuralangelo/scripts/run_colmap.sh ${DATA_PATH}
        ```
        `DATA_PATH`: path to processed data.

        After COLMAP finishes, the folder structure will look like following:
        ```
        DATA_PATH
        ├─ database.db      (COLMAP database)
        ├─ images           (undistorted input images)
        ├─ images_raw       (raw input images)
        ├─ sparse           (COLMAP data from SfM)
        │  ├─ cameras.bin   (camera parameters)
        │  ├─ images.bin    (images and camera poses)
        │  ├─ points3D.bin  (sparse point clouds)
        │  ├─ 0             (a directory containing individual SfM models. There could also be 1, 2... etc.)
        │  ...
        ├─ stereo (COLMAP data for MVS, not used here)
        ...
        ```
        `{DATA_PATH}/images` will be the input image observations for surface reconstruction.

    3. Generate JSON file for data loading

        In this step, we define the bounding region for reconstruction and convert the COLMAP data to JSON format following Instant NGP.
        It is strongly recommended to [inspect](#inspect-and-adjust-colmap-results) the results to verify and adjust the bounding region for improved performance.
        ```bash
        python3 projects/neuralangelo/scripts/convert_data_to_json.py --data_dir ${DATA_PATH} --scene_type ${SCENE_TYPE}
        ```
        The JSON file will be generated in `{DATA_PATH}/transforms.json`.

    4. Config files

        Use the following to configure and generate your config files:
        ```bash
        python3 projects/neuralangelo/scripts/generate_config.py --sequence_name ${SEQUENCE} --data_dir ${DATA_PATH} --scene_type ${SCENE_TYPE}
        ```
        The config file will be generated as `projects/neuralangelo/configs/custom/{SEQUENCE}.yaml`.
        You can add the `--help` flag to list all arguments; for example, consider adding `--auto_exposure_wb` for modeling varying lighting/appearances in the video.
        Alternatively, you can directly modify the hyperparameters in the generated config file.

### Inspect and adjust COLMAP results

For certain cases, the camera poses estimated by COLMAP could be erroneous. In addition, the automated estimation of the bounding sphere could be inaccurate (which ideally should include the scene/object of interest). It is highly recommended that the bounding sphere is adjusted. 
We offer some tools to to inspect and adjust the pre-processing results. Below are some options:

- Blender: Download [Blender](https://www.blender.org/download/) and follow the instructions in our [add-on repo](https://github.com/mli0603/BlenderNeuralangelo). The add-on will save your adjustment of the bounding sphere.
- This [Jupyter notebook](projects/neuralangelo/scripts/visualize_colmap.ipynb) (using K3D) can be helpful for visualizing the COLMAP results. You can adjust the bounding sphere by manually specifying the refining sphere center and size in the `data.readjust` config.

For certain cases, an exhaustive feature matcher may be able to estimate more accurate camera poses.
This could be done by changing `sequential_matcher` to `exhaustive_matcher` in [run_colmap.sh](https://github.com/NVlabs/neuralangelo/blob/main/projects/neuralangelo/scripts/run_colmap.sh#L24).
However, this would take more time to process and could sometimes result in "broken trajectories" (from COLMAP failing due to ambiguous matches).
For more details, please refer to the COLMAP [documentation](https://colmap.github.io/).

## DTU dataset
You can run the following command to download [the DTU dataset](https://roboimagedata.compute.dtu.dk/?page_id=36) that is preprocessed by NeuS authors and generate json files:
```bash
PATH_TO_DTU=datasets/dtu  # Modify this to be the DTU dataset root directory.
bash projects/neuralangelo/scripts/preprocess_dtu.sh ${PATH_TO_DTU}
```

## Tanks and Temples dataset
Download the data from [Tanks and Temples](https://tanksandtemples.org/download/) website.
You will also need to download additional [COLMAP/camera/alignment](https://drive.google.com/file/d/1jAr3IDvhVmmYeDWi0D_JfgiHcl70rzVE/view?resourcekey=) and the images of each scene.  
The file structure should look like (you need to move the downloaded images to folder `images_raw`):
```
tanks_and_temples
├─ Barn
│  ├─ Barn_COLMAP_SfM.log   (camera poses)
│  ├─ Barn.json             (cropfiles)
│  ├─ Barn.ply              (ground-truth point cloud)
│  ├─ Barn_trans.txt        (colmap-to-ground-truth transformation)
│  └─ images_raw            (raw input images downloaded from Tanks and Temples website)
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
