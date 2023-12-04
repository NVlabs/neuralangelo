
source ./set_env.sh
EXPERIMENT=toy_example
GROUP=dtu
NAME=dtu_scan24
CONFIG=projects/neuralangelo/configs/dtu_slim.yaml
GPUS=1  # use >1 for multi-GPU training!
python train.py \
    --logdir=logs/${GROUP}/${NAME} \
    --config=${CONFIG} \
    --show_pbar
