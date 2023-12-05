source ./set_env.sh
CONFIG=neuralangelo/configs/dtu_slim.yaml
python train.py \
    --config=${CONFIG}
