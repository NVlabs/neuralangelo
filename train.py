import os
import argparse
import time
from projects.neuralangelo.utils.config import Config
from projects.neuralangelo.utils.torch_utils import init_cudnn
from projects.neuralangelo.trainer import Trainer
import projects.neuralangelo.utils.torch_utils as th_utils


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--config", help="Path to the training config file.", required=True
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--wandb_name", default="default", type=str)
    args, cfg_cmd = parser.parse_known_args()
    return args, cfg_cmd


def main():
    args, cfg_cmd = parse_args()
    cfg = Config(args.config)
    th_utils.set_random_seed(args.seed)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    cfg.ckpt.output = f"{cfg.ckpt.output}/{timestamp}_neuralangleo"
    os.makedirs(cfg.ckpt.output)
    # Initialize cudnn.
    init_cudnn(cfg.cudnn.deterministic, cfg.cudnn.benchmark)

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
