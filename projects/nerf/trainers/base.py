'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import torch
import wandb
from imaginaire.trainers.base import BaseTrainer
from imaginaire.utils.distributed import is_master, master_only
from tqdm import tqdm

from projects.nerf.utils.misc import collate_test_data_batches, get_unique_test_data, trim_test_samples


class BaseTrainer(BaseTrainer):
    """
    A customized BaseTrainer.
    """

    def __init__(self, cfg, is_inference=True, seed=0):
        super().__init__(cfg, is_inference=is_inference, seed=seed)
        self.metrics = dict()
        # The below configs should be properly overridden.
        cfg.setdefault("wandb_scalar_iter", 9999999999999)
        cfg.setdefault("wandb_image_iter", 9999999999999)
        cfg.setdefault("validation_epoch", 9999999999999)
        cfg.setdefault("validation_iter", 9999999999999)

    def init_losses(self, cfg):
        super().init_losses(cfg)
        self.weights = {key: value for key, value in cfg.trainer.loss_weight.items() if value}

    def _end_of_iteration(self, data, current_epoch, current_iteration):
        # Log to wandb.
        if current_iteration % self.cfg.wandb_scalar_iter == 0:
            # Compute the elapsed time (as in the original base trainer).
            self.timer.time_iteration = self.elapsed_iteration_time / self.cfg.wandb_scalar_iter
            self.elapsed_iteration_time = 0
            # Log scalars.
            self.log_wandb_scalars(data, mode="train")
            # Exit if the training loss has gone to NaN/inf.
            if is_master() and self.losses["total"].isnan():
                self.finalize(self.cfg)
                raise ValueError("Training loss has gone to NaN!!!")
            if is_master() and self.losses["total"].isinf():
                self.finalize(self.cfg)
                raise ValueError("Training loss has gone to infinity!!!")
        if current_iteration % self.cfg.wandb_image_iter == 0:
            self.log_wandb_images(data, mode="train")
        # Run validation on val set.
        if current_iteration % self.cfg.validation_iter == 0:
            data_all = self.test(self.eval_data_loader, mode="val")
            # Log the results to W&B.
            if is_master():
                self.log_wandb_scalars(data_all, mode="val")
                self.log_wandb_images(data_all, mode="val", max_samples=self.cfg.data.val.max_viz_samples)

    def _end_of_epoch(self, data, current_epoch, current_iteration):
        # Run validation on val set.
        if current_epoch % self.cfg.validation_epoch == 0:
            data_all = self.test(self.eval_data_loader, mode="val")
            # Log the results to W&B.
            if is_master():
                self.log_wandb_scalars(data_all, mode="val")
                self.log_wandb_images(data_all, mode="val", max_samples=self.cfg.data.val.max_viz_samples)

    @master_only
    def log_wandb_scalars(self, data, mode=None):
        scalars = dict()
        # Log scalars (basic info & losses).
        if mode == "train":
            scalars.update({"optim/lr": self.sched.get_last_lr()[0]})
            scalars.update({"time/iteration": self.timer.time_iteration})
            scalars.update({"time/epoch": self.timer.time_epoch})
        scalars.update({f"{mode}/loss/{key}": value for key, value in self.losses.items()})
        scalars.update(iteration=self.current_iteration, epoch=self.current_epoch)
        wandb.log(scalars, step=self.current_iteration)

    @master_only
    def log_wandb_images(self, data, mode=None, max_samples=None):
        trim_test_samples(data, max_samples=max_samples)

    def model_forward(self, data):
        # Model forward.
        output = self.model(data)  # data = self.model(data) will not return the same data in the case of DDP.
        data.update(output)
        # Compute loss.
        self.timer._time_before_loss()
        self._compute_loss(data, mode="train")
        total_loss = self._get_total_loss()
        return total_loss

    def _compute_loss(self, data, mode=None):
        raise NotImplementedError

    def train(self, cfg, data_loader, single_gpu=False, profile=False, show_pbar=False):
        self.current_epoch = self.checkpointer.resume_epoch or self.current_epoch
        self.current_iteration = self.checkpointer.resume_iteration or self.current_iteration
        if ((self.current_epoch % self.cfg.validation_epoch == 0 or
             self.current_iteration % self.cfg.validation_iter == 0)):
            # Do an initial validation.
            data_all = self.test(self.eval_data_loader, mode="val", show_pbar=show_pbar)
            # Log the results to W&B.
            if is_master():
                self.log_wandb_scalars(data_all, mode="val")
                self.log_wandb_images(data_all, mode="val", max_samples=self.cfg.data.val.max_viz_samples)
        # Train.
        super().train(cfg, data_loader, single_gpu, profile, show_pbar)

    @torch.no_grad()
    def test(self, data_loader, output_dir=None, inference_args=None, mode="test", show_pbar=False):
        """The evaluation/inference engine.
        Args:
            data_loader: The data loader.
            output_dir: Output directory to dump the test results.
            inference_args: (unused)
            mode: Evaluation mode {"val", "test"}. Can be other modes, but will only gather the data.
        Returns:
            data_all: A dictionary of all the data.
        """
        if self.cfg.trainer.ema_config.enabled:
            model = self.model.module.averaged_model
        else:
            model = self.model.module
        model.eval()
        if show_pbar:
            data_loader = tqdm(data_loader, desc="Evaluating", leave=False)
        data_batches = []
        for it, data in enumerate(data_loader):
            data = self.start_of_iteration(data, current_iteration=self.current_iteration)
            output = model.inference(data)
            data.update(output)
            data_batches.append(data)
        # Aggregate the data from all devices and process the results.
        data_gather = collate_test_data_batches(data_batches)
        # Only the master process should process the results; slaves will just return.
        if is_master():
            data_all = get_unique_test_data(data_gather, data_gather["idx"])
            tqdm.write(f"Evaluating with {len(data_all['idx'])} samples.")
            # Validate/test.
            if mode == "val":
                self._compute_loss(data_all, mode=mode)
                _ = self._get_total_loss()
            if mode == "test":
                # Dump the test results for postprocessing.
                self.dump_test_results(data_all, output_dir)
            return data_all
        else:
            return

    def dump_test_results(self, data_all, output_dir):
        raise NotImplementedError
