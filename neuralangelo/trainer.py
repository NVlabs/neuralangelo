import torch
import torch.nn.functional as torch_F
from tqdm import tqdm
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from neuralangelo.model import Model
from neuralangelo.datasets.dataloader import get_dataloaer
import neuralangelo.utils.misc as misc
import neuralangelo.utils.torch_utils as th_utils
from neuralangelo.utils.timer import Timer


class Trainer(object):
    def __init__(self, cfg, phase="train"):
        self.cfg = cfg
        self.phase = phase
        self.device = torch.device("cuda:0")
        self.model = self.setup_model(cfg)
        self.warm_up_end = cfg.optim.sched.warm_up_end
        self.cfg_gradient = cfg.model.object.sdf.gradient
        if (
            cfg.model.object.sdf.encoding.type == "hashgrid"
            and cfg.model.object.sdf.encoding.coarse2fine.enabled
        ):
            self.c2f_step = cfg.model.object.sdf.encoding.coarse2fine.step
            self.model.neural_sdf.warm_up_end = self.warm_up_end

        self.optim = None
        self.sched = None
        if phase:
            self.optim = th_utils.setup_optimizer(cfg, self.model)
            self.sched = misc.get_scheduler(cfg.optim, self.optim)
        self.init_losses(cfg)
        self.cur_iter = 0
        self.writer = SummaryWriter(cfg.ckpt.output)
        self.timer = Timer()

    def setup_model(self, cfg):
        model = Model(cfg.model, cfg.data)
        model = model.to(self.device)
        print(f"model parameter: {misc.calculate_model_size(model)}")
        return model

    def init_losses(self, cfg):
        self.render_loss = torch.nn.L1Loss().to(self.device)
        self.loss_weight = {
            key: value for key, value in cfg.trainer.loss_weight.items() if value
        }

    def compute_loss(self, data, phase="train"):
        losses = {}
        metrics = {}
        if phase == "train":
            loss_weight = self.cfg.trainer.loss_weight
            # Compute loss only on randomly sampled rays.
            # FIXME:sumRGB?!
            losses["render"] = (
                self.render_loss(data["rgb"], data["image_sampled"])
                * 3
                * loss_weight.render
            )
            losses["eikonal"] = (
                misc.eikonal_loss(data["gradients"], outside=data["outside"])
                * loss_weight.eikonal
            )
            if "curvature" in loss_weight:
                losses["curvature"] = (
                    misc.curvature_loss(data["hessians"], outside=data["outside"])
                    * self.loss_weight["curvature"]
                )
            total = 0
            for key, value in losses.items():
                total += value
            losses["total"] = total
            metrics["psnr"] = (
                -10 * torch_F.mse_loss(data["rgb"], data["image_sampled"]).log10()
            )

        else:
            # Compute loss on the entire image.
            losses["render"] = self.render_loss(data["rgb_map"], data["image"])
            metrics["psnr"] = (
                -10 * torch_F.mse_loss(data["rgb_map"], data["image"]).log10()
            )
        return losses, metrics

    def update_loss_weight(self):
        if "curvature" in self.loss_weight:
            init_weight = self.cfg.trainer.loss_weight.curvature
            if self.cur_iter <= self.warm_up_end:
                self.loss_weight["curvature"] = (
                    self.cur_iter / self.warm_up_end * init_weight
                )
            else:
                decay_factor = self.model.neural_sdf.growth_rate ** (
                    self.model.neural_sdf.anneal_levels - 1
                )
                self.loss_weight["curvature"] = init_weight / decay_factor

    def start_of_iteration(self):
        if self.cfg.model.object.sdf.encoding.coarse2fine.enabled:
            self.model.neural_sdf.set_active_levels(self.cur_iter)
            if self.cfg_gradient.mode == "numerical":
                self.model.neural_sdf.set_normal_epsilon()
                self.update_loss_weight()
        elif self.cfg_gradient.mode == "numerical":
            self.model.neural_sdf.set_normal_epsilon()

    def log_scalars(self, data, losses, metrics, phase):
        scalars = {
            "psnr": metrics["psnr"].detach(),
            "s-var": self.model.s_var.item(),
            "render_loss": losses["render"].item(),
        }
        if "curvature" in self.cfg.trainer.loss_weight and phase == "train":
            scalars["curvature_weight"] = self.loss_weight["curvature"]
            scalars["curvature_loss"] = losses["curvature"]
        if phase == "train" and self.cfg_gradient.mode == "numerical":
            scalars["epsilon"] = self.model.neural_sdf.normal_eps
        if phase == "train":
            scalars["eikonal_weight"] = self.loss_weight["eikonal"]
            scalars["eikonal_loss"] = losses["eikonal"].item()
            scalars["total_loss"] = losses["total"].item()
        if self.cfg.model.object.sdf.encoding.coarse2fine.enabled:
            scalars["active_levels"] = self.model.neural_sdf.active_levels

        for key, value in scalars.items():
            self.writer.add_scalar(f"{phase}/{key}", value, global_step=self.cur_iter)

    def log_images(self, data, phase=None):
        assert phase == "val"
        images_error = (data["rgb_map"] - data["image"]).abs()
        depth_image = 1 / (data["depth_map"] + 1e-8) * self.cfg.trainer.depth_vis_scale
        images = {
            "rgb_target": data["image"].squeeze(),
            "rgb_render": data["rgb_map"].squeeze(),
            "rgb_error": images_error.squeeze(),
            "normal": data["normal_map"].squeeze(),
            "inv_depth": depth_image.squeeze(dim=0),
            "opacity": data["opacity_map"].squeeze(dim=0),
        }
        for key, value in images.items():
            self.writer.add_image(
                f"{phase}_img/{key}", value, global_step=self.cur_iter
            )

    def log_tb(self, data, losses, metrics, phase=""):
        if self.cur_iter % 500 == 0 and phase == "train":
            self.log_scalars(data, losses, metrics, phase)

        if phase == "val":
            self.log_scalars(data, losses, metrics, phase)
            self.log_images(data, phase=phase)

    def train_epoch(self, epoch, data_loader):
        self.model.train()
        for iter, data in enumerate(data_loader):
            self.cur_iter = epoch * len(data) + iter + 1
            self.model.progress = self.cur_iter / self.cfg.max_iter
            self.start_of_iteration()
            self.timer.start_timer("data_to_cuda")
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    data[key] = value.to(self.device)
            self.timer.end_timer("data_to_cuda")
            self.timer.start_timer("model_forward")
            output = self.model(data)
            self.timer.end_timer("model_forward")
            self.timer.start_timer("compute_loss")
            data.update(output)
            losses, metrics = self.compute_loss(data, phase="train")
            self.timer.end_timer("compute_loss")
            self.timer.start_timer("backward")
            losses["total"].backward()
            self.timer.end_timer("backward")
            self.timer.start_timer("step")
            self.optim.step()
            self.sched.step()
            self.optim.zero_grad()
            self.timer.end_timer("step")
            self.timer.start_timer("log_tb")
            self.log_tb(data, losses, metrics, phase="train")
            if self.cur_iter % 1000 == 0:
                logger.info(f"{self.cur_iter}/{self.cfg.max_iter}")
            self.timer.end_timer("log_tb")

            if self.cur_iter > self.cfg.max_iter:
                break

    def train(self):
        cfg = self.cfg
        data_loader = get_dataloaer(cfg, phase="train")
        val_loader = get_dataloaer(cfg, phase="val")

        for epoch in range(cfg.max_epoch):
            self.train_epoch(epoch, data_loader)
            if self.cur_iter % 2000 == 0:
                self.test(val_loader)
        self.timer.print()
        self.model.timer.print()

    @torch.no_grad()
    def test(self, data_loader, phase="val"):
        """The evaluation/inference engine.
        Args:
            data_loader: The data loader.
            phase: Evaluation mode {"val", "test"}. Can be other modes, but will only gather the data.
        """
        self.model.eval()
        data_loader = tqdm(data_loader, desc="Evaluating", leave=False)
        data_batches = []
        for it, data in enumerate(data_loader):
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    data[key] = value.to(self.device)
            output = self.model.inference(data)
            data.update(output)
            data_batches.append(data)
        # Aggregate the data from all devices and process the results.
        data_gather = misc.collate_test_data_batches(data_batches)
        # Only the master process should process the results; slaves will just return.

        data_all = misc.get_unique_test_data(data_gather, data_gather["idx"])
        print(f"Evaluating with {len(data_all['idx'])} samples.")

        if phase == "val":
            losses, metrics = self.compute_loss(data_all, phase=phase)
            self.log_tb(data, losses, metrics, phase=phase)
