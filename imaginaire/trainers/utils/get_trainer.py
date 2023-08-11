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

import importlib
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import lr_scheduler
from imaginaire.models.utils.model_average import ModelAverage


def get_trainer(cfg, is_inference=True, seed=0):
    """Return the trainer object.

    Args:
        cfg (Config): Loaded config object.
        is_inference (bool): Inference mode.

    Returns:
        (obj): Trainer object.
    """
    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg, is_inference=is_inference, seed=seed)
    return trainer


def wrap_model(cfg, model):
    r"""Wrap the networks with AMP DDP and (optionally) model average.

    Args:
        cfg (obj): Global configuration.
        model (obj): Model object.

    Returns:
        (dict):
          - model (obj): Model object.
    """
    # Apply model average wrapper.
    if cfg.trainer.ema_config.enabled:
        model = ModelAverage(model,
                             cfg.trainer.ema_config.beta,
                             cfg.trainer.ema_config.start_iteration,
                             )
    model = _wrap_model(cfg, model)
    return model


class WrappedModel(nn.Module):
    r"""Dummy wrapping the module.
    """

    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        r"""PyTorch module forward function overload."""
        return self.module(*args, **kwargs)


def _wrap_model(cfg, model):
    r"""Wrap a model for distributed data parallel training.

    Args:
        model (obj): PyTorch network model.

    Returns:
        (obj): Wrapped PyTorch network model.
    """
    # Apply DDP wrapper.
    if dist.is_available() and dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            find_unused_parameters=cfg.trainer.ddp_config.find_unused_parameters,
            static_graph=cfg.trainer.ddp_config.static_graph,
            broadcast_buffers=False,
        )
    else:
        model = WrappedModel(model)
    return model


def _calculate_model_size(model):
    r"""Calculate number of parameters in a PyTorch network.

    Args:
        model (obj): PyTorch network.

    Returns:
        (int): Number of parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_optimizer(cfg_optim, model):
    r"""Return the optimizer object.

    Args:
        cfg_optim (obj): Config for the specific optimization module (gen/dis).
        model (obj): PyTorch network object.

    Returns:
        (obj): Pytorch optimizer
    """
    if hasattr(model, 'get_param_groups'):
        # Allow the network to use different hyperparameters (e.g., learning rate) for different parameters.
        params = model.get_param_groups(cfg_optim)
    else:
        params = model.parameters()

    try:
        # Try the PyTorch optimizer class first.
        optimizer_class = getattr(torch.optim, cfg_optim.type)
    except AttributeError:
        raise NotImplementedError(f"Optimizer {cfg_optim.type} is not yet implemented.")
    optimizer_kwargs = cfg_optim.params

    # We will try to use fuse optimizers by default.
    try:
        from apex.optimizers import FusedAdam, FusedSGD
        fused_opt = cfg_optim.fused_opt
    except (ImportError, ModuleNotFoundError):
        fused_opt = False

    if fused_opt:
        if cfg_optim.type == 'Adam':
            optimizer_class = FusedAdam
            optimizer_kwargs['adam_w_mode'] = False
        elif cfg_optim.type == 'AdamW':
            optimizer_class = FusedAdam
            optimizer_kwargs['adam_w_mode'] = True
        elif cfg_optim.type == 'SGD':
            optimizer_class = FusedSGD
    if cfg_optim.type in ["RAdam", "RMSprop"]:
        optimizer_kwargs["foreach"] = fused_opt

    optim = optimizer_class(params, **optimizer_kwargs)

    return optim


def get_scheduler(cfg_optim, optim):
    """Return the scheduler object.

    Args:
        cfg_optim (obj): Config for the specific optimization module (gen/dis).
        optim (obj): PyTorch optimizer object.

    Returns:
        (obj): Scheduler
    """
    if cfg_optim.sched.type == 'step':
        scheduler = lr_scheduler.StepLR(optim,
                                        step_size=cfg_optim.sched.step_size,
                                        gamma=cfg_optim.sched.gamma)
    elif cfg_optim.sched.type == 'constant':
        scheduler = lr_scheduler.LambdaLR(optim, lambda x: 1)
    elif cfg_optim.sched.type == 'linear_warmup':
        scheduler = lr_scheduler.LambdaLR(
            optim, lambda x: x * 1.0 / cfg_optim.sched.warmup if x < cfg_optim.sched.warmup else 1.0)
    elif cfg_optim.sched.type == 'cosine_warmup':

        warmup_scheduler = lr_scheduler.LinearLR(
            optim,
            start_factor=1.0 / cfg_optim.sched.warmup,
            end_factor=1.0,
            total_iters=cfg_optim.sched.warmup
        )
        T_max_val = cfg_optim.sched.decay_steps - cfg_optim.sched.warmup
        cosine_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optim,
            T_max=T_max_val,
            eta_min=getattr(cfg_optim.sched, 'eta_min', 0),
        )
        scheduler = lr_scheduler.SequentialLR(
            optim,
            schedulers=[warmup_scheduler, cosine_lr_scheduler],
            milestones=[cfg_optim.sched.warmup]
        )

    elif cfg_optim.sched.type == 'linear':
        # Start linear decay from here.
        decay_start = cfg_optim.sched.decay_start
        # End linear decay here.
        # Continue to train using the lowest learning rate till the end.
        decay_end = cfg_optim.sched.decay_end
        # Lowest learning rate multiplier.
        decay_target = cfg_optim.sched.decay_target

        def sch(x):
            decay = ((x - decay_start) * decay_target + decay_end - x) / (decay_end - decay_start)
            return min(max(decay, decay_target), 1.)

        scheduler = lr_scheduler.LambdaLR(optim, lambda x: sch(x))
    elif cfg_optim.sched.type == 'step_with_warmup':
        # The step_size and gamma follows the signature of lr_scheduler.StepLR.
        step_size = cfg_optim.sched.step_size,
        gamma = cfg_optim.sched.gamma
        # An additional parameter defines the warmup iteration.
        warmup_step_size = cfg_optim.sched.warmup_step_size

        def sch(x):
            lr_after_warmup = gamma ** (warmup_step_size // step_size)
            if x < warmup_step_size:
                return x / warmup_step_size * lr_after_warmup
            else:
                return gamma ** (x // step_size)

        scheduler = lr_scheduler.LambdaLR(optim, lambda x: sch(x))
    else:
        return NotImplementedError('Learning rate policy {} not implemented.'.format(cfg_optim.sched.type))
    return scheduler
