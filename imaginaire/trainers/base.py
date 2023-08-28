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
import json
import os
import threading
import time
import wandb
from tqdm import tqdm
import inspect

import torch
from torch.autograd import profiler
from torch.cuda.amp import GradScaler, autocast

from imaginaire.datasets.utils.get_dataloader import get_train_dataloader, get_val_dataloader, get_test_dataloader
from imaginaire.models.utils.init_weight import weights_init, weights_rescale
from imaginaire.trainers.utils.get_trainer import _calculate_model_size, get_optimizer, get_scheduler, wrap_model

from imaginaire.utils.misc import to_cuda, requires_grad, to_cpu, Timer
from imaginaire.utils.distributed import master_only_print as print
from imaginaire.utils.distributed import is_master, get_rank
from imaginaire.utils.set_random_seed import set_random_seed


class BaseTrainer(object):
    r"""Base trainer. We expect that all trainers inherit this class.

    Args:
        cfg (obj): Global configuration.
        is_inference (bool): if True, load the test dataloader and run in inference mode.
    """

    def __init__(self, cfg, is_inference=True, seed=0):
        super().__init__()
        print('Setup trainer.')
        self.cfg = cfg
        torch.cuda.set_device(cfg.local_rank)
        # Create objects for the networks, optimizers, and schedulers.
        self.model = self.setup_model(cfg, seed=seed)
        if not is_inference:
            self.optim = self.setup_optimizer(cfg, self.model, seed=seed)
            self.sched = self.setup_scheduler(cfg, self.optim)
        else:
            self.optim = None
            self.sched = None
        self.model = self.wrap_model(cfg, self.model)
        # Data loaders & inference mode.
        self.is_inference = is_inference
        # Initialize automatic mixed precision training.
        self.init_amp()
        # Initialize loss functions.
        self.init_losses(cfg)

        self.checkpointer = Checkpointer(cfg, self.model, self.optim, self.sched)
        self.timer = Timer(cfg)

        # -------- The initialization steps below can be skipped during inference. --------
        if self.is_inference:
            return

        # Initialize logging attributes.
        self.init_logging_attributes()
        # Initialize validation parameters.
        self.init_val_parameters()
        # AWS credentials.
        if hasattr(cfg, 'aws_credentials_file'):
            with open(cfg.aws_credentials_file) as fin:
                self.credentials = json.load(fin)
        else:
            self.credentials = None
        if 'TORCH_HOME' not in os.environ:
            os.environ['TORCH_HOME'] = os.path.join(os.environ['HOME'], ".cache")

    def set_data_loader(self, cfg, split, shuffle=True, drop_last=True, seed=0):
        """Set the data loader corresponding to the indicated split.
        Args:
            split (str): Must be either 'train', 'val', or 'test'.
            shuffle (bool): Whether to shuffle the data (only applies to the training set).
            drop_last (bool): Whether to drop the last batch if it is not full (only applies to the training set).
            seed (int): Random seed.
        """
        assert (split in ["train", "val", "test"])
        if split == "train":
            self.train_data_loader = get_train_dataloader(cfg, shuffle=shuffle, drop_last=drop_last, seed=seed)
        elif split == "val":
            self.eval_data_loader = get_val_dataloader(cfg, seed=seed)
        elif split == "test":
            self.eval_data_loader = get_test_dataloader(cfg)

    def setup_model(self, cfg, seed=0):
        r"""Return the networks. We will first set the random seed to a fixed value so that each GPU copy will be
        initialized to have the same network weights. We will then use different random seeds for different GPUs.
        After this we will wrap the network with a moving average model if applicable.

        The following objects are constructed as class members:
          - model (obj): Model object (historically: generator network object).

        Args:
            cfg (obj): Global configuration.
            seed (int): Random seed.
        """
        # We first set the random seed for all the process so that we initialize each copy of the network the same.
        set_random_seed(seed, by_rank=False)
        # Construct networks
        lib_model = importlib.import_module(cfg.model.type)
        model = lib_model.Model(cfg.model, cfg.data)
        print('model parameter count: {:,}'.format(_calculate_model_size(model)))
        print(f'Initialize model weights using type: {cfg.trainer.init.type}, gain: {cfg.trainer.init.gain}')
        init_bias = getattr(cfg.trainer.init, 'bias', None)
        init_gain = cfg.trainer.init.gain or 1.
        model.apply(weights_init(cfg.trainer.init.type, init_gain, init_bias))
        model.apply(weights_rescale())
        model = model.to('cuda')
        # Different GPU copies of the same model will receive noises initialized with different random seeds
        # (if applicable) thanks to the set_random_seed command (GPU #K has random seed = args.seed + K).
        set_random_seed(seed, by_rank=True)
        return model

    def setup_optimizer(self, cfg, model, seed=0):
        r"""Return the optimizers.

        The following objects are constructed as class members:
          - optim (obj): Model optimizer object.

        Args:
            cfg (obj): Global configuration.
            seed (int): Random seed.
        """
        optim = get_optimizer(cfg.optim, model)
        self.optim_zero_grad_kwargs = {}
        if 'set_to_none' in inspect.signature(optim.zero_grad).parameters:
            self.optim_zero_grad_kwargs['set_to_none'] = True
        return optim

    def setup_scheduler(self, cfg, optim):
        r"""Return the schedulers.

        The following objects are constructed as class members:
          - sched (obj): Model optimizer scheduler object.

        Args:
            cfg (obj): Global configuration.
        """
        return get_scheduler(cfg.optim, optim)

    def wrap_model(self, cfg, model):
        # Moving average model & data distributed data parallel wrapping.
        model = wrap_model(cfg, model)
        # Get actual modules from wrappers.
        if cfg.trainer.ema_config.enabled:
            # Two wrappers (DDP + model average).
            self.model_module = model.module.module
        else:
            # One wrapper (DDP)
            self.model_module = model.module
        return model

    def init_amp(self):
        r"""Initialize automatic mixed precision training."""

        if getattr(self.cfg.trainer, 'allow_tf32', True):
            print("Allow TensorFloat32 operations on supported devices")
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
        else:
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False

        if self.cfg.trainer.amp_config.enabled:
            print("Using automatic mixed precision training.")

        # amp scaler can be used without mixed precision training
        if hasattr(self.cfg.trainer, 'scaler_config'):
            scaler_kwargs = vars(self.cfg.trainer.scaler_config)
            scaler_kwargs['enabled'] = self.cfg.trainer.amp_config.enabled and \
                getattr(self.cfg.trainer.scaler_config, 'enabled', True)
        else:
            scaler_kwargs = vars(self.cfg.trainer.amp_config)   # backward compatibility
            scaler_kwargs.pop('dtype', None)
            scaler_kwargs.pop('cache_enabled', None)

        self.scaler = GradScaler(**scaler_kwargs)

    def init_losses(self, cfg):
        r"""Initialize loss functions. All loss names have weights. Some have criterion modules."""
        self.losses = dict()

        # Mapping from loss names to criterion modules.
        self.criteria = torch.nn.ModuleDict()
        # Mapping from loss names to loss weights.
        self.weights = dict()

        self._init_loss(cfg)  # this should be implemented by children classes

        for loss_name, loss_weight in self.weights.items():
            print("Loss {:<20} Weight {}".format(loss_name, loss_weight))
            if loss_name in self.criteria.keys() and self.criteria[loss_name] is not None:
                self.criteria[loss_name].to('cuda')

    def init_logging_attributes(self):
        r"""Initialize logging attributes."""
        self.current_iteration = 0
        self.current_epoch = 0
        self.start_iteration_time = None
        self.start_epoch_time = None
        self.elapsed_iteration_time = 0
        if self.cfg.speed_benchmark:
            self.timer.reset()

    def init_val_parameters(self):
        r"""Initialize validation parameters."""
        if self.cfg.metrics_iter is None:
            self.cfg.metrics_iter = self.cfg.checkpoint.save_iter
        if self.cfg.metrics_epoch is None:
            self.cfg.metrics_epoch = self.cfg.checkpoint.save_epoch

    def init_wandb(self, cfg, wandb_id=None, project="", run_name=None, mode="online", resume="allow", use_group=False):
        r"""Initialize Weights & Biases (wandb) logger.

        Args:
            cfg (obj): Global configuration.
            wandb_id (str): A unique ID for this run, used for resuming.
            project (str): The name of the project where you're sending the new run.
                If the project is not specified, the run is put in an "Uncategorized" project.
            run_name (str): name for each wandb run (useful for logging changes)
            mode (str): online/offline/disabled
        """
        if is_master():
            print('Initialize wandb')
            if not wandb_id:
                wandb_path = os.path.join(cfg.logdir, "wandb_id.txt")
                if self.checkpointer.resume and os.path.exists(wandb_path):
                    with open(wandb_path, "r") as f:
                        wandb_id = f.read()
                else:
                    wandb_id = wandb.util.generate_id()
                    with open(wandb_path, "w") as f:
                        f.write(wandb_id)
            if use_group:
                group, name = cfg.logdir.split("/")[-2:]
            else:
                group, name = None, os.path.basename(cfg.logdir)

            if run_name is not None:
                name = run_name

            wandb.init(id=wandb_id,
                       project=project,
                       config=cfg,
                       group=group,
                       name=name,
                       dir=cfg.logdir,
                       resume=resume,
                       settings=wandb.Settings(start_method="fork"),
                       mode=mode)
            wandb.config.update({'dataset': cfg.data.name})
            if self.model_module is not None:
                wandb.watch(self.model_module)

    def start_of_epoch(self, current_epoch):
        r"""Things to do before an epoch.

        Args:
            current_epoch (int): Current number of epoch.
        """
        self._start_of_epoch(current_epoch)
        self.current_epoch = current_epoch
        self.start_epoch_time = time.time()

    def start_of_iteration(self, data, current_iteration):
        r"""Things to do before an iteration.

        Args:
            data (dict): Data used for the current iteration.
            current_iteration (int): Current number of iteration.
        """
        data = self._start_of_iteration(data, current_iteration)
        data = to_cuda(data)
        self.current_iteration = current_iteration
        self.model.train()
        self.start_iteration_time = time.time()
        return data

    def end_of_iteration(self, data, current_epoch, current_iteration):
        r"""Things to do after an iteration.

        Args:
            data (dict): Data used for the current iteration.
            current_epoch (int): Current number of epoch.
            current_iteration (int): Current number of iteration.
        """
        self.current_iteration = current_iteration
        self.current_epoch = current_epoch

        # Accumulate time
        self.elapsed_iteration_time += time.time() - self.start_iteration_time
        # Logging.
        if current_iteration % self.cfg.logging_iter == 0:
            avg_time = self.elapsed_iteration_time / self.cfg.logging_iter
            self.timer.time_iteration = avg_time
            print('Iteration: {}, average iter time: {:6f}.'.format(current_iteration, avg_time))
            self.elapsed_iteration_time = 0

            if self.cfg.speed_benchmark:
                # only needed when analyzing computation bottleneck.
                self.timer._print_speed_benchmark(avg_time)

        self._end_of_iteration(data, current_epoch, current_iteration)

        # Save everything to the checkpoint by time period.
        if self.checkpointer.reached_checkpointing_period(self.timer):
            self.checkpointer.save(current_epoch, current_iteration)
            self.timer.checkpoint_tic()  # reset timer

        # Save everything to the checkpoint.
        if current_iteration % self.cfg.checkpoint.save_iter == 0 or \
                current_iteration == self.cfg.max_iter:
            self.checkpointer.save(current_epoch, current_iteration)

        # Save everything to the checkpoint using the name 'latest_checkpoint.pt'.
        if current_iteration % self.cfg.checkpoint.save_latest_iter == 0:
            if current_iteration >= self.cfg.checkpoint.save_latest_iter:
                self.checkpointer.save(current_epoch, current_iteration, True)

        # Update the learning rate policy for the generator if operating in the iteration mode.
        if self.cfg.optim.sched.iteration_mode:
            self.sched.step()

        # This iteration was successfully finished. Reset timeout counter.
        self.timer.reset_timeout_counter()

    def end_of_epoch(self, data, current_epoch, current_iteration):
        r"""Things to do after an epoch.

        Args:
            data (dict): Data used for the current iteration.

            current_epoch (int): Current number of epoch.
            current_iteration (int): Current number of iteration.
        """
        # Update the learning rate policy for the generator if operating in the epoch mode.
        self.current_iteration = current_iteration
        self.current_epoch = current_epoch
        if not self.cfg.optim.sched.iteration_mode:
            self.sched.step()
        elapsed_epoch_time = time.time() - self.start_epoch_time
        # Logging.
        print('Epoch: {}, total time: {:6f}.'.format(current_epoch, elapsed_epoch_time))
        self.timer.time_epoch = elapsed_epoch_time
        self._end_of_epoch(data, current_epoch, current_iteration)

        # Save everything to the checkpoint.
        if current_epoch % self.cfg.checkpoint.save_epoch == 0:
            self.checkpointer.save(current_epoch, current_iteration)

    def _extra_step(self, data):
        pass

    def _start_of_epoch(self, current_epoch):
        r"""Operations to do before starting an epoch.

        Args:
            current_epoch (int): Current number of epoch.
        """
        pass

    def _start_of_iteration(self, data, current_iteration):
        r"""Operations to do before starting an iteration.

        Args:
            data (dict): Data used for the current iteration.
            current_iteration (int): Current epoch number.
        Returns:
            (dict): Data used for the current iteration. They might be
                processed by the custom _start_of_iteration function.
        """
        return data

    def _end_of_iteration(self, data, current_epoch, current_iteration):
        r"""Operations to do after an iteration.

        Args:
            data (dict): Data used for the current iteration.
            current_epoch (int): Current number of epoch.
            current_iteration (int): Current epoch number.
        """
        pass

    def _end_of_epoch(self, data, current_epoch, current_iteration):
        r"""Operations to do after an epoch.

        Args:
            data (dict): Data used for the current iteration.
            current_epoch (int): Current number of epoch.
            current_iteration (int): Current epoch number.
        """
        pass

    def _get_visualizations(self, data):
        r"""Compute visualization outputs.

        Args:
            data (dict): Data used for the current iteration.
        """
        return None

    def _init_loss(self, cfg):
        r"""Every trainer should implement its own init loss function."""
        raise NotImplementedError

    def train_step(self, data, last_iter_in_epoch=False):
        r"""One training step.

        Args:
            data (dict): Data used for the current iteration.
        """
        # Set requires_grad flags.
        requires_grad(self.model_module, True)

        # Compute the loss.
        self.timer._time_before_forward()

        autocast_dtype = getattr(self.cfg.trainer.amp_config, 'dtype', 'float16')
        autocast_dtype = torch.bfloat16 if autocast_dtype == 'bfloat16' else torch.float16
        amp_kwargs = {
            'enabled': self.cfg.trainer.amp_config.enabled,
            'dtype': autocast_dtype
        }
        with autocast(**amp_kwargs):
            total_loss = self.model_forward(data)
            # Scale down the loss w.r.t. gradient accumulation iterations.
            total_loss = total_loss / float(self.cfg.trainer.grad_accum_iter)

        # Backpropagate the loss.
        self.timer._time_before_backward()
        self.scaler.scale(total_loss).backward()

        self._extra_step(data)

        # Perform an optimizer step. This enables gradient accumulation when grad_accum_iter is not 1.
        if (self.current_iteration + 1) % self.cfg.trainer.grad_accum_iter == 0 or last_iter_in_epoch:
            self.timer._time_before_step()
            self.scaler.step(self.optim)
            self.scaler.update()
            # Zero out the gradients.
            self.optim.zero_grad(**self.optim_zero_grad_kwargs)

        # Update model average.
        self.timer._time_before_model_avg()
        if self.cfg.trainer.ema_config.enabled:
            self.model.module.update_average()

        self._detach_losses()
        self.timer._time_before_leave_gen()

    def model_forward(self, data):
        r"""Every trainer should implement its own model forward."""
        raise NotImplementedError

    def train(self, cfg, data_loader, single_gpu=False, profile=False, show_pbar=False):
        r"""Generic training loop. Main structure in a nutshell:
            for epoch in [start_epoch, end_epoch]:
                for batch in dataset (one epoch):
                    train_step(batch)

        Args:
            cfg (obj): Global configuration.
            data_loader (torch.utils.data.DataLoader): PyTorch dataloader.
            single_gpu (bool): Use only a single GPU.
            profile (bool): Enable profiling.
            show_pbar (bool): Whether to show the progress bar
        """
        start_epoch = self.checkpointer.resume_epoch or self.current_epoch  # The epoch to start with.
        current_iteration = self.checkpointer.resume_iteration or self.current_iteration  # The starting iteration.

        self.timer.checkpoint_tic()  # start timer
        self.timer.reset_timeout_counter()
        for current_epoch in range(start_epoch, cfg.max_epoch):
            if not single_gpu:
                data_loader.sampler.set_epoch(current_epoch)
            self.start_of_epoch(current_epoch)
            if show_pbar:
                data_loader_wrapper = tqdm(data_loader, desc=f"Training epoch {current_epoch + 1}", leave=False)
            else:
                data_loader_wrapper = data_loader
            for it, data in enumerate(data_loader_wrapper):
                with profiler.profile(enabled=profile,
                                      use_cuda=True,
                                      profile_memory=True,
                                      record_shapes=True) as prof:
                    data = self.start_of_iteration(data, current_iteration)

                    self.train_step(data, last_iter_in_epoch=(it == len(data_loader) - 1))

                    current_iteration += 1
                    if show_pbar:
                        data_loader_wrapper.set_postfix(iter=current_iteration)
                    if it == len(data_loader) - 1:
                        self.end_of_iteration(data, current_epoch + 1, current_iteration)
                    else:
                        self.end_of_iteration(data, current_epoch, current_iteration)
                    if current_iteration >= cfg.max_iter:
                        print('Done with training!!!')
                        return
                if profile:
                    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
                    prof.export_chrome_trace(os.path.join(cfg.logdir, "trace.json"))

            self.end_of_epoch(data, current_epoch + 1, current_iteration)
        print('Done with training!!!')

    def test(self, data_loader, output_dir, inference_args, show_pbar=False):
        r"""Compute results images and save the results in the specified folder.
        Args:
            data_loader (torch.utils.data.DataLoader): PyTorch dataloader.
            output_dir (str): Target location for saving the output image.
        """
        pass

    def _get_total_loss(self):
        r"""Return the total loss to be backpropagated.
        """
        total_loss = torch.tensor(0., device=torch.device('cuda'))
        # Iterates over all possible losses.
        for loss_name in self.weights:
            if loss_name in self.losses:
                # Multiply it with the corresponding weight and add it to the total loss.
                total_loss += self.losses[loss_name] * self.weights[loss_name]
        self.losses['total'] = total_loss  # logging purpose
        return total_loss

    def _detach_losses(self):
        r"""Detach all logging variables to prevent potential memory leak."""
        for loss_name in self.losses:
            self.losses[loss_name] = self.losses[loss_name].detach()

    def finalize(self, cfg):
        # Finish the W&B logger.
        if is_master():
            wandb.finish()


class Checkpointer(object):

    def __init__(self, cfg, model, optim=None, sched=None):
        self.model = model
        self.optim = optim
        self.sched = sched
        self.logdir = cfg.logdir
        self.save_period = cfg.checkpoint.save_period
        self.strict_resume = cfg.checkpoint.strict_resume
        self.iteration_mode = cfg.optim.sched.iteration_mode
        self.resume = False
        self.resume_epoch = self.resume_iteration = None

    def save(self, current_epoch, current_iteration, latest=False):
        r"""Save network weights, optimizer parameters, scheduler parameters to a checkpoint.

        Args:
            current_epoch (int): Current epoch.
            current_iteration (int): Current iteration.
            latest (bool): If ``True``, save it using the name 'latest_checkpoint.pt'.
        """
        checkpoint_file = 'latest_checkpoint.pt' if latest else \
                          f'epoch_{current_epoch:05}_iteration_{current_iteration:09}_checkpoint.pt'
        if is_master():
            save_dict = to_cpu(self._collect_state_dicts())
            save_dict.update(
                epoch=current_epoch,
                iteration=current_iteration,
            )
            # Run the checkpoint saver in a separate thread.
            threading.Thread(
                target=self._save_worker, daemon=False, args=(save_dict, checkpoint_file, get_rank())).start()
        checkpoint_path = self._get_full_path(checkpoint_file)
        return checkpoint_path

    def _save_worker(self, save_dict, checkpoint_file, rank=0):
        checkpoint_path = self._get_full_path(checkpoint_file)
        # Save to local disk.
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(save_dict, checkpoint_path)
        if rank == 0:
            self.write_latest_checkpoint_file(checkpoint_file)
        print('Saved checkpoint to {}'.format(checkpoint_path))

    def _collect_state_dicts(self):
        r"""Collect all the state dicts from network modules to be saved."""
        return dict(
            model=self.model.state_dict(),
            optim=self.optim.state_dict(),
            sched=self.sched.state_dict(),
        )

    def load(self, checkpoint_path=None, resume=False, load_opt=True, load_sch=True, **kwargs):
        r"""Load network weights, optimizer parameters, scheduler parameters from a checkpoint.
        Args:
            checkpoint_path (str): Path to the checkpoint (local file or S3 key).
            resume (bool): if False, only the model weights are loaded. If True, the metadata (epoch/iteration) and
                           optimizer/scheduler (optional) are also loaded.
            load_opt (bool): Whether to load the optimizer state dict (resume should be True).
            load_sch (bool): Whether to load the scheduler state dict (resume should be True).
        """
        # Priority: (1) checkpoint_path (2) latest_path (3) train from scratch.
        self.resume = resume
        # If checkpoint path were not specified, try to load the latest one from the same run.
        if resume and checkpoint_path is None:
            latest_checkpoint_file = self.read_latest_checkpoint_file()
            if latest_checkpoint_file is not None:
                checkpoint_path = self._get_full_path(latest_checkpoint_file)
        # Load checkpoint.
        if checkpoint_path is not None:
            self._check_checkpoint_exists(checkpoint_path)
            self.checkpoint_path = checkpoint_path
            state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            print(f"Loading checkpoint (local): {checkpoint_path}")
            # Load the state dicts.
            print('- Loading the model...')
            self.model.load_state_dict(state_dict['model'], strict=self.strict_resume)
            if resume:
                self.resume_epoch = state_dict['epoch']
                self.resume_iteration = state_dict['iteration']
                self.sched.last_epoch = self.resume_iteration if self.iteration_mode else self.resume_epoch
                if load_opt:
                    print('- Loading the optimizer...')
                    self.optim.load_state_dict(state_dict['optim'])
                if load_sch:
                    print('- Loading the scheduler...')
                    self.sched.load_state_dict(state_dict['sched'])
                print(f"Done with loading the checkpoint (epoch {self.resume_epoch}, iter {self.resume_iteration}).")
            else:
                print('Done with loading the checkpoint.')
            self.eval_epoch = state_dict['epoch']
            self.eval_iteration = state_dict['iteration']
        else:
            # Checkpoint not found and not specified. We will train everything from scratch.
            print('Training from scratch.')
        torch.cuda.empty_cache()

    def _get_full_path(self, file):
        return os.path.join(self.logdir, file)

    def _get_latest_pointer_path(self):
        return self._get_full_path('latest_checkpoint.txt')

    def read_latest_checkpoint_file(self):
        checkpoint_file = None
        latest_path = self._get_latest_pointer_path()
        if os.path.exists(latest_path):
            checkpoint_file = open(latest_path).read().strip()
            if checkpoint_file.startswith("latest_checkpoint:"):  # TODO: for backward compatibility, to be removed
                checkpoint_file = checkpoint_file.split(' ')[-1]
        return checkpoint_file

    def write_latest_checkpoint_file(self, checkpoint_file):
        latest_path = self._get_latest_pointer_path()
        content = f"{checkpoint_file}\n"
        with open(latest_path, "w") as file:
            file.write(content)

    def _check_checkpoint_exists(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'File not found (local): {checkpoint_path}')

    def reached_checkpointing_period(self, timer):
        save_now = torch.cuda.BoolTensor([False])
        if is_master():
            if timer.checkpoint_toc() > self.save_period:
                save_now.fill_(True)
        if save_now:
            if is_master():
                print('checkpointing period!')
        return save_now
