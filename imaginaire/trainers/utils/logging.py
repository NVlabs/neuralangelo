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

import datetime
import os

import torch.distributed as dist

from imaginaire.utils.distributed import is_master, broadcast_object_list
from imaginaire.utils.distributed import master_only_print as print


def get_date_uid():
    """Generate a unique id based on date.
    Returns:
        str: Return uid string, e.g. '20171122171307111552'.
    """
    return str(datetime.datetime.now().strftime("%Y_%m%d_%H%M_%S"))


def init_logging(config_path, logdir, makedir=True):
    r"""Create log directory for storing checkpoints and output images.

    Args:
        config_path (str): Path to the configuration file.
        logdir (str or None): Log directory name
        makedir (bool): Make a new dir or not
    Returns:
        str: Return log dir
    """
    def _create_logdir(_config_path, _logdir, _root_dir):
        config_file = os.path.basename(_config_path)
        date_uid = get_date_uid()
        # example: logs/2019_0125_1047_58_spade_cocostuff
        _log_file = '_'.join([date_uid, os.path.splitext(config_file)[0]])
        if _logdir is None:
            _logdir = os.path.join(_root_dir, _log_file)
        if makedir:
            print('Make folder {}'.format(_logdir))
            os.makedirs(_logdir, exist_ok=True)
        return _logdir

    root_dir = 'logs'
    if dist.is_available():
        if dist.is_initialized():
            message = [None]
            if is_master():
                logdir = _create_logdir(config_path, logdir, root_dir)
                message = [logdir]

            # Send logdir from master to all workers.
            message = broadcast_object_list(message=message, src=0)
            logdir = message[0]
        else:
            logdir = _create_logdir(config_path, logdir, root_dir)
    else:
        logdir = _create_logdir(config_path, logdir, root_dir)

    return logdir
