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

import collections
import functools
import os
import re

import yaml
from imaginaire.utils.distributed import master_only_print as print
from imaginaire.utils.termcolor import cyan, green, yellow

DEBUG = False
USE_JIT = False


class AttrDict(dict):
    """Dict as attribute trick."""

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                self.__dict__[key] = AttrDict(value)
            elif isinstance(value, (list, tuple)):
                if value and isinstance(value[0], dict):
                    self.__dict__[key] = [AttrDict(item) for item in value]
                else:
                    self.__dict__[key] = value

    def yaml(self):
        """Convert object to yaml dict and return."""
        yaml_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, AttrDict):
                yaml_dict[key] = value.yaml()
            elif isinstance(value, list):
                if value and isinstance(value[0], AttrDict):
                    new_l = []
                    for item in value:
                        new_l.append(item.yaml())
                    yaml_dict[key] = new_l
                else:
                    yaml_dict[key] = value
            else:
                yaml_dict[key] = value
        return yaml_dict

    def __repr__(self):
        """Print all variables."""
        ret_str = []
        for key, value in self.__dict__.items():
            if isinstance(value, AttrDict):
                ret_str.append('{}:'.format(key))
                child_ret_str = value.__repr__().split('\n')
                for item in child_ret_str:
                    ret_str.append('    ' + item)
            elif isinstance(value, list):
                if value and isinstance(value[0], AttrDict):
                    ret_str.append('{}:'.format(key))
                    for item in value:
                        # Treat as AttrDict above.
                        child_ret_str = item.__repr__().split('\n')
                        for item in child_ret_str:
                            ret_str.append('    ' + item)
                else:
                    ret_str.append('{}: {}'.format(key, value))
            else:
                ret_str.append('{}: {}'.format(key, value))
        return '\n'.join(ret_str)


class Config(AttrDict):
    r"""Configuration class. This should include every human specifiable
    hyperparameter values for your training."""

    def __init__(self, filename=None, verbose=False):
        super(Config, self).__init__()
        self.source_filename = filename

        # Load the base configuration file.
        base_filename = os.path.join(
            os.path.dirname(__file__), '../imaginaire/config_base.yaml'
        )
        cfg_base = self.load_config(base_filename)
        recursive_update(self, cfg_base)

        # Update with given configurations.
        cfg_dict = self.load_config(filename)
        recursive_update(self, cfg_dict)

        if verbose:
            print(' imaginaire config '.center(80, '-'))
            print(self.__repr__())
            print(''.center(80, '-'))

    def load_config(self, filename):
        # Update with given configurations.
        assert os.path.exists(filename), f'File {filename} not exist.'
        yaml_loader = yaml.SafeLoader
        yaml_loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        try:
            with open(filename) as file:
                cfg_dict = yaml.load(file, Loader=yaml_loader)
                cfg_dict = AttrDict(cfg_dict)
        except EnvironmentError:
            print(f'Please check the file with name of "{filename}"')
        # Inherit configurations from parent
        parent_key = "_parent_"
        if parent_key in cfg_dict:
            parent_filename = cfg_dict.pop(parent_key)
            cfg_parent = self.load_config(parent_filename)
            recursive_update(cfg_parent, cfg_dict)
            cfg_dict = cfg_parent
        return cfg_dict

    def print_config(self, level=0):
        """Recursively print the configuration (with termcolor)."""
        for key, value in sorted(self.items()):
            if isinstance(value, (dict, Config)):
                print("   " * level + cyan("* ") + green(key) + ":")
                Config.print_config(value, level + 1)
            else:
                print("   " * level + cyan("* ") + green(key) + ":", yellow(value))

    def save_config(self, logdir):
        """Save the final configuration to a yaml file."""
        cfg_fname = f"{logdir}/config.yaml"
        with open(cfg_fname, "w") as file:
            yaml.safe_dump(self.yaml(), file, default_flow_style=False, indent=4)


def rsetattr(obj, attr, val):
    """Recursively find object and set value"""
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    """Recursively find object and return value"""

    def _getattr(obj, attr):
        r"""Get attribute."""
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def recursive_update(d, u):
    """Recursively update AttrDict d with AttrDict u"""
    for key, value in u.items():
        if isinstance(value, collections.abc.Mapping):
            d.__dict__[key] = recursive_update(d.get(key, AttrDict({})), value)
        elif isinstance(value, (list, tuple)):
            if value and isinstance(value[0], dict):
                d.__dict__[key] = [AttrDict(item) for item in value]
            else:
                d.__dict__[key] = value
        else:
            d.__dict__[key] = value
    return d


def recursive_update_strict(d, u, stack=[]):
    """Recursively update AttrDict d with AttrDict u with strict matching"""
    for key, value in u.items():
        if key not in d:
            key_full = ".".join(stack + [key])
            raise KeyError(f"The input key '{key_full}; does not exist in the config files.")
        if isinstance(value, collections.abc.Mapping):
            d.__dict__[key] = recursive_update_strict(d.get(key, AttrDict({})), value, stack + [key])
        elif isinstance(value, (list, tuple)):
            if value and isinstance(value[0], dict):
                d.__dict__[key] = [AttrDict(item) for item in value]
            else:
                d.__dict__[key] = value
        else:
            d.__dict__[key] = value
    return d


def parse_cmdline_arguments(args):
    """
    Parse arguments from command line.
    Syntax: --key1.key2.key3=value --> value
            --key1.key2.key3=      --> None
            --key1.key2.key3       --> True
            --key1.key2.key3!      --> False
    """
    cfg_cmd = {}
    for arg in args:
        assert arg.startswith("--")
        if "=" not in arg[2:]:
            key_str, value = (arg[2:-1], "false") if arg[-1] == "!" else (arg[2:], "true")
        else:
            key_str, value = arg[2:].split("=")
        keys_sub = key_str.split(".")
        cfg_sub = cfg_cmd
        for k in keys_sub[:-1]:
            cfg_sub.setdefault(k, {})
            cfg_sub = cfg_sub[k]
        assert keys_sub[-1] not in cfg_sub, keys_sub[-1]
        cfg_sub[keys_sub[-1]] = yaml.safe_load(value)
    return cfg_cmd
