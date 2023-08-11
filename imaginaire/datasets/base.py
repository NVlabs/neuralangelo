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
import pickle
from collections import OrderedDict
from functools import partial
from inspect import signature

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from imaginaire.datasets.folder import FolderDataset
from imaginaire.datasets.lmdb import \
    IMG_EXTENSIONS, HDR_IMG_EXTENSIONS, LMDBDataset
from imaginaire.datasets.object_store import ObjectStoreDataset
from imaginaire.datasets.utils.data import \
    (VIDEO_EXTENSIONS, Augmentor,
     load_from_folder, load_from_lmdb, load_from_object_store)
from imaginaire.datasets.utils.lmdb import create_metadata


DATASET_TYPES = ['lmdb', 'folder', 'object_store']


class BaseDataset(data.Dataset):
    r"""Base class for image/video datasets.

    Args:
        cfg (Config object): Input config.
        is_inference (bool): Training if False, else validation.
        is_test (bool): Final test set after training and validation.
    """

    def __init__(self, cfg, is_inference, is_test):
        super(BaseDataset, self).__init__()

        self.cfg = cfg
        self.is_inference = is_inference
        self.is_test = is_test
        if self.is_test:
            self.cfgdata = self.cfg.test_data
            data_info = self.cfgdata.test
        else:
            self.cfgdata = self.cfg.data
            if self.is_inference:
                data_info = self.cfgdata.val
            else:
                data_info = self.cfgdata.train
        self.name = self.cfgdata.name
        self.lmdb_roots = data_info.roots
        self.dataset_type = getattr(data_info, 'dataset_type', None)
        self.cache = getattr(self.cfgdata, 'cache', None)
        self.interpolator = getattr(self.cfgdata, 'interpolator', "INTER_LINEAR")

        # Get AWS secret keys.
        if self.dataset_type == 'object_store':
            assert hasattr(cfg, 'aws_credentials_file')
            self.aws_credentials_file = cfg.aws_credentials_file

        # Legacy lmdb/folder only support.
        if self.dataset_type is None:
            self.dataset_is_lmdb = getattr(data_info, 'is_lmdb', False)
            if self.dataset_is_lmdb:
                self.dataset_type = 'lmdb'
            else:
                self.dataset_type = 'folder'
        # Legacy support ends.

        assert self.dataset_type in DATASET_TYPES
        if self.dataset_type == 'lmdb':
            # Add handle to function to load data from LMDB.
            self.load_from_dataset = load_from_lmdb
        elif self.dataset_type == 'folder':
            # For some unpaired experiments, we would like the dataset to be presented in a paired way

            if hasattr(self.cfgdata, 'paired') is False:
                self.cfgdata.paired = self.paired
            # Add handle to function to load data from folder.
            self.load_from_dataset = load_from_folder
            # Create metadata for folders.
            print('Creating metadata')
            all_filenames, all_metadata = [], []
            if self.is_test:
                cfg.data_backup = cfg.data
                cfg.data = cfg.test_data
            for root in self.lmdb_roots:
                filenames, metadata = create_metadata(
                    data_root=root, cfg=cfg, paired=self.cfgdata['paired'])
                all_filenames.append(filenames)
                all_metadata.append(metadata)
            if self.is_test:
                cfg.data = cfg.data_backup
        elif self.dataset_type == 'object_store':
            # Add handle to function to load data from AWS S3.
            self.load_from_dataset = load_from_object_store

        # Get the types of data stored in dataset, and their extensions.
        self.data_types = []  # Names of data types.
        self.dataset_data_types = []  # These data types are in the dataset.
        self.image_data_types = []  # These types are images.
        self.hdr_image_data_types = []  # These types are HDR images.
        self.normalize = {}  # Does this data type need normalization?
        self.extensions = {}  # What is this data type's file extension.
        self.is_mask = {}  # Whether this data type is discrete masks?
        self.num_channels = {}  # How many channels does this data type have?
        self.pre_aug_ops = {}  # Ops on data type before augmentation.
        self.post_aug_ops = {}  # Ops on data type after augmentation.

        # Extract info from data types.
        for data_type in self.cfgdata.input_types:
            name = list(data_type.keys())
            assert len(name) == 1
            name = name[0]
            info = data_type[name]

            if 'ext' not in info:
                info['ext'] = None
            if 'normalize' not in info:
                info['normalize'] = False
            if 'is_mask' not in info:
                info['is_mask'] = False
            if 'pre_aug_ops' not in info:
                info['pre_aug_ops'] = 'None'
            if 'post_aug_ops' not in info:
                info['post_aug_ops'] = 'None'
            if 'computed_on_the_fly' not in info:
                info['computed_on_the_fly'] = False
            if 'num_channels' not in info:
                info['num_channels'] = None

            self.data_types.append(name)
            if not info['computed_on_the_fly']:
                self.dataset_data_types.append(name)

            self.extensions[name] = info['ext']
            self.normalize[name] = info['normalize']
            self.num_channels[name] = info['num_channels']
            self.pre_aug_ops[name] = [op.strip() for op in
                                      info['pre_aug_ops'].split(',')]
            self.post_aug_ops[name] = [op.strip() for op in
                                       info['post_aug_ops'].split(',')]
            self.is_mask[name] = info['is_mask']
            if info['ext'] is not None and (info['ext'] in IMG_EXTENSIONS or info['ext'] in VIDEO_EXTENSIONS):
                self.image_data_types.append(name)
            if info['ext'] is not None and info['ext'] in HDR_IMG_EXTENSIONS:
                self.hdr_image_data_types.append(name)

        # Add some info into cfgdata for legacy support.
        self.cfgdata.data_types = self.data_types
        self.cfgdata.num_channels = [self.num_channels[name]
                                     for name in self.data_types]

        # Augmentations which need full dict.
        self.full_data_post_aug_ops, self.full_data_ops = [], []
        if hasattr(self.cfgdata, 'full_data_ops'):
            ops = self.cfgdata.full_data_ops
            self.full_data_ops.extend([op.strip() for op in ops.split(',')])
        if hasattr(self.cfgdata, 'full_data_post_aug_ops'):
            ops = self.cfgdata.full_data_post_aug_ops
            self.full_data_post_aug_ops.extend(
                [op.strip() for op in ops.split(',')])

        # These are the labels which will be concatenated for generator input.
        self.input_labels = []
        if hasattr(self.cfgdata, 'input_labels'):
            self.input_labels = self.cfgdata.input_labels

        # These are the keypoints which also need to be augmented.
        self.keypoint_data_types = []
        if hasattr(self.cfgdata, 'keypoint_data_types'):
            self.keypoint_data_types = self.cfgdata.keypoint_data_types

        # Create augmentation operations.
        aug_list = data_info.augmentations
        individual_video_frame_aug_list = getattr(data_info, 'individual_video_frame_augmentations', dict())
        post_aug_list = getattr(data_info, 'post_augmentations', dict())
        self.augmentor = Augmentor(
            aug_list, individual_video_frame_aug_list, post_aug_list, self.image_data_types, self.is_mask,
            self.keypoint_data_types, self.interpolator)
        self.augmentable_types = self.image_data_types + \
            self.keypoint_data_types

        # Create torch transformations.
        self.transform = {}
        for data_type in self.image_data_types:
            normalize = self.normalize[data_type]
            self.transform[data_type] = self._get_transform(
                normalize, self.num_channels[data_type])

        # Create torch transformations for HDR images.
        for data_type in self.hdr_image_data_types:
            normalize = self.normalize[data_type]
            self.transform[data_type] = self._get_transform(
                normalize, self.num_channels[data_type])

        # Initialize handles.
        self.sequence_lists = []  # List of sequences per dataset root.
        self.lmdbs = {}  # Dict for list of lmdb handles per data type.
        for data_type in self.dataset_data_types:
            self.lmdbs[data_type] = []
        self.dataset_probability = None
        self.additional_lists = []

        # Load each dataset.
        for idx, root in enumerate(self.lmdb_roots):
            if self.dataset_type == 'lmdb':
                self._add_dataset(root)
            elif self.dataset_type == 'folder':
                self._add_dataset(root, filenames=all_filenames[idx],
                                  metadata=all_metadata[idx])
            elif self.dataset_type == 'object_store':
                self._add_dataset(
                    root, aws_credentials_file=self.aws_credentials_file)

        # Compute dataset statistics and create whatever self.variables required
        # for the specific dataloader.
        self._compute_dataset_stats()

        # Build index of data to sample.
        self.mapping, self.epoch_length = self._create_mapping()

    def _create_mapping(self):
        r"""Creates mapping from data sample idx to actual LMDB keys.
            All children need to implement their own.

        Returns:
            self.mapping (list): List of LMDB keys.
        """
        raise NotImplementedError

    def _compute_dataset_stats(self):
        r"""Computes required statistics about dataset.
           All children need to implement their own.
        """
        pass

    def __getitem__(self, index):
        r"""Entry function for dataset."""
        raise NotImplementedError

    def _get_transform(self, normalize, num_channels):
        r"""Convert numpy to torch tensor.

        Args:
            normalize (bool): Normalize image i.e. (x - 0.5) * 2.
                Goes from [0, 1] -> [-1, 1].
        Returns:
            Composed list of torch transforms.
        """
        transform_list = [transforms.ToTensor()]
        if normalize:
            transform_list.append(
                transforms.Normalize((0.5, ) * num_channels,
                                     (0.5, ) * num_channels, inplace=True))
        return transforms.Compose(transform_list)

    def _add_dataset(self, root, filenames=None, metadata=None,
                     aws_credentials_file=None):
        r"""Adds an LMDB dataset to a list of datasets.

        Args:
            root (str): Path to LMDB or folder dataset.
            filenames: List of filenames for folder dataset.
            metadata: Metadata for folder dataset.
            aws_credentials_file: Path to file containing AWS credentials.
        """
        if aws_credentials_file and self.dataset_type == 'object_store':
            object_store_dataset = ObjectStoreDataset(
                root, aws_credentials_file, cache=self.cache)
            sequence_list = object_store_dataset.sequence_list
        else:
            # Get sequences associated with this dataset.
            if filenames is None:
                list_path = 'all_filenames.json'
                with open(os.path.join(root, list_path)) as fin:
                    sequence_list = OrderedDict(json.load(fin))
            else:
                sequence_list = filenames

            additional_path = 'all_indices.json'
            if os.path.exists(os.path.join(root, additional_path)):
                print('Using additional list for object indices.')
                with open(os.path.join(root, additional_path)) as fin:
                    additional_list = OrderedDict(json.load(fin))
                self.additional_lists.append(additional_list)
        self.sequence_lists.append(sequence_list)

        # Get LMDB dataset handles.
        for data_type in self.dataset_data_types:
            if self.dataset_type == 'lmdb':
                self.lmdbs[data_type].append(
                    LMDBDataset(os.path.join(root, data_type)))
            elif self.dataset_type == 'folder':
                self.lmdbs[data_type].append(
                    FolderDataset(os.path.join(root, data_type), metadata))
            elif self.dataset_type == 'object_store':
                # All data types use the same handle.
                self.lmdbs[data_type].append(object_store_dataset)

    def perform_individual_video_frame(self, data, augment_ops):
        r"""Perform data augmentation on images only.

        Args:
            data (dict): Keys are from data types. Values can be numpy.ndarray
                or list of numpy.ndarray (image or list of images).
            augment_ops (list): The augmentation operations for individual frames.
        Returns:
            (tuple):
              - data (dict): Augmented data, with same keys as input data.
              - is_flipped (bool): Flag which tells if images have been
                left-right flipped.
        """
        if augment_ops:
            all_data = dict()
            for ix, key in enumerate(data.keys()):
                if ix == 0:
                    num = len(data[key])
                    for j in range(num):
                        all_data['%d' % j] = dict()
                for j in range(num):
                    all_data['%d' % j][key] = data[key][j:(j+1)]
            for j in range(num):
                all_data['%d' % j], _ = self.perform_augmentation(
                    all_data['%d' % j], paired=True, augment_ops=augment_ops)
            for key in data.keys():
                tmp = []
                for j in range(num):
                    tmp += all_data['%d' % j][key]
                data[key] = tmp
        return data

    def perform_augmentation(self, data, paired, augment_ops=None):
        r"""Perform data augmentation on images only.

        Args:
            data (dict): Keys are from data types. Values can be numpy.ndarray
                or list of numpy.ndarray (image or list of images).
            paired (bool): Apply same augmentation to all input keys?
            augment_ops (list): The augmentation operations.
        Returns:
            (tuple):
              - data (dict): Augmented data, with same keys as input data.
              - is_flipped (bool): Flag which tells if images have been
                left-right flipped.
        """
        aug_inputs = {}
        for data_type in self.augmentable_types:
            aug_inputs[data_type] = data[data_type]

        augmented, is_flipped = self.augmentor.perform_augmentation(
            aug_inputs, paired=paired, augment_ops=augment_ops)

        for data_type in self.augmentable_types:
            data[data_type] = augmented[data_type]

        return data, is_flipped

    def flip_hdr(self, data, is_flipped=False):
        r"""Flip hdr images.

        Args:
            data (dict): Keys are from data types. Values can be numpy.ndarray
                or list of numpy.ndarray (image or list of images).
            is_flipped (bool): Applying left-right flip to the hdr images
        Returns:
            (tuple):
              - data (dict): Augmented data, with same keys as input data.
        """
        if is_flipped is False:
            return data

        for data_type in self.hdr_image_data_types:
            # print('Length of data: {}'.format(len(data[data_type])))
            data[data_type][0] = data[data_type][0][:, ::-1, :].copy()
        return data

    def to_tensor(self, data):
        r"""Convert all images to tensor.

        Args:
            data (dict): Dict containing data_type as key, with each value
                as a list of numpy.ndarrays.
        Returns:
            data (dict): Dict containing data_type as key, with each value
            as a list of torch.Tensors.
        """
        for data_type in self.image_data_types:
            for idx in range(len(data[data_type])):
                if data[data_type][idx].dtype == np.uint16:
                    data[data_type][idx] = data[data_type][idx].astype(
                        np.float32)
                data[data_type][idx] = self.transform[data_type](
                    data[data_type][idx])
        for data_type in self.hdr_image_data_types:
            for idx in range(len(data[data_type])):
                data[data_type][idx] = self.transform[data_type](
                    data[data_type][idx])
        return data

    def apply_ops(self, data, op_dict, full_data=False):
        r"""Apply any ops from op_dict to data types.

        Args:
            data (dict): Dict containing data_type as key, with each value
                as a list of numpy.ndarrays.
            op_dict (dict): Dict containing data_type as key, with each value
                containing string of operations to apply.
            full_data (bool): Do these ops require access to the full data?
        Returns:
            data (dict): Dict containing data_type as key, with each value
            modified by the op if any.
        """
        if full_data:
            # op needs entire data dict.
            for op in op_dict:
                if op == 'None':
                    continue
                op, op_type = self.get_op(op)
                assert op_type == 'full_data'
                data = op(data)
        else:
            # op per data type.
            if not op_dict:
                return data
            for data_type in data:
                for op in op_dict[data_type]:
                    if op == 'None':
                        continue
                    op, op_type = self.get_op(op)
                    data[data_type] = op(data[data_type])

                    if op_type == 'vis':
                        # We have converted this data type to an image. Enter it
                        # in self.image_data_types and give it a torch
                        # transform.
                        if data_type not in self.image_data_types:
                            self.image_data_types.append(data_type)
                            normalize = self.normalize[data_type]
                            num_channels = self.num_channels[data_type]
                            self.transform[data_type] = \
                                self._get_transform(normalize, num_channels)
                    elif op_type == 'convert':
                        continue
                    elif op_type is None:
                        continue
                    else:
                        raise NotImplementedError
        return data

    def get_op(self, op):
        r"""Get function to apply for specific op.

        Args:
            op (str): Name of the op.
        Returns:
            function handle.
        """
        def list_to_tensor(data):
            r"""Convert list of numeric values to tensor."""
            assert isinstance(data, list)
            return torch.from_numpy(np.array(data, dtype=np.float32))

        def decode_json_list(data):
            r"""Decode list of strings in json to objects."""
            assert isinstance(data, list)
            return [json.loads(item) for item in data]

        def decode_pkl_list(data):
            r"""Decode list of pickled strings to objects."""
            assert isinstance(data, list)
            return [pickle.loads(item) for item in data]

        def list_to_numpy(data):
            r"""Convert list of numeric values to numpy array."""
            assert isinstance(data, list)
            return np.array(data)

        def l2_normalize(data):
            r"""L2 normalization."""
            assert isinstance(data, torch.Tensor)
            import torch.nn.functional as F
            return F.normalize(data, dim=1)

        if op == 'to_tensor':
            return list_to_tensor, None
        elif op == 'decode_json':
            return decode_json_list, None
        elif op == 'decode_pkl':
            return decode_pkl_list, None
        elif op == 'to_numpy':
            return list_to_numpy, None
        elif op == 'l2_norm':
            return l2_normalize, None
        elif '::' in op:
            parts = op.split('::')
            if len(parts) == 2:
                module, function = parts
                module = importlib.import_module(module)
                function = getattr(module, function)
                sig = signature(function)
                num_params = len(sig.parameters)
                assert num_params in [3, 4], \
                    'Full data functions take in (cfgdata, is_inference, ' \
                    'full_data) or (cfgdata, is_inference, self, full_data) ' \
                    'as input.'
                if num_params == 3:
                    function = partial(
                        function, self.cfgdata, self.is_inference)
                elif num_params == 4:
                    function = partial(
                        function, self.cfgdata, self.is_inference, self)
                function_type = 'full_data'
            elif len(parts) == 3:
                function_type, module, function = parts
                module = importlib.import_module(module)

                # Get function inputs, if provided.
                partial_fn = False
                if '(' in function and ')' in function:
                    partial_fn = True
                    function, params = self._get_fn_params(function)

                function = getattr(module, function)

                # Create partial function.
                if partial_fn:
                    function = partial(function, **params)

                # Get function signature.
                sig = signature(function)
                num_params = 0
                for param in sig.parameters.values():
                    if param.kind == param.POSITIONAL_OR_KEYWORD:
                        num_params += 1

                if function_type == 'vis':
                    if num_params != 9:
                        raise ValueError(
                            'vis function type needs to take ' +
                            '(resize_h, resize_w, crop_h, crop_w, ' +
                            'original_h, original_w, is_flipped, cfgdata, ' +
                            'data) as input.')
                    function = partial(function,
                                       self.augmentor.resize_h,
                                       self.augmentor.resize_w,
                                       self.augmentor.crop_h,
                                       self.augmentor.crop_w,
                                       self.augmentor.original_h,
                                       self.augmentor.original_w,
                                       self.augmentor.is_flipped,
                                       self.cfgdata)
                elif function_type == 'convert':
                    if num_params != 1:
                        raise ValueError(
                            'convert function type needs to take ' +
                            '(data) as input.')
                else:
                    raise ValueError('Unknown op: %s' % (op))
            else:
                raise ValueError('Unknown op: %s' % (op))
            return function, function_type
        else:
            raise ValueError('Unknown op: %s' % (op))

    def _get_fn_params(self, function_string):
        r"""Find key-value inputs to function from string definition.

        Args:
            function_string (str): String with function name and args. e.g.
            my_function(a=10, b=20).
        Returns:
            function (str): Name of function.
            params (dict): Key-value params for function.
        """
        start = function_string.find('(')
        end = function_string.find(')')
        function = function_string[:start]
        params_str = function_string[start+1:end]
        params = {}
        for item in params_str.split(':'):
            key, value = item.split('=')
            try:
                params[key] = float(value)
            except Exception:
                params[key] = value
        return function, params

    def __len__(self):
        return self.epoch_length
