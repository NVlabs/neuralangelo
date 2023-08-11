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
import tqdm
import threading
import queue


class Dataset(torch.utils.data.Dataset):

    def __init__(self, cfg, is_inference=False, is_test=False):
        super().__init__()
        self.split = "test" if is_test else "val" if is_inference else "train"

    def _preload_worker(self, data_list, load_func, q, lock, idx_tqdm):
        # Keep preloading data in parallel.
        while True:
            idx = q.get()
            data_list[idx] = load_func(idx)
            with lock:
                idx_tqdm.update()
            q.task_done()

    def preload_threading(self, load_func, num_workers, data_str="images"):
        # Use threading to preload data in parallel.
        data_list = [None] * len(self)
        q = queue.Queue(maxsize=len(self))
        idx_tqdm = tqdm.tqdm(range(len(self)), desc=f"preloading {data_str} ({self.split})", leave=False)
        for i in range(len(self)):
            q.put(i)
        lock = threading.Lock()
        for ti in range(num_workers):
            t = threading.Thread(target=self._preload_worker,
                                 args=(data_list, load_func, q, lock, idx_tqdm), daemon=True)
            t.start()
        q.join()
        idx_tqdm.close()
        assert all(map(lambda x: x is not None, data_list))
        return data_list

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        return len(self.list)
