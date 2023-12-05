import torch
import time


class Timer(object):
    def __init__(self):
        self.time = {}
        self.start = {}
        self.count = {}

    def start_timer(self, name):
        torch.cuda.synchronize()
        self.start[name] = time.time()

    def end_timer(self, name):
        torch.cuda.synchronize()
        if name not in self.start:
            raise f"{name} timer not start"

        if name not in self.time:
            self.time[name] = 0
            self.count[name] = 0
        self.time[name] += time.time() - self.start[name]
        self.count[name] += 1
        del self.start[name]

    def print(self):
        for key, value in self.time.items():
            print(f"{key} timer: {value/self.count[key]}")
