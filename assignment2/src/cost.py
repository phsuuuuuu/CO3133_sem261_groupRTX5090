"""only the code currently not included in train.py"""

import argparse
import json
import time
from pathlib import Path

import torch

class TrainingTimer:
    def __init__(self, device):
        self.device = device
        self.time = 0
        self.device_name = torch.cuda.get_device_name(self.device) if self.device.type =="cuda" else "cpu"
        
    def __enter__(self):
        if self.device.type == "cuda": torch.cuda.synchronize(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.device.type == "cuda": torch.cuda.synchronize(self.device)
        self._end = time.perf_counter()
        self.time = self._end - self._start

def main():
    raise NotImplementedError
if __name__ == "__main__":
    raise NotImplementedError
