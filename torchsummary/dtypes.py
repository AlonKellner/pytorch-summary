import torch


class RandInt:
    def __init__(self, type=torch.ByteTensor, low=0, high=256):
        self.type = type
        self.low = low
        self.high = high