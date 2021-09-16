import torch


class RandInt:
    def __init__(self, type=torch.ByteTensor, high=256):
        self.type = type
        self.high = high