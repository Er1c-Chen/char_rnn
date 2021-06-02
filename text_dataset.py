import torch
import numpy as np


class TextDataset(object):
    def __init__(self, arr):
        self.arr = arr

    def __getitem__(self, item):
        x = self.arr[item, :]

        # 构造 label
        y = torch.zeros(x.shape)
        # 将输入的第一个字符作为最后一个输入的label
        y[:-1], y[-1] = x[1:], x[0]
        return x, y

    def __len__(self):
        return self.arr.shape[0]