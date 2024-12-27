# -*- coding: utf-8 -*- 
# @Time : 2022/8/7 14:48 
# @Author : lepold
# @File : helpers.py

"""
Here are some simple Numpy functions or torch functions that help to achieve simulation or assimilation.

"""


import os
import torch
import numpy as np
import argparse



def torch_2_numpy(u, is_cuda=True):
    """
    Convert ``torch.Tensor`` to ``numpy.ndarray`` in cpu memory.

    """
    assert isinstance(u, torch.Tensor)
    if is_cuda:
        return u.cpu().numpy()
    else:
        return u.numpy()


def numpy2torch(u, is_cuda=True):
    assert isinstance(u, np.ndarray)
    if is_cuda:
        return torch.from_numpy(u).cuda()
    else:
        return torch.from_numpy(u)

