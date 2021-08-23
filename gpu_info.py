import sys
import torch


def print_info():
    print('__Python VERSION:', sys.version)
    print('__pyTorch VERSION:', torch.__version__)
    print('__CUDA VERSION')
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('Active CUDA Device: GPU', torch.cuda.current_device())


def device_no():
    return torch.cuda.device_count()