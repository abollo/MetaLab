import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
import warnings
import torch.distributed as dist
from utils.pytorch_env import *

class TORCH_config(object):
    """ parameters of pytroch(neural networks...,) """

    def __init__(self, args,fix_seed=None):
        if args is not None:
            self.__dict__.update(args.__dict__)
            self.distributed = self.world_size > 1
        else:
            self.seed = 42
            self.distributed = False
            pass

        self.nMostCls = 5000
        self.input_shape = [3, 224, 224]
        self.normal = "normal"

        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            cudnn.deterministic = True
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')
        self.gpu_device = pytorch_env(42)

        if self.gpu_device is not None:
            warnings.warn('You have chosen a specific GPU. This will completely '
                          'disable data parallelism.')



        if self.distributed:
            dist.init_process_group(backend=self.dist_backend, init_method=self.dist_url,world_size=self.world_size)
        self.pretrained = True

        self.lr = 0.0001
        # self.lr = 0.01


