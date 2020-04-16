import math
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import  Optimizer
import numpy as np


# def adjust_learning_rate(optimizer, epoch, base_lr=0.001,warmup=1,milestones=None):
#     """Sets the learning rate: milestone is a list/tuple"""
#     def to(epoch):
#         if epoch <= warmup:
#             return 1
#         elif warmup < epoch <= milestones[0]:
#             return 0
#         for i in range(1, len(milestones)):
#             if milestones[i - 1] < epoch <= milestones[i]:
#                 return i
#         return len(milestones)
#     n = to(epoch)
#     global lr
#     lr =base_lr * (0.2 ** n)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def adjust_learning_rate(optimizer,base_lr=0.001, current_epoch=0,num_epochs=3, power=0.8,warmup=5):
    """Sets the learning rate: milestone is a list/tuple"""

    def lr_poly(base_lr, iter, max_iter, power):
        return base_lr * ((1 - float(iter) / max_iter) ** (power))
    if current_epoch<warmup:
        lr=1e-5*((current_epoch+1)*3)
    else:
        lr = lr_poly(base_lr, current_epoch, num_epochs, power)
    print('learning rate : {0}'.format(lr))
    optimizer.param_groups[0]['lr'] = lr

    return lr




class CyclicScheduler(_LRScheduler):
    def __init__(self, optimizer, epochs, min_lr_factor=0.05, max_lr=1.0):
        half_epochs = epochs // 2
        decay_epochs = epochs * 0.05

        lr_grow = np.linspace(min_lr_factor, max_lr, half_epochs)
        lr_down = np.linspace(max_lr, min_lr_factor, half_epochs - decay_epochs)
        lr_decay = np.linspace(min_lr_factor, min_lr_factor * 0.01, decay_epochs)
        self.learning_rates = np.concatenate((lr_grow, lr_down, lr_decay)) / max_lr
        super().__init__(optimizer)

    def get_lr(self):
        return [
            base_lr * self.learning_rates[self.last_epoch] for base_lr in self.base_lrs
        ]
