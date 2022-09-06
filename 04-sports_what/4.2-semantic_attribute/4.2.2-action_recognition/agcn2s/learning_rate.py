import math
from paddle.optimizer.lr import *
import numpy as np

class CustomWarmupAdjustDecay(LRScheduler):
    r"""
    We combine warmup and stepwise-cosine which is used in slowfast model.
    Args:
        step_base_lr (float): start learning rate used in warmup stage.
        warmup_epochs (int): the number epochs of warmup.
        lr_decay_rate (float|int, optional): base learning rate decay rate.
        step (int): step in change learning rate.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False`` .
    Returns:
        ``CosineAnnealingDecay`` instance to schedule learning rate.
    """

    def __init__(self,
                 step_base_lr,
                 warmup_epochs,
                 lr_decay_rate,
                 boundaries,
                 num_iters=None,
                 last_epoch=-1,
                 verbose=False):
        self.step_base_lr = step_base_lr
        self.warmup_epochs = warmup_epochs
        self.lr_decay_rate = lr_decay_rate
        self.boundaries = boundaries
        self.num_iters = num_iters
        #call step() in base class, last_lr/last_epoch/base_lr will be update
        super(CustomWarmupAdjustDecay, self).__init__(last_epoch=last_epoch,
                                                      verbose=verbose)

    def step(self, epoch=None):
        """
        ``step`` should be called after ``optimizer.step`` . It will update the learning rate in optimizer according to current ``epoch`` .
        The new learning rate will take effect on next ``optimizer.step`` .
        Args:
            epoch (int, None): specify current epoch. Default: None. Auto-increment from last_epoch=-1.
        Returns:
            None
        """
        if epoch is None:
            if self.last_epoch == -1:
                self.last_epoch += 1
            else:
                self.last_epoch += 1 / self.num_iters  # update step with iters
        else:
            self.last_epoch = epoch

        self.last_lr = self.get_lr()

        if self.verbose:
            print('Epoch {}: {} set learning rate to {}.'.format(
                self.last_epoch, self.__class__.__name__, self.last_lr))

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = self.step_base_lr * (self.last_epoch + 1) / self.warmup_epochs
        else:
            lr = self.step_base_lr * (self.lr_decay_rate**np.sum(
                self.last_epoch >= np.array(self.boundaries)))
        return lr