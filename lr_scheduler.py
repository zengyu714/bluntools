import warnings
import numpy as np

from torch.optim import Optimizer


# Reference: https://github.com/Jiaming-Liu/pytorch-lr-scheduler/blob/master/lr_scheduler.py
class ReduceLROnPlateau(object):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        factor: factor by which the learning rate will
            be reduced. new_lr = lr * factor
        patience: number of epochs with no improvement
            after which learning rate will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {min, max}. In `min` mode,
            lr will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing.
        epsilon: threshold for measuring the new optimum,
            to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after lr has been reduced.
        min_lr: lower bound on the learning rate.


    Example:
        --- optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        --- scheduler = ReduceLROnPlateau(optimizer, 'min')
        --- for epoch in range(10):
        ---     train(...)
        ---     val_acc, val_loss = validate(...)
        ---     scheduler.step(val_loss, epoch)
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 verbose=0, epsilon=1e-4, cooldown=0, min_lr=0.0):
        super(ReduceLROnPlateau, self).__init__()

        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau '
                             'does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.monitor_op = None
        self.wait = 0
        self.best = 0
        self.mode = mode
        assert isinstance(optimizer, Optimizer)
        self.optimizer = optimizer
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['min', 'max']:
            raise RuntimeError('Learning Rate Plateau Reducing mode %s is unknown!')
        if self.mode == 'min':
            self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.epsilon)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0
        self.lr_epsilon = self.min_lr * 1e-4

    def reset(self):
        self._reset()

    def step(self, metrics):
        current = metrics
        if current is None:
            warnings.warn('Learning Rate Plateau Reducing requires metrics available!', RuntimeWarning)
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                if self.wait >= self.patience:
                    for param_group in self.optimizer.param_groups:
                        old_lr = float(param_group['lr'])
                        if old_lr > self.min_lr + self.lr_epsilon:
                            new_lr = old_lr * self.factor
                            new_lr = max(new_lr, self.min_lr)
                            param_group['lr'] = new_lr
                            if self.verbose > 0:
                                _show_learning_rate(new_lr)
                            self.cooldown_counter = self.cooldown
                            self.wait = 0
                self.wait += 1

    def in_cooldown(self):
        return self.cooldown_counter > 0


def auto_lr_scheduler(optimizer, patience=200, cooldown=100, verbose=1, min_lr=1e-10):
    return ReduceLROnPlateau(optimizer, patience=patience, cooldown=cooldown, verbose=verbose, min_lr=min_lr)


def step_lr_scheduler(optimizer, epoch, milestones=None, init_lr=0.0001, gamma=0.1, instant=False):
    """Decay learning rate by a factor of 0.1 when epoch reaches milestones

    Arguments:
        optimizer: wrapped optimizer
        epoch: epoch :-)
        milestones:
            + int: decay by gamma every milestones epochs
            + list: decay by gamma once the number of epoch reaches one of the milestones
        init_lr: initial learning rate
        gamma: multiplicative factor of learning rate decay
        instant: compute learning rate immediately, especially at the beginning of train
    """
    if isinstance(milestones, (int, float)):
        return exp_lr_scheduler(optimizer, epoch, init_lr=init_lr, gamma=gamma, lr_decay_epoch=milestones)

    if epoch in milestones or instant:
        factor = np.searchsorted(milestones, epoch + 1)
        lr = init_lr * (gamma ** factor)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        _show_learning_rate(lr)
    return optimizer


def exp_lr_scheduler(optimizer, epoch, init_lr=0.0001, gamma=0.1, lr_decay_epoch=100):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (gamma ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        _show_learning_rate(lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def _show_learning_rate(lr):
    print('*' * 60,
          '     Learning rate is set to {:.0e}     '.format(lr).center(60, '*'),
          '*' * 60, sep='\n')
