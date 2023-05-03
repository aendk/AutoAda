import torch
from torch._six import inf


class PSOptimizer:
    pass


class PSSGD(PSOptimizer):
    def __init__(self, eta: float = 0.5) -> None:
        self.eta = eta

    def update(self, gradients: torch.Tensor) -> torch.Tensor:
        return -self.eta * gradients


class PSAdagrad(PSOptimizer):
    def __init__(self, lr: float = 0.01, initial_accumulator_value: float = 0, eps: float = 1e-10) -> None:
        self.lr = lr
        self.eps = eps
        self.initial_accumulator_value = initial_accumulator_value

    def update(self, gradient: torch.Tensor, accumulator: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            update_accumulator = gradient * gradient
            update_gradient = -self.lr * gradient / (torch.sqrt(accumulator.to(gradient.device)+update_accumulator) + self.eps)
            return update_gradient, update_accumulator

    def update_in_place(self, gradient: torch.Tensor, parameter: torch.Tensor, accumulator: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            parameter[:] = -self.lr * gradient / (torch.sqrt(accumulator + gradient.square()) + self.eps)
            accumulator[:] = gradient.square()


class ReduceLROnPlateau(object):
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8, verbose=True):
        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self._last_lr = None
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold, threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        epoch = self.last_epoch + 1
        self.last_epoch += 1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = self.optimizer.lr

    def _reduce_lr(self, epoch):
        old_lr = self.optimizer.lr
        new_lr = max(old_lr * self.factor, self.min_lr)
        if old_lr - new_lr > self.eps:
            self.optimizer.lr = new_lr
            if self.verbose:
                print('Epoch {}: reducing learning rate to {:.4e}.'.format(epoch, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
