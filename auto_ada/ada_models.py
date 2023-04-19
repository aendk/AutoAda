import torch
import torch.nn as nn
from torch.nn import init, Parameter
import adaps
import math
import sys

from auto_ada.optimizer import PSOptimizer
import auto_ada.utils as utils


class AdaModel(nn.Module):

    def __init__(self, model: nn.Module) -> None:
        super().__init__()

        self.user_model = model
        self.replace_embeddings()
        self.key_offset = -1
        self.kv = None
        self.opt = None
        self._param_buffers = {}
        for (name, param) in self.named_parameters():
            self._param_buffers[name] = torch.empty((2,)+param.size())

    def replace_embeddings(self):
        """
        replaces all nn.Embeddings() with PSEmbeddings.
        """
        for name, elem in self.user_model.named_modules():
            if elem != self.user_model and isinstance(elem, nn.Embedding):
                utils.replace_embedding(self.user_model, name)

    def kv_init(self, kv: adaps.Worker, key_offset=0, opt: PSOptimizer = None, init_vals=True, signalIntent=True):
        kv.begin_setup()
        # init self
        self.kv = kv
        self.key_offset = key_offset
        self.opt = opt
        if signalIntent:
            self.intent()  # intent for all dense parameters from clock 0 to infinity
        # init the densely accessed regular nn.parameter(s) first
        offset = self.key_offset

        for i, (name, param) in enumerate(self.named_parameters()):
            if init_vals:
                self._param_buffers[name][0] = param.clone().detach()
                if self.opt:
                    self._param_buffers[name][1] = self.opt.initial_accumulator_value
                self.kv.set(torch.tensor([offset], dtype=torch.int64), self._param_buffers[name])
            offset += 1

        # init PS-submodules last, key_offset is already set in lens() for these layers.
        for module in self.user_model.modules():
            if module != self.user_model and hasattr(module, "kv_init"):
                module.kv_init(kv, offset, opt, init_vals)
                offset += len(module.lens())

    def grad_hook(self, key: torch.Tensor, name) -> torch.Tensor:
        def hook(grad: torch.Tensor) -> torch.Tensor:
            grad = clip_grad_norm(grad, 100)
            self.opt.update_in_place(grad.cpu(), self._param_buffers[name][0], self._param_buffers[name][1])
            self.kv.push(key, self._param_buffers[name], True)
            return grad
        return hook

    def intent(self, start=0, stop=sys.maxsize):

        num_parameters = sum(1 for i in self.parameters())
        keys = torch.arange(num_parameters) + self.key_offset
        if start == 0 and stop == sys.maxsize:
            print(f"perpetual intent on dense param keys={keys}")
        self.kv.intent(keys, start, stop)

    def pull(self):
        for i, (name, param) in enumerate(self.named_parameters()):
            key = torch.tensor([i+self.key_offset])
            self.kv.pull(key, self._param_buffers[name])
            newParam = Parameter(self._param_buffers[name][0].to(param.device, copy=True, non_blocking=True))
            newParam.register_hook(self.grad_hook(key, name))
            utils.rsetattr(self, name, newParam)

    def forward(self, *args, **kwargs) -> (torch.Tensor, torch.BoolTensor):
        self.kv.advance_clock()
        self.pull()
        return self.user_model(*args, **kwargs)

    def lens(self):
        lens = torch.tensor([param.flatten().shape[0] for param in self.parameters()], dtype=torch.int64) * 2  # twice for optim params
        for module in self.user_model.modules():
            if module != self.user_model and hasattr(module, "lens"):
                lens = torch.cat((lens, module.lens()))

        return lens

    def extra_repr(self):
        return f"Dense PS-enabled:"


class PSEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int = 1024,
        embedding_dim: int = 512,
        padding_idx: int = -1,  # if set to something greater, zero the respective values + optimizer-vals
        max_size: int = 2**20
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._buffer = None
        self.padding_idx = -1  # disabled for now.padding_idx
        self.max_size = max_size
        self.kv = None
        self.opt = None
        self.key_offset = -1

    def kv_init(self, kv: adaps.Worker, key_offset=0, opt: PSOptimizer = None, init_vals=True, kaiming=False):
        self.kv = kv
        self.key_offset = key_offset
        self.opt = opt

        if init_vals:
            for ids in torch.LongTensor(range(self.num_embeddings)).split(self.max_size):
                keys = ids + self.key_offset
                values = torch.empty(keys.size()+(2, self.embedding_dim), dtype=torch.float32)
                nn.init.xavier_uniform_(PSEmbedding._embeddings(values))

                if self.opt:
                    PSEmbedding._accumulators(values)[:] = self.opt.initial_accumulator_value
                self.kv.set(keys.long(), values, True)
            self.kv.waitall()

    def _embeddings(buffer):
        slice_dim = buffer.dim() - 2
        return buffer.select(slice_dim, 0)

    def _accumulators(buffer):
        slice_dim = buffer.dim() - 2
        return buffer.select(slice_dim, 1)

    def intent(self, ids: torch.Tensor, start, stop=0):
        keys = ids.flatten() + self.key_offset
        self.kv.intent(keys, start, stop)

    def pull(self, ids: torch.Tensor):
        keys = ids.flatten() + self.key_offset
        size = ids.size() + (2, self.embedding_dim)
        self._buffer = torch.empty(size, dtype=torch.float32)
        keys = keys.cpu()
        self.kv.pull(keys, self._buffer)

    def forward(self, ids: torch.Tensor, device=None):
        if self._buffer is None:
            self.pull(ids)

        embeddings = PSEmbedding._embeddings(self._buffer).to(device=device).requires_grad_()
        if self.training and self.opt:
            embeddings.register_hook(self.grad_hook(ids))
        elif not self.training:
            self._buffer = None

        return embeddings

    def grad_hook(self, ids: torch.Tensor) -> torch.Tensor:
        def hook(grad: torch.Tensor) -> torch.Tensor:
            grad = clip_grad_norm(grad, 100)
            keys = ids.flatten() + self.key_offset
            self.opt.update_in_place(grad.cpu(), PSEmbedding._embeddings(self._buffer), PSEmbedding._accumulators(self._buffer))
            self.kv.push(keys.cpu(), self._buffer.cpu(), True)
            self._buffer = None

            return grad
        return hook

    def lens(self):
        return torch.ones(self.num_embeddings, dtype=torch.int64) * self.embedding_dim * 2  # twice embedding_dim for optim params

    def extra_repr(self):
       return f"PSEmbedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"


def clip_grad_norm(
        grad: torch.Tensor, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False) -> torch.Tensor:
    r"""Clips gradient norm of a tensor, based on torch.clip_grad_norm_. Works slightly different than the original clip_grad_norm_: does not work in place, returns the clipped gradient instead, and does not support all norms (e.g., 0-norm is not supported) """
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    device = grad.device
    if norm_type == torch.inf:
        total_norm = grad.detach().abs().max().to(device)
    else:
        total_norm = torch.norm(grad.detach(), norm_type).to(device)
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    return grad.detach().mul(clip_coef_clamped.to(grad.device))
