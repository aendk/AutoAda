import torch
from torch import nn
import math
import typing

"""
Adapted from jrzaurin's pytorch-widedeep repository (https://github.com/jrzaurin/pytorch-widedeep)
File: https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/models/tabular/linear/wide.py

"""


class Wide(nn.Module):
  r"""Defines a `Wide` (linear) model where the non-linearities are
  captured via the so-called crossed-columns. This can be used as the
  `wide` component of a Wide & Deep model.
  Parameters
  -----------
  input_dim: int
      size of the Embedding layer. `input_dim` is the summation of all the
      individual values for all the features that go through the wide
      model. For example, if the wide model receives 2 features with
      5 individual values each, `input_dim = 10`
  pred_dim: int, default = 1
      size of the ouput tensor containing the predictions. Note that unlike
      all the other models, the wide model is connected directly to the
      output neuron(s) when used to build a Wide and Deep model. Therefore,
      it requires the `pred_dim` parameter.
  Attributes
  -----------
  wide_linear: nn.Module
      the linear layer that comprises the wide branch of the model
  Examples
  --------
  >>> import torch
  >>> from pytorch_widedeep.models import Wide
  >>> X = torch.empty(4, 4).random_(6)
  >>> wide = Wide(input_dim=X.unique().size(0), pred_dim=1)
  >>> out = wide(X)
  """

  def __init__(self, input_dim: int, pred_dim: int = 1):
    super(Wide, self).__init__()

    self.input_dim = input_dim
    self.pred_dim = pred_dim

    self.wide_linear = nn.Embedding(input_dim, pred_dim)

    # (Sum(Embedding) + bias) is equivalent to (OneHotVector + Linear)
    self.bias = nn.Parameter(torch.zeros(pred_dim))

    self._reset_parameters()

  def _reset_parameters(self) -> None:
    r"""initialize Embedding and bias like nn.Linear. See [original
    implementation](https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear).
    """
    # NOTE this initialization is not used in AdaPM training. We use xavier instead
    nn.init.kaiming_uniform_(self.wide_linear.weight, a=math.sqrt(5))
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.wide_linear.weight)
    bound = 1 / math.sqrt(fan_in)
    nn.init.uniform_(self.bias, -bound, bound)

  def forward(self, X):
    r"""Forward pass. Simply connecting the Embedding layer with the output
    neuron(s)"""

    embeddings = self.wide_linear(X.long())

    out = embeddings.sum(dim=1) + self.bias
    return out, embeddings
