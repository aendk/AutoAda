import torch
import functools
from torch import nn
from typing import Any, Callable, Iterable, TypeVar, Generic, Sequence, List, Optional, Union
import auto_ada.ada_models


def replace_embedding(module, name_to_replace, mock_embedding=False, simulation_queue: Optional = None):
  # recursive call like "conv1.embedding"; call itself with module.conv1 + name_to_replace=embedding
  if "." in name_to_replace:
    # split the module-attribute from its child-attributes
    first, path = name_to_replace.split('.', 1)
    # 1. combine first w/ existing module
    target_attr = getattr(module, first)
    # 2. call this as module with the remaining path.
    replace_embedding(target_attr, path, mock_embedding, simulation_queue)

    return

  # can replace regular embeddings. Embeddings in nn.Sequential are handled below.
  for attr_str in dir(module):
    target_attr = getattr(module, attr_str)
    if attr_str == name_to_replace:

      if mock_embedding:

        new_emb = MockEmbedding(target_attr, name_to_replace, simulation_queue)
        # print(f"new mockembedding={new_emb} slotting in as={name_to_replace}, target_attr={target_attr}")
      else:
        new_emb = auto_ada.ada_models.PSEmbedding(target_attr.num_embeddings, target_attr.embedding_dim)
      # print(f"new embedding={new_emb} slotting in as={name_to_replace}, target_attr={target_attr}")
      setattr(module, attr_str, new_emb)

  if isinstance(module, torch.nn.Sequential):
    for num, elem in enumerate(module):
      if isinstance(elem, nn.Embedding):
        if mock_embedding:
          new_emb = MockEmbedding(module[num], "seq_" + "emb", simulation_queue)
        else:
          new_emb = auto_ada.ada_models.PSEmbedding(module[num].num_embeddings, module[num].embedding_dim)

        module[num] = new_emb


class MockEmbedding(nn.Module):

  def __init__(self, user_emb, original_name, simulation_queue):

    super().__init__()
    self.simulation_queue = simulation_queue
    self.name = original_name
    # self.user_emb = user_emb  # TODO make actual embedding optional, to not waste memory
    self.is_adaps = True
    self.key_offset = user_emb.key_offset
    self.plausible_output = torch.rand(1, user_emb.embedding_dim)  # minimal plausible output

  def forward(self, *inputs, **kwargs):  # TODO engineering: handle kwargs(=dict()) too. (cf. https://www.digitalocean.com/community/tutorials/how-to-use-args-and-kwargs-in-python-3)
    if self.simulation_queue is not None:  # record input into queue
      input_list = list()
      input_list.append(self.key_offset)
      input_list.append(*inputs)
      self.simulation_queue.put(input_list)

    return self.plausible_output  # if this output goes directly into sum() or similar ops


class MockLayer(nn.Module):
  def __init__(self, layer, embedding_downstream=False):
    super().__init__()
    # we need to retain this layer: if it contains an embedding, we need to start simulating it to reach the embedding
    self.replaced_layer = layer
    self.embedding_downstream = embedding_downstream

  def forward(self, *inputs, **kwargs):

    if self.embedding_downstream:
        output = self.replaced_layer(*inputs, **kwargs)
        return output
    else:
      return

  def extra_repr(self):
    return f"Mocklayer, embedding-downstream:{self.embedding_downstream}"


def recursive_wrap_children(module, simulation_queue):
  """
    Recursively wraps each layer in the model in Mocklayers/Embeddings.
    It also marks the all layers which contain embeddings (or which contain layers which contain embeddings).
    These need to be simulated/entered during simulation, Otherwise we the simulation does not include the MockEmbedding
    and we do not get Access pattern info to generate intents with.
  """
  embs_downstream = list()  # contains booleans if the recursive calls to children contained embeddings.
  for name, param in module.named_children():  # recursively call children first, to wrap their children.
    embs_downstream.append(recursive_wrap_children(param, simulation_queue))

  if isinstance(module, auto_ada.ada_models.PSEmbedding) and len(embs_downstream) == 0:
    return True

  # wrap children
  for child_id, (name, param) in enumerate(module.named_children()):
    # special case: embeddings are wrapped in MockEmbeddings
    if isinstance(param, auto_ada.ada_models.PSEmbedding):
      me = MockEmbedding(param, name, simulation_queue)
      setattr(module, name, me)

    else:  # everything else is wrapped into a MockLayer
      ml = MockLayer(param, embs_downstream[child_id])
      setattr(module, name, ml)

  # return if a child of this module is an embedding -> this layer/module should be simulated in that case.
  # so that the underlying embedding is reached.
  for child in embs_downstream:
    if child:
      return True

  return False


def strip_kv_from_model(model):
  """
  Stripping the model of any kv-references is required for deep-copying it.
  """
  for module in model.modules():
      if hasattr(module, "kv"):
        setattr(module, "kv", None)
  return model


def add_kv_to_model(model, kv):
  """
  Inverse of strip_kv_from_model.
  """
  for module in model.modules():
    if hasattr(module, "kv"):
      setattr(module, "kv", kv)
  return model


def rgetattr(obj, path):  # recursive getattr: eg. elmo.scalar_mix.gamma
  return functools.reduce(getattr, path.split('.'), obj)


def rsetattr(obj, path, val):  # recursive setattr: eg. elmo.scalar_mix.gamma
  pre, _, post = path.rpartition('.')
  return setattr(rgetattr(obj, pre) if pre else obj, post, val)
