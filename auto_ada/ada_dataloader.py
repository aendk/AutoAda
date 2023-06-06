from collections import deque
import queue
import copy
import torch
from torch.utils.data import DataLoader
import auto_ada.utils as utils
from typing import Any, Callable, Iterable, TypeVar, Generic, Sequence, List, Optional, Union


def create_intent_sim_model(model, simulation_input_queue):
  """
  This function takes a model and replaces every nn.layer with either a MockLayer or PSEmbedding.
  This model is then used to simulate the forward pass with minimal overhead.
  """

  utils.recursive_wrap_children(model, simulation_input_queue)
  return model


class AdaPSDataLoader:

  def __init__(self, u_model: torch.nn.Module, u_dl: DataLoader, intent_ahead: int, ada_lname_get_keys_fn_tuples: Optional = None, last_user_batch_mods_fn: Optional = None):
    self.user_model = None  # only needed for auto-get-keys, set in prep_auto_gk
    self.kv = u_model.kv
    self.batch_queue = deque()
    self.simulation_input_queue = queue.Queue()  # simulated forward pass publishes input data of each relevant layer in this queue.
    self.distance_between_intent_and_use = intent_ahead
    self.clock = 0  # needed to associate batches to adaps-clocks to send intents signals with correct timestamps.
    self.ada_lname_get_keys_fn_tuples = None

    if ada_lname_get_keys_fn_tuples is not None:
      # prep-work for manual get key functions.
      self.prep_manual_gk(ada_lname_get_keys_fn_tuples)
      self.automatic_get_keys = False
    else:
      self.automatic_get_keys = True
      self.prep_auto_gk(u_model)

    self.last_user_batch_mods_fn = last_user_batch_mods_fn
    if last_user_batch_mods_fn is None:
      print("last_user_batch_mods_fn is not supplied. This means that Intent Signaling does only work if the batch is not modified in any shape or form after collate_fn. "
            "Except are moving operations like .to(device)")

    self.user_dataloader = u_dl

  def prep_auto_gk(self, u_model):
    u_model = utils.strip_kv_from_model(u_model)  # strip kv to deepcopy the model, as kv cannot be copied
    self.user_model = copy.deepcopy(u_model)
    utils.add_kv_to_model(u_model, self.kv)  # add kv to original model again.

    self.user_model = create_intent_sim_model(self.user_model.user_model, self.simulation_input_queue)  # layers are substituted with MockLayers, to make simulation as light as possible.
    print("AUTOMATIC GetKeys Function in AdaDL")

  def prep_manual_gk(self, ada_lname_get_keys_fn_tuples):

    #  Manual GK-(name, func)-tuples can be transmitted either as a single tuple, or as a list of tuples.
    if isinstance(ada_lname_get_keys_fn_tuples, tuple):
      tl = list()
      tl.append(ada_lname_get_keys_fn_tuples)
      ada_lname_get_keys_fn_tuples = tl

    # optimization: change first item of Tuple (layers) to their key_offsets, if this has not been done yet.
    if not isinstance(ada_lname_get_keys_fn_tuples[0][0], int):
      for id, kf_tuple in enumerate(ada_lname_get_keys_fn_tuples):
        (layer, func) = kf_tuple
        ada_lname_get_keys_fn_tuples[id] = (layer.key_offset, func)

    self.ada_lname_get_keys_fn_tuples = ada_lname_get_keys_fn_tuples
    print("MANUAL GetKeys Function in AdaDL")

  def __iter__(self):

    self.iter = iter(self.user_dataloader)
    self.clock = self.kv.current_clock()

    # preload queue
    for _ in range(self.distance_between_intent_and_use):
      self.load_batch_into_queue()
    return self

  def simulate_batch_execution(self, batch):

    self.user_model.train()
    if self.simulation_input_queue.qsize() != 0:
      raise ValueError(f"AdaDL batch simulation queue needs to be empty in between clocks, but qsize={self.simulation_input_queue.qsize()}")

    if self.last_user_batch_mods_fn is not None:  # replicate all last minute changes the user does in their normal train-process.
      batch = self.last_user_batch_mods_fn(batch)

    try:
      _ = self.user_model(batch)
      # currently, the simulation step/ simulated forward pass errors out.
      # this is caused because we cannot escape operations like torch.cat() with our current approach.
      # We present modifications to remedy this in the thesis.
    except Exception:
      pass

  def dispatch_intents(self, keys, key_offset, intent_clock):
    keys = keys.flatten() + key_offset
    self.kv.intent(keys.cpu().long(), intent_clock)

  def generate_intents(self, batch, intent_clock):
    """
    Manual intents: calls the getkey-function + dispatches intents.
    Auto intents: Simulates the forward-pass of this batch using the mock-model;
    in which each layer does no computation, but it can report what input is received.
    For Embeddings, this directly correlates to the parameter access.
    """
    if not self.automatic_get_keys:
      for gk_tuple in self.ada_lname_get_keys_fn_tuples:
        self.dispatch_intents(gk_tuple[1](batch), gk_tuple[0], intent_clock)

    else:
      self.simulate_batch_execution(batch)

      # reading the queue
      while not self.simulation_input_queue.empty():
        layer_input = self.simulation_input_queue.get()
        self.dispatch_intents(layer_input[1], layer_input[0], intent_clock)

  def load_batch_into_queue(self):
    self.clock += 1  
    try:
      new_batch = next(self.iter)

      self.generate_intents(new_batch, self.clock)
      self.batch_queue.append(new_batch)
    except StopIteration:
      return

  def __next__(self):
    self.load_batch_into_queue()

    try:
      return self.batch_queue.popleft()

    except IndexError:
      raise StopIteration

  def __len__(self) -> int:
    return len(self.user_dataloader)
