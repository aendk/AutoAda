import dgl
import sys
import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.nn import GraphConv
import time
from cli import parse_arguments
from data import PartitionedOGBDataset
from auto_ada.ada_models import AdaModel
from auto_ada.ada_dataloader import AdaPSDataLoader
from auto_ada.optimizer import PSAdagrad, ReduceLROnPlateau
import auto_ada.adaps_setup
from gcn.layers import EmbedGraphConv
import cProfile

class Model(nn.Module):
    def __init__(self, num_nodes, h_feats, num_classes):
        super(Model, self).__init__()
        self.conv1 = EmbedGraphConv(num_nodes, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
        self.h_feats = h_feats

    def forward(self, data):
        inputs = data[0]
        mfgs = data[1]
        h = self.conv1(mfgs[0], inputs)
        h = F.relu(h)
        h = self.conv2(mfgs[1], h)
        return h


def run_worker(worker_id, args, kv):
    train(worker_id, args, kv)
    kv.barrier()  # wait for all workers to finish
    kv.finalize()


def batch_mod(batch):
    # stripped from original collate_fn
    input_nodes, _, mfgs = batch
    return [input_nodes, mfgs]


def get_keys_gcn(data):
    # the function has to return the relevant key_offset and the data which the layer will see.
    input_nodes, _, _ = data
    return input_nodes


def train(worker_id, args, kv):
    print(f"Worker {worker_id} training on {args.device} with {torch.get_num_threads()} threads")

    scheduler = ReduceLROnPlateau(
        args.opt,
        factor=args.rlop_factor,
        mode="max",
        patience=args.rlop_patience,
        cooldown=args.rlop_cooldown,
        threshold=args.rlop_threshold,
        min_lr=args.rlop_min_lr,
        eps=args.rlop_eps,
    )

    print("loading training split")
    if args.world_size * args.workers_per_node > 1:
        train_ids = args.data.load_split(worker_id)
    else:
        train_ids = args.data.idx_split["train"]
    print("creating dataloaders..")
    sampler = dgl.dataloading.NeighborSampler([args.num_sampling_neighbors, args.num_sampling_neighbors])
    train_dataloader = dgl.dataloading.DataLoader(
        args.data.graph,  # The graph
        train_ids,  # The node IDs to iterate over in minibatches
        sampler,  # The neighbor sampler
        # device=args.device, # Put the sampled MFGs on CPU or GPU
        use_ddp=False,  # Make it work with distributed data parallel
        batch_size=args.batch_size,  # Per-device batch size.
        # The effective batch size is this number times the number of GPUs.
        shuffle=True,  # Whether to shuffle the nodes for every epoch
        drop_last=False,  # Whether to drop the last incomplete batch
        num_workers=0  # Number of sampler processes
    )
    valid_dataloader = dgl.dataloading.DataLoader(
        args.data.graph,
        args.data.idx_split["valid"],
        sampler,
        device=args.device,
        use_ddp=False,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    kv.begin_setup()
    print("init kv-parameters..")
    model = args.model
    torch.manual_seed(args.model_seed)
    model.kv_init(kv, 0, args.opt, worker_id == 0, args.intent_ahead != 0)
    model.to(args.device)
    kv.end_setup()
    kv.wait_sync()
    print("init kv-parameters done.")
    if args.auto_intent:
        train_dataloader = AdaPSDataLoader(args.model, train_dataloader, args.intent_ahead, last_user_batch_mods_fn=batch_mod)
        valid_dataloader = AdaPSDataLoader(args.model, valid_dataloader, args.intent_ahead, last_user_batch_mods_fn=batch_mod)
    else:
        # reference instead of string: makes sense, some  work in jupyter notebooks, where the model is in memory
        # & the direct reference to the embedding will be autocompleted.
        layer_get_key_tuple = (args.model.user_model.conv1.embedding, get_keys_gcn)  # tuple of (layer, input-data-func)
        train_dataloader = AdaPSDataLoader(args.model, train_dataloader, args.intent_ahead, ada_lname_get_keys_fn_tuples=layer_get_key_tuple, last_user_batch_mods_fn=batch_mod)
        valid_dataloader = AdaPSDataLoader(args.model, valid_dataloader, args.intent_ahead, ada_lname_get_keys_fn_tuples=layer_get_key_tuple, last_user_batch_mods_fn=batch_mod)

    # full replication
    if args.enforce_full_replication > 0:
        print(f"Enforcing full replication: signal intent for keys 0..{args.num_keys} in time [0, {sys.maxsize}]")
        all_keys = torch.arange(args.num_keys)
        for keys in all_keys.split(2 ** 20):
            kv.intent(keys, 0, sys.maxsize)
            time.sleep(3)
    kv.wait_sync()
    kv.wait_sync()

    total_train_time = 0
    kv.barrier()
    profile_please = False
    if worker_id == 0 and profile_please:
        pr = cProfile.Profile()
        pr.enable()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()

        with tqdm.tqdm(train_dataloader, position=worker_id, disable=(not args.progress_bar)) as tq:

            for step, batch in enumerate(tq):
                batch = batch_mod(batch)

                mfgs = batch[1]
                labels = mfgs[-1].dstdata['label'].long()
                predictions = model(batch)

                loss = F.cross_entropy(predictions, labels)
                loss.backward()

                tq.set_postfix({'loss': '%.03f' % loss.item()}, refresh=False)

        print(f"worker {worker_id} finished epoch {epoch+1} in {time.time()-epoch_start}.", flush=True)
        # synchronize replicas
        kv.wait_sync()
        kv.barrier()
        kv.wait_sync()
        epoch_stop = time.time()
        total_train_time += epoch_stop - epoch_start

        if worker_id == 0 and profile_please:
            pr.disable()
            pr.print_stats(sort='cumtime')
        model.eval()

        # Evaluate on only the first GPU.
        if worker_id == 0:
            predictions = []
            labels = []
            with tqdm.tqdm(valid_dataloader, disable=(not args.progress_bar)) as tq, torch.no_grad():
                for input_nodes, output_nodes, mfgs in tq:
                    inputs = mfgs[0].srcdata['_ID']
                    labels.append(mfgs[-1].dstdata['label'].long().cpu().numpy())
                    predictions.append(model([inputs, mfgs]).argmax(1).cpu().numpy())
                predictions = np.concatenate(predictions)
                labels = np.concatenate(labels)
                accuracy = sklearn.metrics.accuracy_score(labels, predictions)
                print(
                    f"All workers finished epoch {epoch + 1} (epoch: {epoch_stop - epoch_start:.3f}s, total: {total_train_time:.3f}s). Validation accuracy: {accuracy}",
                    flush=True)
                if args.reduce_lr_on_plateau:
                    scheduler.step(accuracy)

        # synchronize workers
        kv.barrier()

        # maximum time
        if (args.max_runtime != 0 and
                (total_train_time > args.max_runtime or
                 total_train_time + (epoch_stop - epoch_start) > args.max_runtime * 1.05)):
            print(
                f"Worker {worker_id} stops after epoch {epoch + 1} because max. time is reached: {total_train_time}s (+ 1 epoch) > {args.max_runtime}s",
                flush=True)
            break


processes = []
if __name__ == "__main__":
    # run cli
    args = parse_arguments()

    # load the dataset
    if args.nodes == 0 and args.external_num_keys is not None:
        # no need to load the dataset in the scheduler if the number of keys was passed externally
        print(f"skipping the dataset read in the scheduler. manually setting num_keys={args.external_num_keys} instead")
        args.num_keys = args.external_num_keys
    else:
        # setup dataset
        print(f"loading dataset: {args.dataset}")
        args.data = PartitionedOGBDataset(args.dataset, args.world_size * args.workers_per_node, args.data_root)

        # setup optimizer and model
        print(f"create model with {args.data.num_features()} nodes")
        regular_model = Model(args.data.num_features(), args.embedding_dim, args.data.num_classes())
        args.model = AdaModel(regular_model)

        if args.external_num_keys is not None:
            assert args.num_keys == args.external_num_keys

    # start the necessary processes on this node
    auto_ada.adaps_setup.start_lapse(args.model, run_worker_func=run_worker, args=args)
