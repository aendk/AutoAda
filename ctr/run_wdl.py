import torch
import time
import os
import torch.nn as nn
import tqdm
import sys
import ctr.criteo_dataset
from ctr.cli import parse_arguments
from auto_ada.ada_models import AdaModel
import auto_ada.adaps_setup
from ctr.models import wdl_hugectr
from auto_ada.ada_dataloader import AdaPSDataLoader
from sklearn.metrics import log_loss
import cProfile

def run_worker(worker_id, args, kv):
    train(worker_id, args, kv)
    kv.barrier()  # wait for all workers to finish
    kv.finalize()

def run_eval(model, test_dl, args, epoch):
    # file name of best performing permutation of this model, e.g. CriteoNetwork_best_model.pt
    best_model_path = type(model.user_model).__name__ + "_best_model.pt"
    loss_sum = 0.0
    loss_sum_bce = 0.0
    bce = nn.BCEWithLogitsLoss()

    with tqdm.tqdm(test_dl, disable=(not args.progress_bar)) as tq, torch.no_grad():
        for batch_id, data in enumerate(tq):
            labels = data['labels'].to(args.device)

            pred = model(last_user_batch_mods(data))

            loss_sum_bce += bce(input=pred, target=labels)

            # regular logloss w/ extra sigmoid beforehand
            sig_pred = torch.sigmoid(pred)
            loss = log_loss(y_true=labels.cpu().data.numpy(), y_pred=sig_pred.cpu().data.numpy().astype("float64"))
            loss_sum += loss

        avg_test_logloss = loss_sum / len(tq)
        avg_test_logloss_bce = loss_sum_bce / len(tq)
        print(f"EVAL: Epoch {epoch} average logloss {avg_test_logloss} , BCE-loss={avg_test_logloss_bce}")

        if not hasattr(args, 'best_loss'):  # piggyback off args
            args.best_loss = float('inf')
        if args.save_model and args.best_loss > avg_test_logloss:
            args.best_loss = avg_test_logloss
            #torch.save(model.state_dict(), best_model_path)

        return avg_test_logloss_bce


def last_user_batch_mods(data):
    return [data["dense_features"], data["sparse_features"]]


def get_keys(data):  # a single gk-function for two layers.
    return data['sparse_features']


def train(worker_id, args, kv):
    print(f"Worker {worker_id} training on {args.device} with {torch.get_num_threads()} threads")
    # Initialize distributed training context.

    kv.begin_setup()  # TODO imo this whole block could be moved to the AdaModel entirely.
    model = args.model
    torch.manual_seed(args.model_seed)
    model.kv_init(kv, 0, args.opt, worker_id == 0, args.intent_ahead > 0)
    model.to(args.device)
    kv.end_setup()
    kv.wait_sync()

    # full replication
    if args.enforce_full_replication > 0:
        print(f"Enforcing full replication: signal intent for keys 0..{args.num_keys} in time [0, {sys.maxsize}]")
        all_keys = torch.arange(args.num_keys)
        for keys in all_keys.split(2**20):
            kv.intent(keys, 0, sys.maxsize)
            time.sleep(3)
    kv.wait_sync()
    kv.wait_sync()

    train_dl, test_dl = ctr.criteo_dataset.init_distributed_dataloaders(rank=worker_id, world_size=args.world_size, batch_size=args.batch_size, num_workers=args.dl_workers, data_root_dir=args.dataset_dir)

    if args.auto_intent:  # automatic intent
        train_dl = AdaPSDataLoader(model, train_dl, args.intent_ahead, last_user_batch_mods_fn=last_user_batch_mods)
        test_dl = AdaPSDataLoader(model, test_dl, args.intent_ahead, last_user_batch_mods_fn=last_user_batch_mods)

    else:  # manual intent by selected by supplying a get_key_tuple (or a list of them) as manual intent functions.
        get_key_tuple_list = list()
        get_key_tuple_list.append((model.user_model.sparse_embedding1, get_keys))  # both layers are fed the same data, so we can reuse the same func.
        get_key_tuple_list.append((model.user_model.wide.wide_linear, get_keys))
        train_dl = AdaPSDataLoader(model, train_dl, args.intent_ahead, ada_lname_get_keys_fn_tuples=get_key_tuple_list, last_user_batch_mods_fn=last_user_batch_mods)
        test_dl = AdaPSDataLoader(model, test_dl, args.intent_ahead, ada_lname_get_keys_fn_tuples=get_key_tuple_list, last_user_batch_mods_fn=last_user_batch_mods)

    loss_fn = nn.BCEWithLogitsLoss()
    kv.barrier()  # workers start epoch at the same time

    total_train_time = 0
    start_time = time.time()
    print(f"Setup done, training of first epoch starting now.")
    profile_please = False
    if worker_id == 0 and profile_please:
        pr = cProfile.Profile()
        pr.enable()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        train_dl.user_dataloader.sampler.set_epoch(epoch)  
        total_train_loss = 0.0
        total_reg_loss = 0.0
        num_batches = -1

        with tqdm.tqdm(train_dl, position=worker_id, disable=(not args.progress_bar)) as tq:
            num_batches = len(tq)
            print(f"Epoch={epoch} actually starting the epoch @ {time.time()-epoch_start}. Number of batches in worker {worker_id}: {num_batches}")
            for batch_id, data in enumerate(tq):
                if batch_id == 0:
                    print(f" Epoch={epoch}, first batch happening {time.time() - epoch_start} after 'Setup done'")

                labels = data['labels']

                pred = model(last_user_batch_mods(data))

                main_loss = loss_fn(input=pred, target=labels)
                reg_loss = model.user_model.get_regularization(args.l2_reg_linear, args.l2_reg_embedding)
                loss = main_loss + reg_loss

                loss.backward()
                # optimizer step is done in the grad_hooks of the backward pass.
                total_train_loss += loss.item()
                total_reg_loss += reg_loss.item()

                tq.set_postfix({'loss': '%.04f' % loss.item(), 'reg_loss': '%.04f' % reg_loss.item()}, refresh=False)


        # synchronize replicas
        print(f"worker {worker_id} finished epoch {epoch+1} in {time.time()-epoch_start}. Avg train loss: {total_train_loss/num_batches}. Avg reg loss: {total_reg_loss/num_batches}", flush=True)
        kv.wait_sync()  #  this triple can be substituted w/ model.end_epoch()
        kv.barrier()
        kv.wait_sync()
        epoch_stop = time.time()
        total_train_time += epoch_stop - epoch_start

        if worker_id == 0 and profile_please:
            pr.disable()
            pr.print_stats(sort='cumtime')
        # Evaluate on only the first worker/device/GPU only.
        if args.eval_model and worker_id == 0:

            model.eval()
            test_loss = run_eval(model, test_dl, args, epoch)

            print(f"All workers finished epoch {epoch+1} (epoch: {epoch_stop-epoch_start:.3f}s, total: {total_train_time:.3f}s). Test loss: {test_loss}", flush=True)

        kv.barrier()

        # maximum time
        if (args.max_runtime != 0 and
                (total_train_time > args.max_runtime or
                 total_train_time + (epoch_stop-epoch_start) > args.max_runtime * 1.05)):
            print(f"Worker {worker_id} stops after epoch {epoch+1} because max. time is reached: {total_train_time}s (+ 1 epoch) > {args.max_runtime}s", flush=True)
            break

    end_time = time.time()
    print(f"All {epoch} epochs done in {end_time-start_time}s")
    kv.barrier()


if __name__ == "__main__":
    args = parse_arguments()


    # the model
    criteo_model = wdl_hugectr.WdlHugeCtr(args.feature_dim, args.embedding_dim)
    args.model = AdaModel(criteo_model)

    # start the necessary processes
    auto_ada.adaps_setup.start_lapse(args.model, run_worker_func=run_worker, args=args)
