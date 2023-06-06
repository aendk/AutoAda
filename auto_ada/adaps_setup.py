import os
import torch
import adaps
from threading import Thread
from torch import cuda
from torch.multiprocessing import Process, set_start_method

from signal import signal, SIGINT

from auto_ada.optimizer import PSAdagrad


def init_scheduler(dummy, args):
    os.environ['DMLC_NUM_SERVER'] = str(args.world_size)
    os.environ['DMLC_ROLE'] = 'scheduler'
    os.environ['DMLC_PS_ROOT_URI'] = args.root_uri
    os.environ['DMLC_PS_ROOT_PORT'] = args.root_port
    print("running scheduler")
    adaps.scheduler(args.num_keys, args.workers_per_node)


def init_node(local_rank, args, run_worker_func):
    """Start up a Lapse node (server + multiple worker threads)"""
    os.environ['DMLC_NUM_SERVER'] = str(args.world_size)
    os.environ['DMLC_ROLE'] = 'server'
    os.environ['DMLC_PS_ROOT_URI'] = args.root_uri
    os.environ['DMLC_PS_ROOT_PORT'] = args.root_port

    adaps.setup(args.num_keys, args.workers_per_node, use_techniques=args.sys_techniques, num_channels=args.communication_channels)
    server = adaps.Server(args.lens)
    rank = server.my_rank()
    print(f"Started server with rank {rank} with {args.lens.shape} keys and {args.lens.sum()} total values.")

    # make sure all servers are set up
    server.barrier()

    threads = []
    for w in range(args.workers_per_node):
        args.local_id = local_rank * args.workers_per_node + w
        worker_id = rank * args.workers_per_node + w

        # assign training device to worker
        if args.cuda:
            local_worker_id = local_rank * args.workers_per_node + w
            if args.device_ids:
                device_id = args.device_ids[local_worker_id]
            else:
                device_id = local_worker_id % cuda.device_count()
            args.device = torch.device("cuda:" + str(device_id))
        else:
            args.device = torch.device("cpu")
        # run worker
        t = Thread(target=run_worker_func, args=(worker_id, args, adaps.Worker(w, server)))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # shutdown lapse node
    server.shutdown()


def kill_processes(signal_received, frame):
    # Kills all started lapse processes
    print('\nSIGINT or CTRL-C detected. Shutting down all processes and exiting..')
    for p in processes:
        p.kill()
    exit(0)


processes = []


def start_lapse(model, run_worker_func, args):

    # read environment variables when running with tracker
    if args.tracker:
        lapse_env = {'DMLC_NUM_SERVER', 'DMLC_ROLE', 'DMLC_PS_ROOT_URI', 'DMLC_PS_ROOT_PORT'}
        assert os.environ.keys() >= lapse_env, f'Missing Lapse environment variables. Check {lapse_env} are set.'
        args.role = os.environ['DMLC_ROLE']
        args.root_uri = os.environ['DMLC_PS_ROOT_URI']
        args.root_port = os.environ['DMLC_PS_ROOT_PORT']
        args.world_size = int(os.environ['DMLC_NUM_SERVER'])
        if args.role == 'scheduler':
            args.scheduler = True
            args.nodes = 0
        else:
            args.scheduler = False

    print(args)

    # setup optimizer and model
    args.opt = PSAdagrad(
        lr=args.learning_rate,
        initial_accumulator_value=args.initial_accumulator_value,
        eps=args.epsilon,
    )
    args.model = model

    # calculate parameter lens
    args.lens = args.model.lens()
    args.num_keys = len(args.lens)

    # catch interrupt (to shut down lapse processes)
    signal(SIGINT, kill_processes)

    # "spawn" required for cuda training
    set_start_method('spawn', force=True)

    # launch lapse scheduler
    if args.scheduler:
        p = Process(target=init_scheduler, args=(0, args))
        p.start()
        processes.append(p)

    # launch lapse processes
    for local_rank in range(args.nodes):
        p = Process(target=init_node, args=(local_rank, args, run_worker_func))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
