from argparse import ArgumentParser
from torch import cuda


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', default="ogbn-arxiv", type=str, help='name of ogb node property prediction dataset')
    parser.add_argument('-dr', '--data_root', default="dataset", type=str, help='relative path under which the dataset is or should be stored')
    parser.add_argument('--toggle_progress_bar', default=True, dest='progress_bar', action='store_false', help='disable progress bar')
    # cuda options
    cudaoptions = parser.add_argument_group('cuda')
    cudaoptions.add_argument('--no_cuda', default=cuda.is_available(), dest='cuda', action='store_false', help='disable CUDA training')
    cudaoptions.add_argument('-di', '--device_ids', nargs='+', type=int, help='IDs of cuda devices for each worker')
    # distribution options
    distribution = parser.add_argument_group('distribution')
    distribution.add_argument('-be', '--backend', default="gloo", type=str, choices=['gloo', 'nccl'], help='backend')
    distribution.add_argument('-rs', '--scheduler', default=False, action='store_true', help='run scheduler')
    distribution.add_argument('-ru', '--root_uri', default='127.0.0.1', type=str, help='adress of the scheduler node')
    distribution.add_argument('-rp', '--root_port', default='9091', type=str, help='port of the scheduler node')
    distribution.add_argument('-nn', '--nodes', default=1, type=int, help='number of local server nodes to create')
    distribution.add_argument('-nr', '--node_ranking', default=0, type=int, help='ranking among the nodes')
    distribution.add_argument('-nw', '--workers_per_node', default=1, type=int, help='number of worker threads per node')
    distribution.add_argument('-ws', '--world_size', type=int, help='total number of worker nodes')
    distribution.add_argument('-t', '--tracker', default=False, action='store_true', help='use this option if running with AdaPM tracker scripts')
    distribution.add_argument('-nk', '--external_num_keys', default=None, type=int, help='manually pass the number of keys so the scheduler does not have to load the dataset')
    # model options
    model = parser.add_argument_group('model')
    model.add_argument('-ed', '--embedding_dim', default=500, type=int, help='dimension of the initial node embedding')
    model.add_argument('-nsn', '--num_sampling_neighbors', default=16, type=int, help='number of neighbors to sample')
    # optimizer parameters
    optimizer = parser.add_argument_group('optimizer')
    optimizer.add_argument('-eps', '--epsilon', default=1e-10, type=float, help='adagrad epsilon')
    optimizer.add_argument('-iac', '--initial_accumulator_value', default=0.0, type=float, help='adagrad initial accumulator value')
    optimizer.add_argument('-lr', '--learning_rate', default=0.01, type=float, help='adagrad learning rate')
    optimizer.add_argument('-rlop', '--reduce_lr_on_plateau', default=False, action='store_true', help='wether to reduce the lr on plateau')
    optimizer.add_argument('-rlopf', '--rlop_factor', default=0.1, type=float, help='factor by which the learning rate will be reduced')
    optimizer.add_argument('-rlopp', '--rlop_patience', default=2, type=int, help='number of epochs with no improvement after which learning rate will be reduced')
    optimizer.add_argument('-rlopc', '--rlop_cooldown', default=0, type=int, help='number of epochs to wait before resuming normal operation after lr has been reduced')
    optimizer.add_argument('-rlopt', '--rlop_threshold', default=1e-4, type=float, help='threshold for measuring the new optimum, to only focus on significant changes')
    optimizer.add_argument('-rlopm', '--rlop_min_lr', default=0, type=float, help='lower bound on the learning rate')
    optimizer.add_argument('-rlope', '--rlop_eps', default=1e-8, type=float, help='minimal decay applied to lr')
    # training options
    training = parser.add_argument_group('training')
    training.add_argument('-b', '--batch_size', default=32, type=int, help='per-device batch size')
    training.add_argument('-e', '--epochs', default=1, type=int, help='number of epochs to run')
    training.add_argument('--model_seed', default=23, type=int, help='set model seed')
    training.add_argument('--max_runtime', default=0, type=int, help='maximum run time, in seconds')
    # Parameter manager options
    adapm = parser.add_argument_group('AdaPM')
    adapm.add_argument('--sys.techniques', default="all", dest="sys_techniques", type=str, help='which management techniques to use')
    adapm.add_argument('--sys.channels', default=4, dest="communication_channels", type=int, help='number of communication channels')
    adapm.add_argument('-ia', '--intent_ahead', default=30, type=int, help='number of training batches to pre load and intent; 0 disables intent signaling')
    adapm.add_argument('--enforce_full_replication', default=0, type=int, help='whether to enforce full replication parameter management')
    # PyTorch integration
    adapm.add_argument('-ai', '--auto_intent', default=False, type=bool, help=' Enable to automatically generate intents via simulation. Disable to rely on manual intent functions. ')
    args = parser.parse_args()
    if args.world_size is None:
        args.world_size = args.nodes
        args.scheduler = True
    return args


if __name__ == "__main__":
    print(parse_arguments())
