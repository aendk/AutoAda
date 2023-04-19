from argparse import ArgumentParser
from torch import cuda


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', default="../../data/ctr-criteo-subset/", type=str, help='Set path to dataset here.')
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
    distribution.add_argument('-t', '--tracker', default=False, action='store_true', help='use this option if running with adaps tracker scripts')
    # model options
    model = parser.add_argument_group('model')
    model.add_argument('-ed', '--embedding_dim', default=128, type=int, help='dimension of the initial node embedding')
    model.add_argument('-fd', '--feature_dim', default=33762588, type=int, help='feature dimension of the embeddings, default: max index of ctr-subset')
    model.add_argument('-mt', '--model_type', default="hugectr_wdl", type=str, choices=["hugectr_wdl", "faulty_het", "original_wide_deep"], help='Model Type; HugeCTR-WDL the working default')

    # optimizer parameters
    optimizer = parser.add_argument_group('optimizer')
    optimizer.add_argument('-eps', '--epsilon', default=1e-10, type=float, help='adagrad epsilon')
    optimizer.add_argument('-iac', '--initial_accumulator_value', default=0.0, type=float, help='adagrad initial accumulator value')
    optimizer.add_argument('-lr', '--learning_rate', default=0.0001, type=float, help='adagrad learning rate; learning rate > 0.001 can degrade performance (Deep&Cross, Wang et al.)')
    # training options
    training = parser.add_argument_group('training')
    training.add_argument('-b', '--batch_size', default=128, type=int, help='per-device batch size')
    training.add_argument('-e', '--epochs', default=1, type=int, help='number of epochs to run')  # 1 for debug.
    training.add_argument('-dlw', '--dl_workers', default=4, type=int, help='number of workers per data loader')
    training.add_argument('-sm', '--save_model', default=False, type=bool, help='set if you want to save the model with the best accuracy.')
    training.add_argument('-em', '--eval_model', default=True, type=bool, help='set if you want to run an evaluation/validation step after each epoch')
    optimizer.add_argument('-l2rl', '--l2_reg_linear', default=1e-5, type=float, help='L2 regularization for linear/wide part')
    optimizer.add_argument('-l2re', '--l2_reg_embedding', default=1e-5, type=float, help='L2 regularization for embedding part')
    training.add_argument('--model_seed', default=23, type=int, help='set model seed')
    training.add_argument('--max_runtime', default=0, type=int, help='maximum run time, in seconds')
    # AdaPS options
    adaps = parser.add_argument_group('AdaPS')
    adaps.add_argument('--sys.techniques', default="all", dest="sys_techniques", type=str, help='which management techniques to use')
    adaps.add_argument('--sys.channels', default=4, dest="communication_channels", type=int, help='number of communication channels')
    adaps.add_argument('-ia', '--intent_ahead', default=50, type=int, help='number of training batches to pre load and intent; 0 disables intent signaling')
    adaps.add_argument('--enforce_full_replication', default=0, type=int, help='whether to enforce full replication parameter management')
    # PyTorch integration
    adaps.add_argument('-ai', '--auto_intent', default=False, type=bool, help=' Enable to automatically generate intents via simulation. Disable to rely on manual intent functions. ')

    args = parser.parse_args()
    if args.world_size is None:
        args.world_size = args.nodes
        args.scheduler = True
    return args


if __name__ == "__main__":
    print(parse_arguments())
