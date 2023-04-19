import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# DDP
from torch.utils.data import DistributedSampler
from typing import Tuple
import numpy as np


def load_npy(path, start, stop):
    # nparr = np.load(path, mmap_mode='c') #Used for faster startup during testing.
    nparr = np.load(path)
    subset = nparr[int(start * len(nparr)): int(stop * len(nparr))]
    return subset


class CTRCriteoDataset(Dataset):
    def __init__(self, train, root_dir, sampled_labels_p, sampled_dense_feats_p, sampled_sparse_feats_p):
        """
        Args:
            root_dir (string): Directory containing .npy-Files.
            train (bool): decides which subset will be used for training; First 90% as train-set, rest for testing.
            sampled_labels_p (string): file name of labels
            sampled_dense_feats_p (string): file name of dense features
            sampled_sparse_feats_p (string): file name of sparse features
        """
        # same train/test split as Wang et al.(Deep & Cross): first 6/7 for training,
        # last 1/7 randomly 50:50 for test and validation.
        self.train = train
        start = 0
        stop = 1.00
        if train:
            stop = 6 / 7
        else:
            start = 6 / 7

        self.sampled_labels = load_npy(root_dir + sampled_labels_p, start, stop)
        self.sampled_dense_feats = load_npy(root_dir + sampled_dense_feats_p, start, stop)
        self.sampled_sparse_feats = load_npy(root_dir + sampled_sparse_feats_p, start, stop)

        if not train:  # select random subset for test-set.
            # get 50% of existing subset for training
            new_len = int(self.sampled_labels.shape[0] / 2)
            np.random.seed(423)  # randomly selected seed
            self.sampled_labels = self.sampled_labels[np.random.randint(self.sampled_labels.shape[0], size=new_len)]
            np.random.seed(423)  # reset seed
            self.sampled_dense_feats = self.sampled_dense_feats[
                                       np.random.randint(self.sampled_dense_feats.shape[0], size=new_len), :]
            np.random.seed(423)  # reset seed
            self.sampled_sparse_feats = self.sampled_sparse_feats[
                                        np.random.randint(self.sampled_sparse_feats.shape[0], size=new_len), :]

    def __len__(self):
        return len(self.sampled_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        samples = {"labels": self.sampled_labels[idx],
                   "dense_features": self.sampled_dense_feats[idx].astype("float32"),
                   "sparse_features": self.sampled_sparse_feats[idx]}

        if type(samples["labels"]) == np.int64:  # Het-subset has labels as scalars, not arrays.
            samples["labels"] = np.atleast_1d(samples["labels"])

        for key in samples:
            samples[key] = torch.from_numpy(samples[key])

        samples["labels"] = samples["labels"].long().float()
        samples["sparse_features"] = samples["sparse_features"].long()
        # around batch 2450 w/ 128 samples each, there is an 'inf' leading to NaN in training.
        samples["dense_features"] = torch.nan_to_num(samples["dense_features"])

        return samples


def load_data(root_dir):
    # default assumption: we use the criteo-subset from het.
    sampled_labels_p = "sampled_labels.npy"
    sampled_dense_feats_p = "sampled_dense_feats.npy"
    sampled_sparse_feats_p = "sampled_sparse_feats.npy"
    if "subset" in root_dir:
        print("subset is loaded only")

    elif "criteo" in root_dir and "kaggle" in root_dir:
        print(f"loading criteo-kaggle from '{root_dir}'; assuming sparse .npy file is already continuous")
        sampled_labels_p = "criteo_kaggle_labels.npy"
        sampled_dense_feats_p = "criteo_kaggle_dense.npy"
        sampled_sparse_feats_p = "criteo_kaggle_sparse_continuous.npy"

    else:
        print(f"dataset root directory={root_dir}. \n either needs to contain 'subset' or 'kaggle' and 'criteo'")
        exit(1)

    training_data = CTRCriteoDataset(train=True,
                                     root_dir=root_dir,
                                     sampled_labels_p=sampled_labels_p,
                                     sampled_dense_feats_p=sampled_dense_feats_p,
                                     sampled_sparse_feats_p=sampled_sparse_feats_p)
    test_data = CTRCriteoDataset(train=False,
                                 root_dir=root_dir,
                                 sampled_labels_p=sampled_labels_p,
                                 sampled_dense_feats_p=sampled_dense_feats_p,
                                 sampled_sparse_feats_p=sampled_sparse_feats_p)

    return training_data, test_data


def init_dataloaders(data_root_dir, batch_size, num_workers=2):
    training_data, test_data = load_data(data_root_dir)
    prefetch_factor = 3
    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=num_workers,
                                  prefetch_factor=prefetch_factor)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader


def init_distributed_dataloaders(rank: int,
                         world_size: int,
                         data_root_dir: str,
                         batch_size: int,
                         num_workers: int) -> Tuple[DataLoader, DataLoader]:
    training_data, test_data = load_data(data_root_dir)
    # Create data loaders.
    sampler = DistributedSampler(training_data,
                                 num_replicas=world_size,  # Number of workers
                                 rank=rank,  # similar to lapse worker-id
                                 shuffle=True,  # Shuffling is done by Sampler
                                 seed=42)

    # Create data loaders.
    train_dataloader = DataLoader(training_data,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  sampler=sampler,
                                  pin_memory=True)

    test_dataloader = DataLoader(test_data,
                                 batch_size=16384)  # batch_size optimized for speed, as no training takes place.

    return train_dataloader, test_dataloader
