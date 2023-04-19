import sys
import statistics
import torch
from torch import nn
import ctr.criteo_dataset as criteoData
from torch.nn.parallel.distributed import DistributedDataParallel
import traceback

class CriteoNetwork(nn.Module):

    def __init__(self, feature_dimension, embed_dim,
                 batch_size):
        super(CriteoNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()  # what benefit was there to define relu yourselves?
        # alternatively use torch.nn.functional as F + F.relu(whatever_layer)

        self.w1 = nn.Linear(13, 256)
        # torch.nn.init.normal_(self.w1, std=0.01)  # TODO HET-initialization did not really work, needed/ required?
        self.w2 = nn.Linear(256, 256)
        # torch.nn.init.normal_(self.w2, std=0.01)
        self.w3 = nn.Linear(256, 256)
        # torch.nn.init.normal_(self.w3, std=0.01)
        self.w4 = nn.Linear(256 + 26 * embed_dim, 1)
        # torch.nn.init.normal_(self.w4, std=0.01)

        # TODO seems like the embeddings are not set up identically in C++
        # self.embeddings = nn.Embedding(batch_size,26, embedding_dim=embed_dim)
        self.embeddings = nn.Embedding(num_embeddings=feature_dimension, embedding_dim=embed_dim)


    # def forward(self, dense_features, sparse_features):
    def forward(self, data):

        #traceback.print_tb(sys.exc_info()[2], sys.stdout) // TODO WIP to print out stacktrace
        # dense
        dense_in = data[0]
        r1 = self.w1(dense_in)
        relu1 = self.relu(r1)
        relu2 = self.relu(self.w2(relu1))
        y3 = self.w3(relu2)
        # sparse
        embs = self.embeddings(data[1])
        # print(f"embs done={embs}")
        self.flat_embeddings = self.flatten(embs)
        # concat +  dense
        y4 = torch.cat((self.flat_embeddings, y3), 1)
        y = self.w4(y4)
        return y

def last_user_batch_mods(samples):
    data = [samples["dense_features"], samples["sparse_features"]]
    return data

def train(rank, device, epoch, dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch_id, samples in enumerate(dataloader):
        # print(f"train:: batch{batch} of epoch{epoch} to now start training w/ it")
        # print(f"trainloop:: batch={samples} ")
        # TODO this has to be done in the collate_fn
        # or we have the possibility like this to make it easy for the user and us.
        data = last_user_batch_mods(samples)
        for elem in data:
            elem.to(device)


        # Compute prediction error
        pred = model(data)
        unsqueezed_labels = torch.unsqueeze(samples['labels'], dim=1)

        loss = loss_fn(pred, unsqueezed_labels.float())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_id % 100 == 0:
            loss, current = loss.item(), batch_id * len(samples)
            print(f"n={rank} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(rank, device, epoch, dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    batch_accuracies = []
    with torch.no_grad():
        for samples in dataloader:

            data = last_user_batch_mods(samples)
            for elem in data:
                elem.to(device)
            pred = model(data)

            unsqueezed_labels = torch.unsqueeze(samples['labels'], dim=1)
            test_loss += loss_fn(pred, unsqueezed_labels.float()).item()
            correct += (pred.argmax(1) == unsqueezed_labels.float()).type(torch.float).sum().item()
            predictions = pred > 0.5
            batch_accuracies.append((unsqueezed_labels == predictions).sum().item() / unsqueezed_labels.size(0))
    test_loss /= num_batches

    print(f"n={rank} Avg test loss: {test_loss:>8f}, avg accuracy: {statistics.mean(batch_accuracies)}\n")

def setup_loss_opt(model, learning_rate):
    loss_fn = nn.BCEWithLogitsLoss()  # thx to https://discuss.pytorch.org/t/crossentropy-in-pytorch-getting-target-1-out-of-bounds/71575
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    return loss_fn, optimizer

def setup(device, feature_dim, embed_dim, batch_size, learning_rate, data_root_dir=None):
    train_dataloader, test_dataloader = criteoData.init_DataLoaders(batch_size, num_workers=2, data_root_dir=data_root_dir)
    model = CriteoNetwork(feature_dim, embed_dim, batch_size).to(device)

    loss_fn, optimizer = setup_loss_opt(model, learning_rate)

    return model, loss_fn, optimizer, train_dataloader, test_dataloader


def setup_DDP_model(rank, world_size, device, feature_dim, embed_dim, batch_size, learning_rate, data_root_dir=None):

    train_dataloader, test_dataloader = criteoData.init_DataLoaders_DDP(rank, world_size, batch_size, num_workers=2, data_root_dir=data_root_dir)
    model = CriteoNetwork(feature_dim, embed_dim, batch_size).to(device)
    model = DistributedDataParallel(model)

    loss_fn, optimizer = setup_loss_opt(model, learning_rate)

    return model, loss_fn, optimizer, train_dataloader, test_dataloader


def main():  # test-params
    batch_size = 128
    batch_size = 4  
    feature_dimension = 33762577  
    embed_dim = 8  
    learning_rate = 0.01  
    model, loss_fn, optimizer, train_dl, test_dl = setup("cpu", feature_dimension, embed_dim, batch_size, learning_rate)

    for epoch in range(5):
        train("norank", "cpu", epoch, train_dl, model, loss_fn, optimizer)
        test("norank", "cpu", epoch, test_dl, model, loss_fn)


if __name__ == "__main__":
    main()
