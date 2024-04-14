import jax.numpy as jnp
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        y = self.dataset.targets[item]
        x = self.dataset.data[item].ravel() / 255.0
        return x, y

    def feature_intervals(self):
        X = self.dataset.data.numpy()
        X = X.reshape(X.shape[0], jnp.prod(jnp.array(X.shape[1:])))
        lowerBounds = X.min(axis=0)
        upperBounds = X.max(axis=0)
        ivals = jnp.vstack((lowerBounds, upperBounds))
        ivals = jnp.moveaxis(ivals, 0, 1)
        return ivals


if __name__ == "__main__":
    from torchvision.datasets import MNIST

    trainData = MNIST(root='data/mnist', download=True, train=True)
    dataset = Dataset(trainData)
    print(dataset[0]) # (x: (batch_size, p), y: (batch_size,))
