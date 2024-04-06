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


if __name__ == "__main__":
    from torchvision.datasets import MNIST

    trainData = MNIST(root='data/mnist', download=True, train=True)
    dataset = Dataset(trainData)
    print(dataset[0]) # (x: (batch_size, p), y: (batch_size,))
