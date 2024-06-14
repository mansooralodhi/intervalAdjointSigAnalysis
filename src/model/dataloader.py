from torch.utils import data


class Dataloader(data.DataLoader):
    def __init__(self, dataset,
                 batch_size=1,
                 shuffle=False,
                 sampler=None,
                 batch_sanpler=None,
                 num_worker=0,
                 pin_memory=False,
                 drop_last=False,
                 timeout=0,
                 worker_init_fn=None):
        super(Dataloader, self).__init__(dataset,
                                         batch_size,
                                         shuffle,
                                         sampler,
                                         batch_sampler=batch_sanpler,
                                         num_workers=num_worker,
                                         collate_fn=data.default_collate,
                                         pin_memory=pin_memory,
                                         drop_last=drop_last,
                                         timeout=timeout,
                                         worker_init_fn=worker_init_fn)


if __name__ == "__main__":
    from torchvision.datasets import MNIST
    from src.model.dataset import Dataset

    trainData = MNIST(root='data/mnist', download=True, train=True)
    dataset = Dataset(trainData)
    trainDataloader = Dataloader(dataset)
    print(next(iter(trainDataloader)))  # (x: (batch_size, p), y: (batch_size,))
