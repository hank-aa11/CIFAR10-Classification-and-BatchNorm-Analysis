"""Data loaders for CIFAR-10 (Part 2)."""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.datasets as datasets


class PartialDataset(Dataset):
    """Wrap a dataset and limit it to ``n_items`` for fast debugging."""
    def __init__(self, dataset, n_items=10):
        self.dataset = dataset
        self.n_items = min(n_items, len(dataset))

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return self.n_items


def get_cifar_loader(root='./data/', batch_size=128, train=True,
                     shuffle=True, num_workers=4, n_items=-1):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    tf = transforms.Compose([transforms.ToTensor(), normalize])
    dataset = datasets.CIFAR10(root=root, train=train, download=True,
                               transform=tf)
    if n_items > 0:
        dataset = PartialDataset(dataset, n_items)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


if __name__ == '__main__':
    loader = get_cifar_loader(root='./data/')
    for X, y in loader:
        img = np.transpose(X[0].numpy(), [1, 2, 0]) * 0.5 + 0.5
        plt.imshow(img); plt.savefig('sample.png')
        print('shape', X.shape, 'label', y[0].item())
        break