import torch
from pathlib import Path
from torchvision.datasets import CIFAR10, SVHN, STL10

def load_dataset(name:str, root_dir='./data/dataset'):

    Path(root_dir).mkdir(parents=True, exist_ok=True)

    if name == 'cifar10':

        train_ds = CIFAR10(root=root_dir, train=True,download=True)
        train_data = torch.tensor(train_ds.data)
        train_label = torch.tensor(train_ds.targets)

        test_ds = CIFAR10(root=root_dir, train=False,download=True)
        test_data = torch.tensor(test_ds.data)
        test_label = torch.tensor(test_ds.targets)
        
        n_class = len(train_ds.classes)

    elif name == 'svhn':

        train_ds = SVHN(root=root_dir, split='train', download=True)
        train_data = torch.tensor(train_ds.data).permute(0, 2, 3, 1)
        train_label = torch.tensor(train_ds.labels)

        test_ds = SVHN(root=root_dir, split='test', download=True)
        test_data = torch.tensor(test_ds.data).permute(0, 2, 3, 1)
        test_label = torch.tensor(test_ds.labels)
        
        n_class = train_ds.labels.max() + 1
        
    elif name == 'stl10':

        train_ds = STL10(root=root_dir, split='train', download=True)
        train_data = torch.tensor(train_ds.data).permute(0, 2, 3, 1)
        train_label = torch.tensor(train_ds.labels)

        test_ds = STL10(root=root_dir, split='test', download=True)
        test_data = torch.tensor(test_ds.data).permute(0, 2, 3, 1)
        test_label = torch.tensor(test_ds.labels)
        n_class = train_ds.labels.max() + 1

    else:
        raise ValueError('Invalid name: dataset name must be one of cifar10, svhn, stl10')
        
    return (train_data, train_label), (test_data, test_label), n_class