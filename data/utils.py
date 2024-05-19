import torch
from enum import Enum
from pathlib import Path
from torchvision.datasets import CIFAR10, SVHN, STL10


class DatasetName(Enum):
    CIFAR10 = "cifar10"
    SVHN = "svhn"
    STL10 = "stl10"


def load_dataset(name: DatasetName, root_dir="./data/dataset", split="train"):

    assert split in ["train", "test"], "split must be one of train, test"
    Path(root_dir).mkdir(parents=True, exist_ok=True)

    if name == DatasetName.CIFAR10:
        ds = CIFAR10(root=root_dir, train=(split == "train"), download=True)
        image = torch.tensor(ds.data).permute(0, 3, 1, 2)
        label = torch.tensor(ds.targets)

    elif name == DatasetName.SVHN:
        ds = SVHN(root=root_dir, split=split, download=True)
        image = torch.tensor(ds.data)
        label = torch.tensor(ds.labels)

    elif name == DatasetName.STL10:
        ds = STL10(root=root_dir, split=split, download=True)
        image = torch.tensor(ds.data)
        label = torch.tensor(ds.labels)

    else:
        raise ValueError(
            "Invalid name: dataset name must be one of cifar10, svhn, stl10"
        )

    return image, label
