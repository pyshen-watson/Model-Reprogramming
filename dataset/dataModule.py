import pytorch_lightning as pl
from enum import Enum
from pathlib import Path
from colorama import Fore, Style
from torchvision.datasets import CIFAR10, SVHN, STL10, ImageFolder
from torchvision import transforms as T
from torch.utils.data import DataLoader
from functools import partial


class DatasetName(Enum):
    CIFAR10 = "cifar10"
    SVHN = "svhn"
    STL10 = "stl10"
    IMAGENET_F10 = "imagenet-f10"
    # IMAGENET_R10 = "imagenet-r10"

    @staticmethod
    def member():
        return [ds.name for ds in DatasetName]


def get_transform(size=32):
    return T.Compose(
        [
            T.ToTensor(),
            T.Resize((size, size), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class ImageDataModule(pl.LightningDataModule):

    def __init__( self, name: DatasetName, root_dir="./dataset/data", size=32, batch_size=128, num_workers=12, ):
        super().__init__()

        transform = get_transform(size)
        self.create_loader = partial( DataLoader, num_workers=num_workers, batch_size=batch_size )

        if name == DatasetName.CIFAR10:
            self.train_ds = CIFAR10(root_dir, train=True, transform=transform, download=True)
            self.test_ds = CIFAR10(root_dir, train=False, transform=transform, download=True)

        elif name == DatasetName.SVHN:
            self.train_ds = SVHN(root_dir, split="train", transform=transform, download=True)
            self.test_ds = SVHN(root_dir, split="test", transform=transform, download=True)

        elif name == DatasetName.STL10:
            self.train_ds = STL10(root_dir, split="train", transform=transform, download=True)
            self.test_ds = STL10(root_dir, split="test", transform=transform, download=True)

        elif name == DatasetName.IMAGENET_F10:
            self.train_ds = ImageFolder(Path(root_dir) / "train", transform=transform)
            self.test_ds = ImageFolder(Path(root_dir) / "test", transform=transform)

        else:
            err_msg = f"✗ Invalid name: dataset name must be one of {DatasetName.member()}"
            raise ValueError(Fore.RED + err_msg + Style.RESET_ALL)

    def train_dataloader(self):
        return self.create_loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self.create_loader(self.test_ds)

    def test_dataloader(self):
        return self.create_loader(self.test_ds)
