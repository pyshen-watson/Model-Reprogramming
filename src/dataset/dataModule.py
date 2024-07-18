import pytorch_lightning as pl
from enum import Enum
from pathlib import Path
from functools import partial
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10, SVHN, STL10, ImageFolder
from dataclasses import dataclass
from ..common import log_fail


class DsName(Enum):
    CIFAR10 = "cifar10"
    SVHN = "svhn"
    STL10 = "stl10"
    IMAGENET10 = "imagenet10"

    @staticmethod
    def member_list():
        return ', '.join([ds.name for ds in DsName])
 

    @staticmethod
    def value_to_member(value:str):
        try:
            return DsName._value2member_map_[value]
        except:
            err_msg = f"✗ Invalid name: dataset name must be one of {DsName.member_list()}."
            log_fail(err_msg)


@dataclass
class ImageDataModule(pl.LightningDataModule):

    name: DsName
    root_dir: str = "./dataset/data"
    size: int = 32
    batch_size: int = 128
    num_workers: int = 12

    def __post_init__(self):
        
        self.train_T = T.Compose([
                T.ToTensor(),
                T.RandomHorizontalFlip(),
                T.RandomRotation(15),
                T.Resize((self.size, self.size), antialias=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        self.test_T =  T.Compose([
                T.ToTensor(),
                T.Resize((self.size, self.size), antialias=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        self.prepare_data()
        self.create_loader = partial(DataLoader, num_workers=self.num_workers, batch_size=self.batch_size)

    def prepare_data(self):

        if self.name == DsName.CIFAR10:
            cifar10 = partial(CIFAR10, root=self.root_dir, download=True)
            self.train_ds = cifar10(train=True, transform=self.train_T)
            self.test_ds  = cifar10(train=False, transform=self.test_T)

        elif self.name == DsName.SVHN:
            svhn = partial(SVHN, root=self.root_dir, download=True)
            self.train_ds = svhn(split="train", transform=self.train_T)
            self.test_ds  = svhn(split="test", transform=self.test_T)

        elif self.name == DsName.STL10:
            stl10 = partial(STL10, root=self.root_dir, download=True)
            self.train_ds = stl10(split="train", transform=self.train_T)
            self.test_ds  = stl10(split="test", transform=self.test_T)

        elif self.name == DsName.IMAGENET10:
            root_dir = Path(self.root_dir)
            self.train_ds = ImageFolder(root_dir / "train", transform=self.train_T)
            self.test_ds = ImageFolder(root_dir / "test", transform=self.test_T)

        else:
            err_msg = f"✗ Invalid name: dataset name must be one of {DsName.member_list()}."
            log_fail(err_msg)

    def train_dataloader(self):
        return self.create_loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self.create_loader(self.test_ds)

    def test_dataloader(self):
        return self.create_loader(self.test_ds)
