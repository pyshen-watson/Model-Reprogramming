import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from enum import Enum
from dataclasses import dataclass
from .dataset import load_dataset


class DatasetName(Enum):
    CIFAR10 = "cifar10"
    SVHN = "svhn"
    STL10 = "stl10"


@dataclass
class DataModule(pl.LightningDataModule):

    name: DatasetName
    root_dir: str = "./data/dataset"
    batch_size: int = 128

    def __post_init__(self):
        self.train_ds, self.test_ds, self.n_class = load_dataset(self.name.value, self.root_dir)
        self.mean = self.train_ds[0].float().mean((0, 1, 2)) / 255.0
        self.std = self.train_ds[0].float().std((0, 1, 2)) / 255.0

    def X_dataloader(self, ds, shuffle=False):
        ds = TensorDataset(ds[0], ds[1])
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def train_dataloader(self):
        return self.X_dataloader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self.X_dataloader(self.test_ds, shuffle=False)

    def test_dataloader(self):
        return self.X_dataloader(self.test_ds, shuffle=False)
