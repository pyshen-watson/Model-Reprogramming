import pytorch_lightning as pl
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from .utils import DatasetName, load_dataset


class ImageDataset(Dataset):

    def __init__(self, name, root_dir="./data/dataset", split="train"):
        self.image, self.label = load_dataset(name, root_dir, split)
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self, idx):
        return self.transform(self.image[idx]), self.label[idx]

    def __len__(self):
        return len(self.image)


class ImageDataModule(pl.LightningDataModule):

    def __init__(self, name: DatasetName, root_dir="./data/dataset", batch_size=128, num_workers=4):

        self.train_ds = ImageDataset(name, root_dir, "train")
        self.test_ds = ImageDataset(name, root_dir, "test")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_class = self.train_ds.label.max().item() + 1

    def X_dataloader(self, ds, shuffle=False):
        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)

    def train_dataloader(self):
        return self.X_dataloader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self.X_dataloader(self.test_ds)

    def test_dataloader(self):
        return self.X_dataloader(self.test_ds)
