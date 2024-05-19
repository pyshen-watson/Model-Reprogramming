import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data import DatasetName, ImageDataModule
from model import DNN6, CNN6, SourceModule
from argparse import ArgumentParser

pl.seed_everything(0)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", "-m", type=str)
    parser.add_argument("--batch_size", "-b", type=int, default=128)
    parser.add_argument("--weight_decay", "-w", type=float, default=0.0)
    parser.add_argument("--lr", "-l", type=float, default=1e-3)
    parser.add_argument("--num_workers", "-n", type=int, default=16)
    parser.add_argument("--dry_run", "-d", action="store_true")
    return parser.parse_args()


def get_model(model_name, n_class):
    if model_name == "DNN6":
        return DNN6(n_class)
    elif model_name == "CNN6":
        return CNN6(n_class)
    else:
        raise ValueError(f"Model {model_name} not found")


if __name__ == "__main__":

    args = parse_args()
    dm = ImageDataModule( DatasetName.CIFAR10, num_workers=args.num_workers, batch_size=args.batch_size )
    model = get_model(args.model, dm.n_class)
    source_module = SourceModule(model, lr=args.lr, weight_decay=args.weight_decay)

    ckpt_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    estp_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")

    trainer = pl.Trainer(
        max_epochs=200,
        benchmark=True,
        callbacks=[ckpt_callback, estp_callback],
        fast_dev_run=args.dry_run,
    )

    trainer.fit(source_module, dm.train_dataloader(), dm.val_dataloader())
