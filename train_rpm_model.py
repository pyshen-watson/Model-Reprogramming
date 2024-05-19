import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data import DatasetName, ImageDataModule
from model import DNN6, CNN6, ReprogrammingModule
from argparse import ArgumentParser


pl.seed_everything(0)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", "-m", type=str)
    parser.add_argument("--source_model_path", "-s", type=str)
    parser.add_argument("--batch_size", "-b", type=int, default=128)
    parser.add_argument("--lr", "-l", type=float, default=1e-3)
    parser.add_argument("--num_workers", "-n", type=int, default=16)
    parser.add_argument("--dry_run", "-d", action="store_true")
    return parser.parse_args()

def get_model(model_name, weight_path, n_class):
    if model_name == "DNN6":
        return DNN6(n_class).load(weight_path)
    elif model_name == "CNN6":
        return CNN6(n_class).load(weight_path)
    else:
        raise ValueError(f"Model {model_name} not found")


if __name__ == "__main__":

    args = parse_args()
    dm = ImageDataModule( DatasetName.SVHN, num_workers=args.num_workers, batch_size=args.batch_size )
    model = get_model(args.model, args.source_model_path, dm.n_class)
    RPM_module = ReprogrammingModule(source_model=model, rescale_size=24, source_size=32, lr=1e-3)

    ckpt_callback = ModelCheckpoint( monitor="val_loss", save_top_k=1, mode="min", verbose=True)
    estp_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min", verbose=True)
    

    trainer = pl.Trainer(
        max_epochs=200,
        benchmark=True,
        callbacks=[ckpt_callback, estp_callback],
        fast_dev_run=args.dry_run,
        logger=TensorBoardLogger("lightning_logs", name=model.name),
    )

    trainer.fit(RPM_module, dm.train_dataloader(), dm.val_dataloader())
