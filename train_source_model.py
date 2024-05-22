from argparse import ArgumentParser
from data import DatasetName, ImageDataModule
from model import DNN, CNN, SourceModule

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

pl.seed_everything(43)


def get_args():
    parser = ArgumentParser()
    parser.add_argument( "-n", "--exp_name", type=str, default="untitled", help="Name of the experiment (default: 'untitled')", ) 
    parser.add_argument( "-m", "--model", type=str, help="Type of model to use: 'DNN' or 'CNN' (required)", ) 
    parser.add_argument( "-b", "--batch_size", type=int, default=256, help="Batch size for training (default: 256)", ) 
    parser.add_argument( "-w", "--weight_decay", type=float, default=0.0, help="Weight decay (default: 0.0)", ) 
    parser.add_argument( "-L", "--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)" ) 
    parser.add_argument( "-N", "--num_workers", type=int, default=12, help="Number of workers for data loading (default: 12)", ) 
    parser.add_argument( "-d", "--dry_run", action="store_true", help="Perform a dry run without training (default: False)", ) 
    parser.add_argument( "-e", "--max_epochs", type=int, default=200, help="Maximum number of epochs for training (default: 200)", )
    parser.add_argument( "-l", "--num_layers", type=int, default=6, help="Number of layers for DNN model (default: 6)", )
    return parser.parse_args()


def get_data_module(args):
    return ImageDataModule(
        name=DatasetName.CIFAR10,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )


def get_backbone(args, n_class=10):
    assert args.model in ["DNN", "CNN"], f"Model {args.model} not found"
    model_dict = {"DNN": DNN, "CNN": CNN}
    return model_dict[args.model](n_class, args.num_layers, width=2048).set_name(args.exp_name)


def get_module(args, backbone):
    return SourceModule(source_model=backbone, lr=args.lr, weight_decay=args.weight_decay)


def get_trainer(args):

    return pl.Trainer(
        accelerator="gpu", 
        max_epochs=args.max_epochs,
        benchmark=True,
        fast_dev_run=args.dry_run,
        logger=TensorBoardLogger("lightning_logs", name=args.exp_name),
        callbacks=[
            ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min"),
            EarlyStopping(monitor="val_loss", patience=3, mode="min"),
        ]
    )


if __name__ == "__main__":

    args = get_args()
    data_module = get_data_module(args)
    backbone = get_backbone(args, data_module.n_class)
    src_module = get_module(args, backbone)
    trainer = get_trainer(args)
    
    trainer.fit(
        src_module, 
        data_module.train_dataloader(), 
        data_module.val_dataloader(),
    )
