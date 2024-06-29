from argparse import ArgumentParser
from dataset import DatasetName, ImageDataModule
from model import VGG, SourceModule

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

pl.seed_everything(42, workers=True)


def get_args():
    parser = ArgumentParser()
    parser.add_argument( "-n", "--exp_name", type=str, default="untitled", help="Name of the experiment (default: 'untitled')", ) 
    parser.add_argument( "-m", "--model", type=int, default=16, help="The number of layers 13, 16 or 19 (required)", ) 
    parser.add_argument( "-b", "--batch_size", type=int, default=128, help="Batch size for training (default: 128)", ) 
    parser.add_argument( "-w", "--weight_decay", type=float, default=0.001, help="Weight decay (default: 0.001)", ) 
    parser.add_argument( "-L", "--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)" ) 
    parser.add_argument( "-N", "--num_workers", type=int, default=12, help="Number of workers for data loading (default: 12)", ) 
    parser.add_argument( "-d", "--dry_run", action="store_true", help="Perform a dry run without training (default: False)", ) 
    parser.add_argument( "-e", "--max_epochs", type=int, default=70, help="Maximum number of epochs for training (default: 50)", )
    parser.add_argument( "-l", "--num_layers", type=int, default=6, help="Number of layers for DNN model (default: 6)", )
    parser.add_argument( "-r", "--root_dir", type=str, default="dataste/data", help="The path to the dataset (default: dataste/data)", )
    parser.add_argument( "-s", "--src_size", type=int, default=32, help="The input size of source dataset (default: 32)", )
    return parser.parse_args()


def get_data_module(args):
    return ImageDataModule(
        name=DatasetName.IMAGENET10,
        root_dir=args.root_dir,
        size=args.src_size,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )


def get_backbone(args, n_class=10):
    if args.model == 13:
        return VGG(input_size=args.src_size, n_class=n_class, num_blocks=[2, 2, 2, 2, 2]).set_name(args.exp_name)
    elif args.model == 16:
        return VGG(input_size=args.src_size, n_class=n_class, num_blocks=[2, 2, 3, 3, 3]).set_name(args.exp_name)
    elif args.model == 19:
        return VGG(input_size=args.src_size, n_class=n_class, num_blocks=[2, 2, 4, 4, 4]).set_name(args.exp_name)
    else:
        raise ValueError("Invalid model number")


def get_module(args, backbone):
    return SourceModule(source_model=backbone, lr=args.lr, weight_decay=args.weight_decay)


def get_trainer(args):

    return pl.Trainer(
        accelerator="gpu",
        devices=[0,1],
        max_epochs=args.max_epochs,
        benchmark=True,
        fast_dev_run=args.dry_run,
        logger=TensorBoardLogger("lightning_logs", name=args.exp_name), #Ming-Yu: MY_logs; default "lightning_logs"
        check_val_every_n_epoch=2, # Ming-Yu: accelerate
        log_every_n_steps=20,
        callbacks=[
            ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min"),
            EarlyStopping(monitor="val_loss", patience=70, mode="min"), # Ming-Yu: 70; default pat=3
        ]
    )


if __name__ == "__main__":

    args = get_args()
    data_module = get_data_module(args)
    backbone = get_backbone(args, 10)
    # src_module = get_module(args, backbone)
    src_module = SourceModule.load_from_checkpoint('lightning_logs/IN10-VGG16-SRC/version_1/checkpoints/epoch=15-step=784.ckpt', source_model=backbone, lr=args.lr, weight_decay=args.weight_decay)
    trainer = get_trainer(args)
    
    trainer.fit(
        src_module, 
        data_module.train_dataloader(), 
        data_module.val_dataloader(),
    )
