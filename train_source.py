import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from pathlib import Path
from argparse import ArgumentParser
from model import Backbone, SourceModule
from dataset import DsName, ImageDataModule


def parse_args():
    parser = ArgumentParser()
    parser.add_argument( "-R", "--random_seed", type=int, default=42, help="The random seed for all dependencies (default: 42)", )

    # For data
    parser.add_argument( "-r", "--root_dir", type=str, default="dataset/data", help="The path to the dataset (default: dataste/data)", )
    parser.add_argument( "-s", "--src_size", type=int, default=224, help="The input size of source dataset (default: 224)", )
    parser.add_argument( "-N", "--num_workers", type=int, default=12, help="Number of workers for data loading (default: 12)", ) 
    parser.add_argument( "-b", "--batch_size", type=int, default=128, help="Batch size for training (default: 128)", ) 

    # For model
    parser.add_argument( "-l", "--level", type=int, default=1, help="The number of conv layers per block (required)", ) 
    parser.add_argument( "-P", "--pooling", type=int, default=3, help="The number of pooling, i.e. the number of blocks (required)", ) 
    parser.add_argument( "-n", "--exp_name", type=str, default="untitled", help="Name of the experiment (default: 'untitled')", ) 

    # For optimizer
    parser.add_argument( "-L", "--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)" ) 
    parser.add_argument( "-w", "--weight_decay", type=float, default=1e-3, help="Weight decay (default: 1e-3)", ) 

    # For trainer
    parser.add_argument( "-d", "--dry_run", action="store_true", help="Perform a dry run without training", ) 
    parser.add_argument( "-M", "--max_steps", type=int, default=5000, help="Maximum number of steps for training (default: 5000)", )
    parser.add_argument( "-p", "--patience", type=int, default=5, help="The patience epochs for early stop. (default: 5)", )

    return parser.parse_args()

def create_dataModule(args):
    return ImageDataModule(
        name=DsName.IMAGENET10,
        root_dir=args.root_dir,
        size=args.src_size,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

def create_model(args, log_dir, n_class=10):

    # Create the backbone
    level = args.level
    assert level >= 1 and level <= 3, "The level is recommended between 1~3"
    num_blocks = [level] * args.pooling # 
    backbone = Backbone(n_class, num_blocks).set_name(args.exp_name)

    # Create the lightning module and set the backbone
    model = SourceModule(
        log_dir=Path(log_dir), 
        hp=vars(args), 
        lr=args.lr, 
        wd=args.weight_decay)

    return model.set_source_model(backbone)

def create_trainer(args):

    return pl.Trainer(
        accelerator="gpu",
        benchmark=True,
        max_steps=args.max_steps,
        fast_dev_run=args.dry_run,
        logger=TensorBoardLogger("lightning_logs", name=args.exp_name), 
        log_every_n_steps=20,
        callbacks=[ 
            ModelCheckpoint(monitor="val_loss", filename='{epoch}-{step}-{val_loss:.4f}-{val_acc:.4f}', save_top_k=1, mode="min"),
            EarlyStopping(monitor="val_loss", patience=args.patience, mode="min")]
    )


if __name__ == "__main__":

    # Get args and set random seed for every dependencies
    args = parse_args()
    pl.seed_everything(args.random_seed, workers=True)

    # Prepare the dataloader
    dm = create_dataModule(args)
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # Train the source model
    trainer = create_trainer(args)
    model = create_model(args, trainer.logger.log_dir)
    trainer.fit(model, train_loader, val_loader)
