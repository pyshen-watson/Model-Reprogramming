import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from pathlib import Path
from argparse import ArgumentParser
from model import SourceWrapper, CNN, VGG
from dataset import DsName, ImageDataModule

torch.set_float32_matmul_precision('high')

def parse_args():
    parser = ArgumentParser()
    parser.add_argument( "-r", "--random_seed", type=int, default=42, help="The random seed for all dependencies (default: 42)")

    # For data
    parser.add_argument("--root_dir", type=str, default="../data/ImageNet10", help="The path to the dataset (default: ../data/ImageNet10)")
    parser.add_argument("--src_size", type=int, default=224, help="The input size of source dataset (default: 224)")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of workers for data loading (default: 12)") 
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training (default: 128)") 

    # For model
    parser.add_argument( "-M", "--model", type=str, choices=["CNN", "VGG"], help="The type of model (required)", ) 
    parser.add_argument( "-L", "--level", type=int, default=1, help="The number of conv layers per block (default: 1)", ) 
    parser.add_argument( "-P", "--pooling", type=int, default=3, help="The number of pooling, i.e. the number of blocks (default: 3)", ) 
    parser.add_argument( "-W", "--conv_width", type=int, default=128, help="The number of channel of convolutional layer (default: 128)", ) 
    parser.add_argument( "-S", "--std_weight", type=float, default=2.0**0.5, help="The standard deviation of initialization (default: √2)", ) 

    # For optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate (default: 1e-3)" ) 
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay (default: 1e-3)", ) 

    # For trainer
    parser.add_argument( "-m", "--max_steps", type=int, default=5000, help="Maximum number of steps for training (default: 5000)", )
    parser.add_argument( "-d", "--dry_run", action="store_true", help="Perform a dry run without training", ) 
    parser.add_argument( "-p", "--patience", type=int, default=5, help="The patience epochs for early stop. (default: 5)", )

    return parser.parse_args()

def create_dataModule(args):
    return ImageDataModule(
        name=DsName.IMAGENET10,
        root_dir=args.root_dir,
        size=args.src_size, # Default:224
        num_workers=args.num_workers, # Default:12
        batch_size=args.batch_size, # Default:128
    )

def create_model(args, n_class=10):
    
    if args.model == "CNN":
        return CNN(n_class=n_class, 
                   n_layer=args.level, 
                   input_size=args.src_size, 
                   conv_width=args.conv_width,
                   init_std=args.std_weight)
    
    elif args.model == "VGG":
        return VGG(n_class=n_class,
                   level=args.level,
                   pooling=args.pooling,
                   init_std=args.std_weight)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
def create_wrapper(args, model, log_dir):

    # The lightning module model
    wrapper = SourceWrapper(
        name=f"SRC_{args.model}{args.level}", # Ex. SRC_CNN3
        log_dir=Path(log_dir), # Ex. lightning_logs/SRC_CNN3/version_0
        hp=vars(args), # The hyperparameters for logging
        lr=args.learning_rate, 
        wd=args.weight_decay)
    
    # Set the source model in the last because dataclass assigns 
    # the backbone before the lightning module's __init__ method.
    return wrapper.set_source_model(model)

def create_trainer(args):

    exp_name = f"SRC_{args.model}{args.level}" # Ex. SRC_CNN3
    ckpt_name = '{epoch}-{step}-{val_loss:.4f}-{val_acc:.4f}' # Ex. epoch=2-step=294-val_loss=92.7337-val_acc=0.4800.ckpt

    return pl.Trainer(
        accelerator="gpu",
        benchmark=True,
        max_steps=args.max_steps,
        fast_dev_run=args.dry_run,
        logger=TensorBoardLogger("lightning_logs", name=exp_name), 
        log_every_n_steps=20,
        callbacks=[ 
            ModelCheckpoint(monitor="val_loss", filename=ckpt_name, save_top_k=1, mode="min"),
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
    model = create_model(args)
    wrapped_model = create_wrapper(args, model, trainer.logger.log_dir)
    trainer.fit(wrapped_model, train_loader, val_loader)
