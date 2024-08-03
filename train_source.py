import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from pathlib import Path
from src.common import basic_parser
from src.model import SourceWrapper, DNN, VGG, Resnet
from src.dataset import ImageDataModule

torch.set_float32_matmul_precision("high")


def create_dataModule(args):
    return ImageDataModule(
        name=args.dataset,
        root_dir=args.data_dir,
        size=args.src_size,  # Default:112
        num_workers=args.num_workers,  # Default:12
        batch_size=args.batch_size,  # Default:128
    )


def create_trainer(args, exp_name):

    # Ex. epoch=2-step=294-val_loss=92.7337-val_acc=0.4800.ckpt
    ckpt_name = "{epoch}-{step}-{val_loss:.4f}-{val_acc:.4f}"  

    return pl.Trainer(
        accelerator="gpu",
        devices=[args.gpu_id],
        benchmark=True,
        max_steps=args.max_steps,
        fast_dev_run=args.dry_run,
        logger=TensorBoardLogger("lightning_logs", name=exp_name),
        log_every_n_steps=20,
        callbacks=[
            ModelCheckpoint(monitor="val_loss", filename=ckpt_name, save_top_k=1, mode="min"),
            EarlyStopping(monitor="val_loss", patience=args.patience, mode="min"),
        ],
    )


def create_model(args, n_class=10):

    input_size = (1, 3, args.src_size, args.src_size)

    if args.model == "DNN":
        return DNN(
            input_size=input_size,
            width=args.conv_width,
            level=args.level,
            n_class=n_class,
        )

    elif args.model == "CNN":
        return VGG(
            input_size=input_size,
            width=args.conv_width,
            level=args.level,
            group=1,
            n_class=n_class,
        )

    elif args.model == "VGG":
        return VGG(
            input_size=input_size,
            width=args.conv_width,
            level=args.level,
            group=args.group,
            n_class=n_class,
        )
    
    elif args.model == "ResNet":
        return Resnet(
            input_size=input_size,
            width=args.conv_width,
            level=args.level,
            group=args.group,
            n_class=n_class,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")


def create_wrapper(args, exp_name, log_dir, model):

    wrapper = SourceWrapper(
        name=exp_name,
        log_dir=Path(log_dir),  # Ex. lightning_logs/VGG-2x2/version_0
        hp=vars(args),  # The hyperparameters for logging
        lr=args.learning_rate,
        wd=args.weight_decay,
        loss_fn=args.loss_fn # CE or MSE
    )

    # Set the source model in the last because dataclass assigns
    # the backbone before the lightning module's __init__ method.
    return wrapper.set_source_model(model)


if __name__ == "__main__":

    # Get args and set random seed for every dependencies
    args = basic_parser.parse_args()
    pl.seed_everything(args.random_seed, workers=True)

    if args.model in ["CNN", "DNN"]:
        exp_name = f"{args.model}-{args.level}" # Ex. CNN-3
    else:
        exp_name = f"{args.model}-{args.level}x{args.group}" # Ex. VGG-3x2

    # Prepare the dataloader
    dm = create_dataModule(args)
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # Train the source model
    trainer = create_trainer(args, exp_name)
    model = create_model(args)
    wrapper = create_wrapper(args, exp_name, trainer.logger.log_dir, model)
    trainer.fit(wrapper, train_loader, val_loader)
