import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from pathlib import Path
from src.common import basic_parser
from src.model import VGG, ReprogrammingWrapper
from src.dataset import DsName, ImageDataModule

torch.set_float32_matmul_precision("high")


def parse_args():

    # For data and weight
    basic_parser.add_argument( "--target_ds", type=str, default="cifar10", choices=["cifar10", "stl10", "svhn", "imagenet10"], help="The name of target dataset, it can be cifar10, stl10 or svhn (default: cifar10)", )
    basic_parser.add_argument( "--tgt_size", type=int, default=64, help="The size of target data (default: 32)", )
    basic_parser.add_argument( "--weight_path", type=str, help="Path to the source model weight file (required)", )

    # For transformation layer
    basic_parser.add_argument( "-F", "--fc_layer", action="store_true", help="Use fully connected layer (default: False)", )
    basic_parser.add_argument( "-V", "--visual_prompt", action="store_true", help="Use visual prompt", )
    return basic_parser.parse_args()


def create_dataModule(args):
    return ImageDataModule(
        name=DsName.value_to_member(args.target_ds),
        root_dir=args.data_dir,
        size=args.tgt_size,  # Default:64
        num_workers=args.num_workers,  # Default:12
        batch_size=args.batch_size,  # Default:128
    )


def create_trainer(args, exp_name):

    # Ex. epoch=2-step=294-val_loss=92.7337-val_acc=0.4800.ckpt
    ckpt_name = "{epoch}-{step}-{val_loss:.4f}-{val_acc:.4f}"  

    return pl.Trainer(
        accelerator="gpu",
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

    if args.model == "VGG":
        return VGG(
            input_size=input_size,
            n_class=n_class,
            group=args.pooling,
            level=args.level,
            width=args.conv_width,
        ).load(args.weight_path)
    else:
        raise ValueError(f"Unknown model: {args.model}")


def create_wrapper(args, exp_name, log_dir, model):

    wrapper = ReprogrammingWrapper(
        name=exp_name,
        log_dir=Path(log_dir),
        hp=vars(args),
        lr=args.learning_rate,
        wd=args.weight_decay,
        source_size=args.src_size,
        target_size=args.tgt_size,
        vp=args.visual_prompt,
        # fc=args.fc_layer
    )

    # Set the source model in the last because dataclass assigns
    # the backbone before the lightning module's __init__ method.
    return wrapper.set_source_model(model)


if __name__ == "__main__":

    # Get args and set random seed for every dependencies
    args = parse_args()
    pl.seed_everything(args.random_seed, workers=True)
    exp_name = f"{args.model}-{args.level}x{args.pooling}(source)"  # Ex. VGG-3x2(source)
    exp_name += "_VP" if args.visual_prompt else ""
    # exp_name += "_FC" if args.fc_layer else ""


    # Prepare the dataloader
    dm = create_dataModule(args)
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # Train the source model
    trainer = create_trainer(args, exp_name)
    model = create_model(args)
    wrapper = create_wrapper(args, exp_name, trainer.logger.log_dir, model)

    if args.visual_prompt:  # or args.fc_layer:
        trainer.fit(wrapper, train_loader, val_loader)
    else:  # Baseline
        trainer.test(wrapper, val_loader)
