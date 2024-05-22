from argparse import ArgumentParser
from data import DatasetName, ImageDataModule
from model import DNN, CNN, ReprogrammingModule

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

pl.seed_everything(98)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument( "-n", "--exp_name", type=str, default="untitled", help="Name of the experiment (default: 'untitled')", ) 
    parser.add_argument( "-m", "--model", type=str, help="Type of model to use: 'DNN' or 'CNN' (required)", ) 
    parser.add_argument( "-b", "--batch_size", type=int, default=128, help="Batch size for training (default: 128)", ) 
    parser.add_argument( "-L", "--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)" ) 
    parser.add_argument( "-N", "--num_workers", type=int, default=16, help="Number of workers for data loading (default: 16)", ) 
    parser.add_argument( "-d", "--dry_run", action="store_true", help="Perform a dry run without training (default: False)", ) 
    parser.add_argument( "-e", "--max_epochs", type=int, default=200, help="Maximum number of epochs for training (default: 10)", )
    parser.add_argument( "-p", "--weight_path", type=str, help="Path to the source model weight file (required)", )
    parser.add_argument( "-l", "--num_layers", type=int, default=6, help="Number of layers for DNN model (default: 6)", )
    parser.add_argument( "-v", "--visual_prompt", action="store_true", help="Use visual prompt (default: False)", )
    parser.add_argument( "-f", "--fc_layer", action="store_true", help="Use fully connected layer (default: False)", )
    return parser.parse_args()

def get_data_module(args):
    return ImageDataModule(
        name=DatasetName.SVHN,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
    
def get_backbone(args, n_class=10):
    assert args.model in ["DNN", "CNN"], f"Model {args.model} not found"
    model_dict = {"DNN": DNN, "CNN": CNN}
    return model_dict[args.model](n_class, args.num_layers, width=2048).set_name(args.exp_name).load(args.weight_path)

def get_module(args, backbone):
    return ReprogrammingModule(
        source_model=backbone, 
        lr=args.lr,
        visual_prompt=args.visual_prompt,
        fc_layer=args.fc_layer,
    )

def get_trainer(args):

    return pl.Trainer(
        accelerator="gpu", 
        devices=[1],
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

    args = parse_args()
    data_module = get_data_module(args)
    backbone = get_backbone(args, data_module.n_class)
    rpm_module = get_module(args, backbone)
    trainer = get_trainer(args)

    trainer.fit(
        rpm_module, 
        data_module.train_dataloader(), 
        data_module.val_dataloader(),
    )
    # trainer.test(
    #     rpm_module, 
    #     data_module.test_dataloader(),  
    # )