import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data import DatasetName, ImageDataModule
from model import DNN6, CNN6, SourceModule

dm = ImageDataModule(DatasetName.CIFAR10, num_workers=16, batch_size=128)

source_model = CNN6(dm.n_class)
source_module = SourceModule(source_model, lr=1e-3)

ckpt_callback = ModelCheckpoint( monitor="val_loss", save_top_k=1, mode="min", verbose=True)
estp_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min", verbose=True)

trainer = pl.Trainer(
    max_epochs=200,
    benchmark=True,
    callbacks=[ckpt_callback, estp_callback],
)

trainer.fit(source_module, dm.train_dataloader(), dm.val_dataloader())
