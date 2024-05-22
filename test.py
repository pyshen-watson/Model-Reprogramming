#%%
import pytorch_lightning as pl
from model import DNN6, CNN, SourceModule
pl.seed_everything(42)

#%%
ckpt_path = ".record/SourceModel/C-0/checkpoints/epoch=4-step=980.ckpt"
source_model = CNN(10)
source_module = SourceModule.load_from_checkpoint(ckpt_path, source_model=source_model)
# %%
print(source_model.parameters())
# %%
