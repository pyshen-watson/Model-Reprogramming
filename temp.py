from colorama import Fore, Style
from dataset import ImageDataModule, DatasetName
from time import perf_counter


for name in DatasetName:

    start = perf_counter()
    root_dir = "/home/pyshen/DataCenter/ImageNet/IN_first10/" if name == DatasetName.IMAGENET_F10 else "dataset/data"
    size = 224 if name == DatasetName.IMAGENET_F10 else 32
    dm = ImageDataModule(name, root_dir, size)
    images, labels = next(iter(dm.train_dataloader()))
    end = perf_counter()

    msg = f'✓ Read samples from {name} in {(end-start)/1000:.3f} seconds, {images.shape=} {labels.shape=}'
    
    print(Fore.GREEN + msg + Style.RESET_ALL)