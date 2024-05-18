from data import DatasetName, DataModule
dm = DataModule(DatasetName.CIFAR10)
print(dm.mean, dm.std)