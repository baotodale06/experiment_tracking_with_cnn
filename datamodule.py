import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.data_dir = cfg.data.data_dir
        self.batch_size = cfg.data.batch_size
        self.num_workers = cfg.data.num_workers
        self.val_split = cfg.data.val_split

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def prepare_data(self):
        # Download only (called once per node)
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Train / validation split
        full_train = MNIST(
            self.data_dir,
            train=True,
            transform=self.transform
        )

        val_size = int(len(full_train) * self.val_split)
        train_size = len(full_train) - val_size

        self.train_set, self.val_set = random_split(
            full_train,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Test set
        self.test_set = MNIST(
            self.data_dir,
            train=False,
            transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
