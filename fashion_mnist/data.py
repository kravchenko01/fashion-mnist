from typing import Optional

import lightning.pytorch as pl
import torch
from torchvision import datasets, transforms


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_path = data_path
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = datasets.FashionMNIST(
            root=self.data_path,
            train=True,
            download=False,
            transform=transforms.Compose([transforms.ToTensor()]),
        )
        self.test_dataset = datasets.FashionMNIST(
            root=self.data_path,
            train=False,
            download=False,
            transform=transforms.Compose([transforms.ToTensor()]),
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=3,
            shuffle=True,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=3,
            shuffle=False,
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        return self.test_dataloader()
