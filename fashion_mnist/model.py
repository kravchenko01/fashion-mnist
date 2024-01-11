from typing import Any

import lightning.pytorch as pl
import torch
import torchmetrics
from omegaconf import DictConfig


class FashionCNN(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(FashionCNN, self).__init__()

        self.cfg = cfg

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )

        self.fc1 = torch.nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.drop = torch.nn.Dropout1d(0.25)
        self.fc2 = torch.nn.Linear(in_features=600, out_features=120)
        self.fc3 = torch.nn.Linear(
            in_features=120, out_features=self.cfg.model.num_classes
        )

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.cfg.model.num_classes
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        images, labels = batch
        preds = self(images)
        loss = self.loss_fn(preds, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        images, labels = batch
        preds = self(images)
        self.acc.update(preds, labels)
        return {"test_acc": self.acc}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        images, labels = batch
        outputs = self(images)
        return torch.max(outputs, 1)[1]

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.train.learning_rate)
        return optimizer

    def on_test_epoch_end(self):
        self.log("test_acc", self.acc.compute())
        self.acc.reset()
