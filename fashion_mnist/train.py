import hydra
import lightning.pytorch as pl
import torch
from data import MyDataModule
from model import FashionCNN
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Загружаем данные, обучаем модель и сохраняем на диск
    pl.seed_everything(42)

    dm = MyDataModule(
        data_path=cfg.data.path,
        batch_size=cfg.data.batch_size,
    )
    model = FashionCNN(cfg)

    loggers = [
        pl.loggers.MLFlowLogger(
            experiment_name=cfg.mlflow.experiment_name,
            tracking_uri=cfg.mlflow.uri,
        ),
    ]

    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     dirpath="./", filename="weights_FashionCNN"
    # )

    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        max_epochs=cfg.train.epochs,
        enable_checkpointing=False,
        logger=loggers,
        # callbacks=[checkpoint_callback],
    )

    trainer.fit(model, datamodule=dm)

    X = torch.randn(cfg.data.batch_size, 1, 28, 28)
    torch.onnx.export(model, X, "./model.onnx")


if __name__ == "__main__":
    main()
