import hydra
import lightning.pytorch as pl
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

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="./", filename="weights_FashionCNN"
    )

    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        max_epochs=cfg.train.epochs,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
