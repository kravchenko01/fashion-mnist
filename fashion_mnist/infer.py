import hydra
import lightning.pytorch as pl
import numpy as np
from data import MyDataModule
from model import FashionCNN
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # считываем с диска модель, загружаем валидационный датасет,
    # предсказываем моделью ответы для этих данных,
    # записываем ответы на диск в .csv файл,
    # выводим в stdout необходимые метрики на этом датасете.
    pl.seed_everything(42)

    dm = MyDataModule(
        data_path=cfg.data.path,
        batch_size=cfg.data.batch_size,
    )
    model = FashionCNN.load_from_checkpoint("./weights_FashionCNN.ckpt", cfg=cfg)

    trainer = pl.Trainer(accelerator="cpu")

    trainer.test(model, datamodule=dm)

    predictions = trainer.predict(model, datamodule=dm)
    np.savetxt("predicted_labels.csv", np.array(predictions).flatten(), delimiter=",")


if __name__ == "__main__":
    main()
