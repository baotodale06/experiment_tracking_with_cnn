import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


from lightning_module import LeNetLitModule
from datamodule import MNISTDataModule

import os

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # Reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # Logger
    wandb_logger = WandbLogger(
        project=cfg.logging.project,
        entity=cfg.logging.entity,
        log_model=cfg.logging.log_model,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    ckpt_dir = os.path.join(os.getcwd(), "checkpoints")

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor=cfg.training.checkpoint.monitor,
        mode=cfg.training.checkpoint.mode,
        save_top_k=cfg.training.checkpoint.save_top_k,
        filename=cfg.training.checkpoint.filename,
    )

    # Data
    datamodule = MNISTDataModule(cfg)

    # Model
    model = LeNetLitModule(cfg)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        logger=wandb_logger,
        deterministic=True,
        callbacks=[checkpoint_cb],
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
