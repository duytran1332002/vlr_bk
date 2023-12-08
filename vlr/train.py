import sys
import os
sys.path.append(os.getcwd())
import hydra
import logging

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from vlr.utils.avg_ckpts import ensemble
from datamodule.data_module import DataModule
from vlr.driver.lightning import ModelModule


@hydra.main(version_base="1.3.2", config_path="configs", config_name="config")
def main(cfg):
    seed_everything(42, workers=True)
    cfg.devices = torch.cuda.device_count()

    checkpoint = ModelCheckpoint(
        monitor="monitoring_step",
        mode="max",
        dirpath=os.path.join(cfg.exp_dir, cfg.exp_name) if cfg.exp_dir else None,
        save_last=True,
        every_n_train_steps = 100,
        filename="{epoch}-{monitoring_step:.2f}",
        save_top_k=10,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint, lr_monitor]

    # Set modules and trainer
    modelmodule = ModelModule(cfg)
    datamodule = DataModule(cfg)
    trainer = Trainer(
        **cfg.trainer,
        logger=WandbLogger(name=cfg.exp_name, project="1st_200h_visual_pretrained"),
        callbacks=callbacks,
        accelerator="gpu"
    )
    try:
        if cfg.ckpt_path != '':
            trainer.fit(model=modelmodule, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
        else:
            trainer.fit(model=modelmodule, datamodule=datamodule)
        ensemble(cfg)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt by user detected. Stopping...")
        trainer.save_checkpoint("last.ckpt")
        ensemble(cfg)


if __name__ == "__main__":
    main()
