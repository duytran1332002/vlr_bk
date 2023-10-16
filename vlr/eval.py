import logging
import sys
import os
sys.path.append(os.getcwd())

import hydra
import torch

from pytorch_lightning import Trainer
from vlr.driver.lightning import ModelModule
from vlr.datamodule.data_module import DataModule

@hydra.main(version_base="1.3.2", config_path="configs", config_name="config")
def main(cfg):
    # Set modules and trainer
    modelmodule = ModelModule(cfg)
    datamodule = DataModule(cfg)
    trainer = Trainer(num_nodes=1, devices=1)
    # Training and testing
    modelmodule.model.load_state_dict(torch.load(cfg.pretrained_model_path, map_location=lambda storage, loc: storage))
    trainer.test(model=modelmodule, datamodule=datamodule)


if __name__ == "__main__":
    main()
