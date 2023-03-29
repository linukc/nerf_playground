""" Training script. """

from datetime import datetime
from os import makedirs, mkdir
from os.path import join as osp

import wandb
import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from train_utils import setup_loguru_level, analyze_git_repo, set_seed
from model.nerf_trainer import NerfTrainer


def run_experiment(cfg: DictConfig) -> None:
    """ Train script entrypoint.

    Arguments
    ---------
    cfg:
        Configuration object.
    """

    if cfg.wandb.use:
        wandb.init(project=cfg.wandb.project)
    exp_name = datetime.now().strftime("exp_%d%m%Y_%H:%M:%S")\
        if not cfg.wandb.use else wandb.run.name

    if cfg.wandb.use:
        config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
        wandb.run.config.update(config)
        wandb.config["exp_name"] = exp_name
        logger.info(f"Start wandb logging: {wandb.run.name}")
    else:
        logger.warning("Skip wandb logging!")

    makedirs(osp(cfg.training.exp_folder, exp_name))
    mkdir(osp(cfg.training.exp_folder, exp_name, "checkpoints"))
    OmegaConf.save(cfg, osp(cfg.training.exp_folder, exp_name, "config.yaml"))

    trainer = NerfTrainer(cfg)
    trainer.run(exp_name)

    if cfg.wandb.use:
        wandb.finish()
    logger.success("Finish training.")

@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig=None) -> None:
    """ Entrypoint function. 
    
    Arguments
    ---------
    cfg:
        Configuration object.
    """

    setup_loguru_level(cfg.logger.level)

    if cfg.git.consistency:
        analyze_git_repo()
    else:
        logger.warning("Ignore uncommited changes and unpushed git commits!")

    set_seed(cfg.training.seed)
    run_experiment(cfg)

if __name__ == "__main__":
    main()
