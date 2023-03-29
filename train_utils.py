""" Utils functions definitions for train.py. """

import os
import sys
import random

import git
import torch
import numpy as np
from loguru import logger


def setup_loguru_level(level: str="INFO") -> None:
    """ Set loguru.logger level.
    
    Arguments
    ---------
    level:
        One of DEBUG->INFO->SUCCESS->WARNING->ERROR->CRITICAL.
    """

    logger.remove()
    logger.add(sys.stderr, level=level)
    getattr(logger, level.lower())(f"Set {level} level for logger output.")

def analyze_git_repo(remote_branch: str=None) -> None:
    """ Check if current code has commit on remote.
    
    Arguments
    ---------
    remote_branch:
        Name of branch to compare last local commit.
    """

    repo = git.Repo(".")
    local_branch = repo.active_branch.name
    remote_branch = local_branch if remote_branch is None else remote_branch

    uncommited_files = not "clear" in repo.git.status()
    unpushed_commits = bool(repo.git.log(f"origin/{remote_branch}..{local_branch}"))

    if uncommited_files:
        logger.critical("Save new changes!")
        sys.exit()
    elif unpushed_commits:
        logger.critical("Push all commits!")
        sys.exit()

def set_seed(seed: int) -> None:
    """ Set random seed to reproducibility as wandb [1].

    Arguments
    ---------
    seed:
        Random seed.

    [1]: https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/
    How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    """

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set to {seed}.")
