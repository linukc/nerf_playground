"""Main training script."""

import sys
from datetime import datetime
from os.path import join as osp
from os import makedirs, mkdir

import torch
import hydra
from loguru import logger
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import wandb

from model.nerf_model import NeRFModel
from datasets.blender import BlenderDataset
from utils import set_seed, repeater, make_objects, analyze
from utils import have_uncommitted_changes, setup_loguru_level

#pylint: disable=too-many-arguments, too-many-locals
def eval_step(cfg, nerf_model, dataset, dataloader, step, exp_name):
    """ Evaluation step.
        Isolate gpu inference into func scope to release memory after with gc.
    """

    tqdm.write(f"[TEST] =======> Step: {step}")
    ckpt_path = osp(cfg.training.exp_folder, exp_name,
                    "checkpoints", f"step_{step}.pth")
    nerf_model.save_state(ckpt_path)

    if cfg.wandb.use:
        artifact = wandb.Artifact(f"model_{step}", type='model')
        artifact.add_file(ckpt_path)
        wandb.run.log_artifact(artifact)

    gt_pixels, gt_depth = [], []
    pred_pixels, pred_depth, pred_disp, pred_accmap = [], [], [], []

    with torch.no_grad():

        for test_batch in dataloader:
            test_rays = test_batch["ray"].to(cfg.training.device)
            gt_pixels.append(test_batch["gt_pixel"].cpu())
            gt_depth.append(test_batch["gt_depth"].cpu())

            coarse_bundle, fine_bundle = nerf_model(test_rays)
            if not fine_bundle:
                bundle = coarse_bundle
            else:
                bundle = fine_bundle

            pred_pixels.append(bundle['rgb_map'].detach().cpu())
            pred_depth.append(bundle['depth_map'].detach().cpu().unsqueeze(1))
            pred_disp.append(bundle['disp_map'].detach().cpu().unsqueeze(1))
            pred_accmap.append(bundle['acc_map'].detach().cpu().unsqueeze(1))

        analyze(gt_pixels, pred_pixels, gt_depth, pred_depth, pred_disp, pred_accmap,
            image_size=dataset.image_size, use_wandb=cfg.wandb.use,
            image_num=dataset.test_num / cfg.dataset.test_each,
            path=osp(cfg.training.exp_folder, exp_name), step=step)

def run_experiment(cfg: DictConfig) -> None: #pylint: disable=too-many-statements
    """Train script entrypoint.

    Parameters
    ----------
    cfg:
        Configuration object for training procedure.
    """
    exp_name = datetime.now().strftime("exp_%d%m%Y_%H:%M:%S")
    if cfg.wandb.use:
        exp_name = wandb.run.name
    makedirs(osp(cfg.training.exp_folder, exp_name))
    mkdir(osp(cfg.training.exp_folder, exp_name, "checkpoints"))

    nerf_model = NeRFModel(cfg)
    loss, optimizer, scheduler = make_objects(cfg,
        model_params=list(nerf_model.parameters()))

    train_dataset = BlenderDataset(**cfg.dataset, split='train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, **cfg.dataloader.train)
    train_dataloader = repeater(train_dataloader)

    test_dataset = BlenderDataset(**cfg.dataset, split='test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, **cfg.dataloader.val)

    nerf_model.train()
    nerf_model.to(cfg.training.device)

    step = 1
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    with tqdm(total=cfg.training.num_iterations, desc="Training process") as pbar:
        for batch in train_dataloader:

            rays = batch["ray"].to(cfg.training.device)
            pixels = batch["pixel"].to(cfg.training.device)

            start_time.record()
            coarse_bundle, fine_bundle = nerf_model(rays)
            end_time.record()
            torch.cuda.synchronize()
            time = start_time.elapsed_time(end_time)

            if not fine_bundle:
                loss_value = loss(coarse_bundle['rgb_map'], pixels)
            else:
                coarse_loss = loss(coarse_bundle['rgb_map'], pixels)
                fine_loss = loss(fine_bundle['rgb_map'], pixels)
                loss_value = coarse_loss + fine_loss

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_postfix({"loss": loss_value.item(), "f+b_per_step(ms)": time})
            pbar.update()

            if cfg.wandb.use and (step % cfg.wandb.log_each == 0 or step == 1):
                wandb.log({"loss": loss_value.item(),
                           "lr": scheduler.get_last_lr()[0],
                           "f+b_per_step(ms)": time,
                           "step": step})

            if step % cfg.training.eval_each == 0:
                nerf_model.eval()
                nerf_model.volume_renderer.mode = "test"
                eval_step(cfg, nerf_model, test_dataset, test_dataloader, step, exp_name)
                nerf_model.train()
                nerf_model.volume_renderer.mode = "training"

            step += 1
            if step > cfg.training.num_iterations:
                break

    logger.success("Finish training.")
    ckpt_path = osp(cfg.training.exp_folder, exp_name,
                    "checkpoints", "latest.pth")
    nerf_model.save_state(ckpt_path)
    if cfg.wandb.use:
        artifact = wandb.Artifact("model_latest", type='model')
        artifact.add_file(ckpt_path)
        wandb.run.log_artifact(artifact)

#pylint: disable=no-value-for-parameter
@hydra.main(version_base=None, config_path="configs", config_name="default_train")
def main(cfg: DictConfig) -> None:
    """Entrypoint function."""

    setup_loguru_level(cfg.logger.level)

    if cfg.git.check_uncommited:
        if have_uncommitted_changes():
            logger.critical("Commit your changes to set proper hash in wandb config.")
            sys.exit()
    else:
        logger.warning("Ignore files in git stage area and use last commit in wandb config!")

    if cfg.wandb.use:
        wandb.init(project=cfg.wandb.project)
        config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
        wandb.run.config.update(config)
        logger.info(f"Start wandb logging: {wandb.run.name}")
    else:
        logger.warning("Skip wandb logging!")

    set_seed(cfg.training.seed)

    run_experiment(cfg)

if __name__ == "__main__":
    main()
