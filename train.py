"""Main training script."""

import sys
from datetime import datetime
from os.path import join as osp
from os import makedirs, mkdir
import torchvision as tvn
import os 

from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from einops import rearrange

import torch
import hydra
from loguru import logger
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import wandb

from model.nerf_model import NeRFModel
from datasets.blender import BlenderCombined
from utils import set_seed, repeater, make_objects, analyze
from utils import have_uncommitted_changes, setup_loguru_level

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image

def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x) # change nan to 0
    mi = np.min(x) # get minimum depth
    ma = np.max(x)
    x = (x-mi)/max(ma-mi, 1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("my_experiment_with_depth")

#pylint: disable=too-many-arguments, too-many-locals
def eval_step(cfg, nerf_model, dataset, dataloader, step, exp_name):
    """ Evaluation step.
        Isolate gpu inference into func scope to release memory after with gc.
    """

    tqdm.write(f"[TEST] =======> Step: {step}")
    ckpt_path = osp(cfg.training.exp_folder, exp_name,
                    "checkpoints", f"step_{step}.pth")
    nerf_model.save_state(ckpt_path)

    if cfg.wandb.use and cfg.wandb.save_val_ckpt:
        artifact = wandb.Artifact(f"model_{step}", type='model')
        artifact.add_file(ckpt_path)
        wandb.run.log_artifact(artifact)

    gt_pixels, gt_depth = [], []
    pred_pixels, pred_depth, pred_accmap = [], [], []

    with torch.no_grad():

        for test_batch in dataloader:
            test_rays = test_batch["rays"].to(cfg.training.device)
            gt_pixels.append(test_batch["rgbs"].cpu())
            gt_depth.append(test_batch["depth"].cpu())

            coarse_bundle, fine_bundle = nerf_model(test_rays)
            if not fine_bundle:
                bundle = coarse_bundle
            else:
                bundle = fine_bundle

            pred_pixels.append(bundle['rgb_map'].detach().cpu())
            pred_accmap.append(bundle['acc_map'].detach().cpu().unsqueeze(1))

            pred_depth.append(bundle['depth_map'].detach().cpu()) #
        
        pred_depth = torch.cat(pred_depth, dim=0).reshape(8, 400, 400)
        #path = osp(cfg.training.exp_folder, exp_name)
        # makedirs(os.path.join(path, "depth"), exist_ok=True)
        # with open(f"{path}/depth/depth_pred_value_{step}.npy", 'wb') as file:
        #     np.save(file, pred_depth.numpy())
        # depth = visualize_depth(pred_depth) # (3, H, W)
        #writer.add_image('depth', depth, step)

        # tvn.utils.save_image(tensor=depth,
        #                fp=f"{path}/depth/depth_{step}.png")
        gt_depth = torch.cat(gt_depth, dim=0).reshape(8, 400, 400)
        # with open(f"{path}/depth/depth_gt_value_{step}.npy", 'wb') as file:
        #     np.save(file, gt_depth.numpy())
        # gt_depth_viz = visualize_depth(gt_depth)
        #print(gt_depth.shape)
        # tvn.utils.save_image(tensor=gt_depth_viz,
        #                fp=f"{path}/depth/depth_gt_{step}.png")
        
        from utils import calc_depth_metrics
        gt_pixels = torch.cat(gt_pixels, dim=0).reshape(8, 400, 400, 3)
        mask = gt_pixels.sum(dim=-1)/3
        mask = mask != 1.0
        # tvn.utils.save_image(tensor=gt_pixels.permute((2, 0, 1)),
        #                fp=f"{path}/depth/mask_{step}.png")

        depth_metrics = calc_depth_metrics(pred_depth, gt_depth, mask)
        #print(res)
        wandb.log(depth_metrics)

        viz_metrics = {"psnr": 0, "ssim": 0}
        pred_pixels = rearrange(torch.cat(pred_pixels, dim=0),
        '(num h w) c -> num c h w', h=400, w=400, c=3)
        gt_pixels = gt_pixels.permute((0, 3, 1, 2))
        viz_metrics["ssim"] = round(ssim(pred_pixels, gt_pixels).item(), 3)
        viz_metrics["psnr"] = round(psnr(pred_pixels, gt_pixels).item(), 3)
        wandb.log(viz_metrics)
        
        # analyze(gt_pixels, pred_pixels, gt_depth, pred_depth, pred_accmap,
        #     image_size=dataset.image_size, use_wandb=cfg.wandb.use,
        #     image_num=8,
        #     path=osp(cfg.training.exp_folder, exp_name), step=step)

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
    OmegaConf.save(cfg, osp(cfg.training.exp_folder, exp_name, "config.yaml"))

    nerf_model = NeRFModel(cfg)
    loss, optimizer, scheduler = make_objects(cfg,
        model_params=list(nerf_model.parameters()))

    train_dataset = BlenderCombined(root_path=cfg.dataset.root_path,
                                    image_size=cfg.dataset.image_size,
                                    **cfg.dataset.train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, **cfg.dataloader.train)
    train_dataloader = repeater(train_dataloader)

    test_dataset = BlenderCombined(root_path=cfg.dataset.root_path,
                                   image_size=cfg.dataset.image_size,
                                   **cfg.dataset.val)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, **cfg.dataloader.val)

    nerf_model.train()
    nerf_model.to(cfg.training.device)

    step = 1
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    with tqdm(total=cfg.training.num_iterations, desc="Training process") as pbar:
        for batch in train_dataloader:

            rays = batch["rays"].to(cfg.training.device)
            pixels = batch["rgbs"].to(cfg.training.device)

            start_time.record()
            coarse_bundle, fine_bundle = nerf_model(rays)
            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time)

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

            pbar.set_postfix({"loss": loss_value.item(), "f+b_per_step(ms)": elapsed_time})
            pbar.update()

            if cfg.wandb.use and (step % cfg.wandb.log_each == 0 or step == 1):
                wandb.log({"loss": loss_value.item(),
                           "psnr": mse2psnr(fine_loss.detach().cpu()).item(),
                           "lr": scheduler.get_last_lr()[0],
                           "f+b_per_step(ms)": elapsed_time,
                           "step": step})

            #typical each N step and additional val at the begining (largest slope in metrics)
            if (step % cfg.training.eval_each == 0) or ( (step < 5_000) and (step % 250 == 0)) or (step in range(50, 1051, 100)):
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
@hydra.main(version_base=None, config_path="configs", config_name="train")
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
