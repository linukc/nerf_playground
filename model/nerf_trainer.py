""" Wrapper for NeRF training. """

from typing import Iterator
from itertools import repeat
from os.path import join as osp

import wandb
import torch
from tqdm import tqdm
from loguru import logger
from einops import rearrange
from omegaconf import DictConfig
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr

from model.nerf_model import NeRFModel
from datasets.blender import BlenderCombined
#from depth_metrics import calc_depth_metrics

def repeater(dataloader: DataLoader) -> Iterator[DataLoader]:
    """ Create infinite dataloader. """

    for loader in repeat(dataloader):
        for data in loader:
            yield data

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

class NerfTrainer():
    """ Wrapper for NeRF training. """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        train_dataset = BlenderCombined(root_path=cfg.dataset.root_path,
                                        image_size=cfg.dataset.image_size,
                                        split="train")
        self.train_dataloader = DataLoader(train_dataset, **cfg.dataloader.train)
        self.train_dataloader = repeater(self.train_dataloader)

        val_dataset = BlenderCombined(root_path=cfg.dataset.root_path,
                                       image_size=cfg.dataset.image_size,
                                       split="val")
        self.val_dataloader = DataLoader(val_dataset, **cfg.dataloader.val)

        self.nerf_model = NeRFModel(cfg)
        self.nerf_model.to(cfg.training.device)
        logger.info(f"Move model to {cfg.training.device}.")

        self.loss = instantiate(cfg.training.loss)
        self.optimizer = getattr(torch.optim, cfg.training.optimizer.type)\
            (params=list(self.nerf_model.parameters()),
             **cfg.training.optimizer.args)
        lr_fn = lambda step: cfg.training.scheduler.gamma ** (step / cfg.training.num_iterations)
        self.scheduler = getattr(torch.optim.lr_scheduler, cfg.training.scheduler.type)\
            (self.optimizer, lr_lambda=lr_fn)

        logger.info("Create all components.")

    # pylint: disable=too-many-locals, too-many-statements
    def run(self, exp_name):
        """ Start NeRF training. """

        self.nerf_model.set_train()

        step = 1
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        with tqdm(total=self.cfg.training.num_iterations, desc="Training process") as pbar:
            for batch in self.train_dataloader:
                rays = batch["rays"].to(self.cfg.training.device)
                pixels = batch["rgbs"].to(self.cfg.training.device)

                start_time.record()
                coarse_bundle, fine_bundle = self.nerf_model(rays)
                end_time.record()
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time)

                coarse_loss = self.loss(coarse_bundle['rgb_map'], pixels)
                fine_loss = self.loss(fine_bundle['rgb_map'], pixels)
                total_loss = coarse_loss + fine_loss

                self.optimizer.zero_grad()
                total_loss = coarse_loss + fine_loss
                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                total_loss_value = total_loss.detach().cpu().item()
                train_fine_psnr = mse2psnr(fine_loss.detach().cpu()).item()
                pbar.set_postfix({"total_loss": total_loss_value,
                                  "train_fine_psnr": train_fine_psnr,
                                  "f+b_per_step(ms)": elapsed_time})
                pbar.update()

                statistics = {"total_loss": total_loss_value,
                              "train_fine_psnr": train_fine_psnr,
                              "lr": self.scheduler.get_last_lr()[0],
                              "f+b_per_step(ms)": elapsed_time,
                              "step": step}

                if step % self.cfg.training.eval_each == 0:
                    self.nerf_model.set_eval()
                    metrics = self.validation_step()
                    statistics.update(metrics)
                    self.nerf_model.set_train()

                    ckpt_path = osp(self.cfg.training.exp_folder, exp_name,
                        "checkpoints", f"step_{step}.pth")
                    mlp_state = self.nerf_model.return_state()
                    torch.save(mlp_state, ckpt_path)

                    if self.cfg.wandb.use and self.cfg.wandb.save_val_ckpt:
                        artifact = wandb.Artifact(f"model_{step}", type='model')
                        artifact.add_file(ckpt_path)
                        wandb.run.log_artifact(artifact)

                    tqdm.write(f"[EVAL] PSNR: {metrics.get('psnr')}, SSIM: {metrics.get('ssim')}")

                if self.cfg.wandb.use and (step % self.cfg.wandb.log_each == 0 or
                                           step % self.cfg.training.eval_each == 0):
                    wandb.log(statistics)

                step += 1
                if step > self.cfg.training.num_iterations:
                    break

        ckpt_path = osp(self.cfg.training.exp_folder, exp_name,
            "checkpoints", "latest.pth")
        mlp_state = self.nerf_model.return_state()
        torch.save(mlp_state, ckpt_path)

        if self.cfg.wandb.use and self.cfg.wandb.save_val_ckpt:
            artifact = wandb.Artifact("model_latest", type='model')
            artifact.add_file(ckpt_path)
            wandb.run.log_artifact(artifact)

    def validation_step(self) -> dict:
        """
        Perform evaluation step. 

        Returns
        -------
        metrics:
            Dict with metrics.
        """

        gt_pixels, gt_depth = [], []
        pred_pixels, pred_depth = [], []
        with torch.no_grad():
            for batch in self.val_dataloader:
                rays = batch["rays"].to(self.cfg.training.device)
                _, fine_bundle = self.nerf_model(rays)

                gt_pixels.append(batch["rgbs"].cpu())
                if "depth" in batch:
                    gt_depth.append(batch["depth"].cpu())

                pred_pixels.append(fine_bundle['rgb_map'].detach().cpu())
                pred_depth.append(fine_bundle['depth_map'].detach().cpu())

        pred_depth = torch.cat(pred_depth, dim=0).reshape(8, 400, 400)
        gt_depth = torch.cat(gt_depth, dim=0).reshape(8, 400, 400)
        gt_pixels = torch.cat(gt_pixels, dim=0).reshape(8, 400, 400, 3)

        depth_metrics = {}
        # depth metrics make sense only on first n hundred iterations
        # mask = gt_pixels.sum(dim=-1) / 3 # white background
        # mask = mask != 1.0 # don't calculate on bkg

        # if gt_depth.nelement() != 0:
        #     depth_metrics = calc_depth_metrics(pred_depth, gt_depth, mask)

        viz_metrics = {"psnr": 0, "ssim": 0}
        pred_pixels = rearrange(torch.cat(pred_pixels, dim=0),
            '(num h w) c -> num c h w', h=400, w=400, c=3)
        gt_pixels = gt_pixels.permute((0, 3, 1, 2))
        viz_metrics["ssim"] = round(ssim(pred_pixels, gt_pixels).item(), 3)
        viz_metrics["psnr"] = round(psnr(pred_pixels, gt_pixels).item(), 3)

        if gt_depth.nelement() != 0 and depth_metrics:
            viz_metrics.update(depth_metrics)

        return viz_metrics
