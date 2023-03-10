"""Main training script."""

import sys
from datetime import datetime
from os.path import join as osp
from os import makedirs, mkdir
import numpy as np

import torch
import hydra
from loguru import logger
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import wandb

from model.nerf_model import NeRFModel
from datasets.load_blender import load_blender_data
from utils import set_seed, make_objects, analyze
from utils import have_uncommitted_changes, setup_loguru_level

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


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

    nerf_model.train()
    nerf_model.to(cfg.training.device)

    step = 1
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    ####
    images, poses, render_poses, hwf, i_split = load_blender_data("/media/sergey_mipt/data/datasets/nerf_synthetic/lego",
                                                                    True, 8)
    print('Loaded blender', images.shape, render_poses.shape, hwf, "/media/sergey_mipt/data/datasets/nerf_synthetic/lego")
    i_train, i_val, i_test = i_split

    near = 2.
    far = 6.

    images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:]) # white bkg

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    K = None
    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    poses = torch.Tensor(poses).to("cuda")

    ####

    N_rand = cfg.dataloader.train.batch_size
    with tqdm(total=cfg.training.num_iterations, desc="Training process") as pbar:
        for i in range(cfg.training.num_iterations):

            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to("cuda")
            pose = poses[img_i, :3,:4]

            rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

            
            coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

            coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
            select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
            select_coords = coords[select_inds].long()  # (N_rand, 2)
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            batch_rays = torch.cat([rays_o, rays_d], 1)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)


            start_time.record()
            coarse_bundle, fine_bundle = nerf_model(batch_rays)
            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time)

            if not fine_bundle:
                loss_value = loss(coarse_bundle['rgb_map'], target_s)
            else:
                coarse_loss = loss(coarse_bundle['rgb_map'], target_s)
                fine_loss = loss(fine_bundle['rgb_map'], target_s)
                loss_value = coarse_loss + fine_loss

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_postfix({"loss": loss_value.item(), "f+b_per_step(ms)": elapsed_time})
            pbar.update()

            if cfg.wandb.use and (step % cfg.wandb.log_each == 0 or step == 1):
                wandb.log({"loss": loss_value.item(),
                           "lr": scheduler.get_last_lr()[0],
                           "f+b_per_step(ms)": elapsed_time,
                           "psnr": mse2psnr(fine_loss.detach().cpu()).item()})

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
