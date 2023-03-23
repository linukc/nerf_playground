"""Utils to setup logging in the training process."""

import os
import sys
import random
from itertools import repeat
from subprocess import check_output, CalledProcessError

import cv2
import torchvision.transforms as T
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from einops import rearrange
#pylint: disable=import-error
import torchvision.utils as tvn
from omegaconf import DictConfig
#pylint: disable=import-error
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
#pylint: disable=import-error
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import wandb


def run_bash_command(bash_cmd: str) -> str:
    """Execute a bash command and capture the resulting stdout without quotes.

    Parameters
    ----------
    bash_cmd: str
        The bash command to run.

    Returns
    -------
    stdout: str
        The resulting stdout output.
    """

    stdout = check_output(bash_cmd.split()).decode('utf-8').rstrip('\n')
    return stdout[1:-1]

def have_uncommitted_changes() -> bool:
    """Check if you have uncommented changes in code
    to prevent from future mess between different versions of experiments.

    Returns
    -------
    flag : bool
    """

    try:
        bash_cmd = "git status"
        flag = "nothing to commit" not in run_bash_command(bash_cmd)
    except CalledProcessError:
        logger.critical("Please, create git repo.")
        sys.exit()

    return flag

def get_current_commit_hash() -> str:
    """Get the hash of the most recent commit.

    Returns
    -------
    hash : str
        The 6 character hash of the most recent commit.
    """

    bash_cmd = "git log -1 --pretty=format:'%H'"
    return run_bash_command(bash_cmd)[:6]

def setup_loguru_level(level: str="INFO") -> None:
    """Set loguru.logger level.
    DEBUG->INFO->SUCCESS->WARNING->ERROR->CRITICAL
    """

    logger.remove()
    logger.add(sys.stderr, level=level)
    getattr(logger, level.lower())(f"Set {level} level for logger output.")

def set_seed(seed: int) -> None:
    """Set random seed to reproducibility as wandb [1].

    Parameters
    ----------
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

def make_objects(cfg: DictConfig, model_params: list):
    """ Create loss, optimizer and schedualer by config."""

    loss = getattr(torch.nn.modules.loss, cfg.training.loss.type)()

    optimizer = getattr(torch.optim, cfg.training.optimizer.type)\
        (params=model_params, **cfg.training.optimizer.args)

    #pylint: disable=unnecessary-lambda-assignment
    lr_fn = lambda step: cfg.training.scheduler.gamma ** (step / cfg.training.num_iterations)
    sheduler = getattr(torch.optim.lr_scheduler, cfg.training.scheduler.type)\
        (optimizer, lr_lambda=lr_fn)

    return loss, optimizer, sheduler

def repeater(dataloader):
    """ Create infinite dataloader."""

    for loader in repeat(dataloader):
        for data in loader:
            yield data

#pylint: disable=too-many-arguments, too-many-locals
def analyze(gt_pixels: torch.Tensor, pred_pixels, gt_depth, pred_depth, pred_accmap,
    image_size, use_wandb, image_num, path, step):
    """ Calculate metrics by predictions."""

    def norm_image_and_color(tensor: torch.Tensor):
        min_, _ = torch.min(tensor, dim=0, keepdim=True)
        max_, _ = torch.max(tensor, dim=0, keepdim=True)
        x = (tensor - min_) / (max_ - min_ + 1e-8) * 255
        x = x.squeeze(dim=1).numpy().astype(np.uint8) # 8 400 400
        out = torch.zeros((8, 3, 400, 400))
        for i in range(8):
            color= Image.fromarray(cv2.applyColorMap(x[i], cv2.COLORMAP_TURBO))
            color = T.ToTensor()(color) # (3, H, W)
            out[i] = color
        return out * 255

    gt_pixels = rearrange(torch.cat(gt_pixels, dim=0),
        '(num h w) c -> num c h w', h=image_size, w=image_size, c=3)
    pred_pixels = rearrange(torch.cat(pred_pixels, dim=0),
        '(num h w) c -> num c h w', h=image_size, w=image_size, c=3)
    gt_depth = rearrange(torch.cat(gt_depth, dim=0),
        '(num h w) c -> num c h w', h=image_size, w=image_size, c=1)
    pred_depth = rearrange(torch.cat(pred_depth, dim=0),
        '(num h w) c -> num c h w', h=image_size, w=image_size, c=1)
    pred_accmap = rearrange(torch.cat(pred_accmap, dim=0),
        '(num h w) c -> num c h w', h=image_size, w=image_size, c=1)

    if gt_pixels.shape[0] != image_num:
        logger.critical("Check tensors rearrange!")
        sys.exit()
    if image_num > 8:
        logger.warning("Edit image saving (now it is suitable for 8 images).")

    os.makedirs(os.path.join(path, "images"), exist_ok=True)
    tvn.save_image(tensor=torch.cat((pred_pixels, gt_pixels), dim=0),
        fp=f"{path}/images/images_{step}.png", nrow=8)
    os.makedirs(os.path.join(path, "depth"), exist_ok=True)
    tvn.save_image(tensor=norm_image_and_color(pred_depth),
                        fp=f"{path}/depth/depth_{step}.png", nrow=8)
    #tvn.save_image(tensor=pred_disp, fp=f"{path}/disp_{step}.png", nrow=8)
    os.makedirs(os.path.join(path, "acc_map"), exist_ok=True)
    tvn.save_image(tensor=norm_image_and_color(pred_accmap), fp=f"{path}/acc_map/acc_map_{step}.png", nrow=8)

    mask = gt_pixels.sum(dim=1)/3 # 8 400 400
    mask.unsqueeze_(dim=1)
    mask = mask != 1.0
    mask2 = gt_depth > 2.0
    mask = torch.logical_and(mask, mask2)
    depth_metrics = calc_depth_metrics(pred_depth, gt_depth, mask)

    mask = mask.float()
    os.makedirs(os.path.join(path, "object_mask"), exist_ok=True)
    tvn.save_image(tensor=mask, fp=f"{path}/object_mask/mask_{step}.png", nrow=8)
    
    tqdm.write(", ".join([f"{item[0]}: {item[1]}" for item in depth_metrics.items()]))

    msg = "rewrite metrics -> change to functional; \
    psnr -> check data range of pred_pixels (1 or 255) \
    check Max I value"
    #logger.critical(msg)
    psnr_metric = PeakSignalNoiseRatio()
    psnr = round(psnr_metric(pred_pixels, gt_pixels).item(), 3)

    ssim_metric = StructuralSimilarityIndexMeasure()
    ssim = round(ssim_metric(pred_pixels, gt_pixels).item(), 3)

    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True)
    lpips = round(lpips_metric((pred_pixels / 255).clamp(0, 1), gt_pixels / 255).detach().cpu().item(), 3)

    px_metrics = {"psnr": psnr, "ssim": ssim, "lpips": lpips}
    tqdm.write(", ".join([f"{item[0]}: {item[1]}" for item in px_metrics.items()]))

    if use_wandb:
        wandb.log(depth_metrics)
        wandb.log(px_metrics)

def calc_depth_metrics(pred_t, gt_t, masks):
    """Calc depth metrics."""
    #print(masks.shape)
    batch_size = gt_t.size(0)
    scale, abs_diff, abs_rel, sq_rel, a_1, a_2, a_3 = 0, 0, 0, 0, 0, 0, 0
    #tqdm.write("[WARNING] Calculate depth metrics after scaling by gt_depth!")

    for pair in zip(pred_t, gt_t, masks):
        pred, grount_truth, one_mask = pair
        # print(pred.shape)
        # print(grount_truth.shape)
        # print(one_mask.shape)
        #only for blender lego
        valid_gt = grount_truth[one_mask]
        valid_pred = pred[one_mask]
        scale += torch.median(valid_gt) / torch.median(valid_pred)

        #valid_pred *= scale
        #print("calc without scale for blender dataset")
        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a_1 += (thresh < 1.25).float().mean()
        a_2 += (thresh < 1.25 ** 2).float().mean()
        a_3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)
        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

        names = ["scale", "abs_diff", "abs_rel", "sq_rel", "a1", "a2", "a3"]
        metrics = [round(metric.item() / batch_size, 3)  for metric in
            [scale, abs_diff, abs_rel, sq_rel, a_1, a_2, a_3]]

    return {f"median_{k}": v for k, v in zip(names, metrics)}
