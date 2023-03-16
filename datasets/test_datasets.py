from loguru import logger
import os
import random
import numpy as np
import torch
from load_blender import load_blender_data


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


if __name__ == "__main__":

    ### !!!!!!!!!!!!!!!!!!!!!!
    # Original code
    ###

    # images, poses, _, hwf, i_split = load_blender_data("/media/sergey_mipt/data/datasets/nerf_synthetic/lego", True, 8)
    # # images - картинки
    # # poses - считанные преобразования из yaml
    # i_train, i_val, i_test = i_split
    # images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:]) # white bkg
    
    # # Cast intrinsics to right types
    # H, W, focal = hwf
    # H, W = int(H), int(W)
    # hwf = [H, W, focal]
    # K = None
    # if K is None:
    #     K = np.array([
    #         [focal, 0, 0.5*W],
    #         [0, focal, 0.5*H],
    #         [0, 0, 1]
    #     ])

    # poses = torch.Tensor(poses)
    # N_rand = 2

    # for k in range(10):
    #     # Random from one image
    #     #logger.warning("from 0 image for reproducibility")
    #     #
    #     img_i = np.random.choice(i_train) #replace = True, meaning that a value of a can be selected multiple times.
    #     #img_i=0
    #     target = images[img_i]
    #     target = torch.Tensor(target)
    #     pose = poses[img_i, :3,:4]

    #     rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

        
    #     coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

    #     coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
    #     select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
    #     select_coords = coords[select_inds].long()  # (N_rand, 2)
    #     rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    #     rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    #     batch_rays = torch.cat([rays_o, rays_d], 1)
    #     target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

    #     print(batch_rays)
    #     print(target_s)

    # ## !!!!!!!!!!!!!!!!!!!!!!!!
    # My code
    # ##
    set_seed(0)
    from blender import BlenderDataset
    train_dataset = BlenderDataset(path="/media/sergey_mipt/data/datasets/nerf_synthetic/lego",
                                   split='train',
                                   image_size=400,
                                   bg_color=255)
    # train_dataloader = torch.utils.data.DataLoader(train_dataset,
    #                                                shuffle=False,
    #                                                num_workers=0,
    #                                                pin_memory=True,
    #                                                batch_size=2)
    # sample = next(iter(train_dataloader))
    # sample = train_dataset[0]
    # print(sample.get("ray"))
    # print(sample.get("pixel"))
    print(train_dataset._get_pixels_rays_from_camera_metadata()[1][0, :5, ...])

    set_seed(0)
    from simple_nerf_dataset import BlenderDatasetOld
    train_dataset = BlenderDatasetOld("/media/sergey_mipt/data/datasets/nerf_synthetic/lego", split='train', img_wh=(400, 400))
    # train_dataloader = torch.utils.data.DataLoader(train_dataset,
    #                                                shuffle=False,
    #                                                num_workers=0,
    #                                                pin_memory=True,
    #                                                batch_size=2)
    # sample = next(iter(train_dataloader))
    # sample = train_dataset[0]
    print(train_dataset.dir_cam[0, :5, ...])
    # print(sample.get("rays"))
    # print(sample.get("rgbs"))