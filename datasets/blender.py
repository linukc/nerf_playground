"""Datasets implementation."""

import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
from loguru import logger
from einops import rearrange
from torchvision import transforms
from torch.utils.data import Dataset
from .ray_utils import get_ray_dir_cam, get_rays_torch

depth_index = {"lego": "0001",
                "chair": "0000",
                "drums": "0001",
                "ficus": "0136",
                "hotdog": "0029",
                "materials": "0000",
                "mic": "0186",
                "ship": "0002"}

class BlenderCombined(Dataset):
    def __init__(self, root_path, image_size, **kwargs):
        self.root_dir = root_path
        self.image_size = image_size
        logger.warning(f"Resize 800x800 values to {self.image_size}x{self.image_size}!")
        self.split = kwargs.get("section")
        self.define_transforms()

        self.read_meta()

    def read_meta(self):
        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.image_size, self.image_size
        self.focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        # when W=800

        self.focal *= self.image_size / 800  # modify focal length to match size self.img_wh

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])

        # ray directions for all pixels in camera coordinates, same for all images (same H, W, focal)
        self.dir_cam = \
            get_ray_dir_cam(h, w, self.focal) # (h, w, 3)

        self.all_rays = []
        self.all_rgbs = []
        if self.split == 'train':  # create buffer of all rays and rgb data
            for frame in self.meta['frames']:
                pose = np.array(frame['transform_matrix'])[:3, :4]
                c2w = torch.FloatTensor(pose)

                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                img = Image.open(image_path)
                img = img.resize((self.image_size,) * 2, Image.LANCZOS)
                img = self.transform(img)  # (4, h, w)
                img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA -> (640000, 4)
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB -> (h*w, 3)
                self.all_rgbs += [img]

                rays_o, rays_d = get_rays_torch(self.dir_cam, c2w)  # both (h*w, 3)

                self.all_rays += [
                    torch.cat(
                        [rays_o, rays_d, self.near * torch.ones_like(rays_o[:, :1]), self.far * torch.ones_like(rays_o[:, :1])], 1
                    )
                ]  # (h*w, 8)
        elif self.split == 'test':
            self.all_depth = []
            frames = self.meta["frames"][::25]
            for frame in frames:
                # create data for each image separately
                c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]

                img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
                img = img.resize((self.image_size,) * 2, Image.LANCZOS)
                img = self.transform(img)  # (4, H, W)
                #valid_mask = (img[-1] > 0).flatten()  # (H*W) valid color area
                img = img.view(4, -1).permute(1, 0)  # (H*W, 4) RGBA
                img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
                self.all_rgbs += [img]

                split_name = self.root_dir.split("/")[-1]
                depth_path = f"{frame['file_path']}_depth_{depth_index.get(split_name)}.png"
                #print(depth_path)
                depth = cv2.imread(os.path.join(self.root_dir, depth_path), 0) # grayscale
                depth = cv2.resize(depth, (400,400), interpolation=cv2.INTER_LANCZOS4)
                #cv2.imwrite("gt_depth.png", depth)
                # depth_image = Image.open(
                #     os.path.join(self.root_dir, depth_path)).resize(
                #         (self.image_size,) * 2, Image.LANCZOS).convert('L')
                #depth_image = np.array(depth_image)
                #https://github.com/bmild/nerf/issues/107
                #https://github.com/bmild/nerf/issues/77
                # depth_values has some noise with zero D -> value == 2.0 should be filtered in metric calculation
                near = 2.
                far = 6.
                depth_values = (far+near)*(1 - (depth / 255.).astype(np.float32))
                depth_values = torch.from_numpy(depth_values).view(-1, 1) # 400, 400
                self.all_depth += [depth_values]

                #depth_values = self.transform(depth_values)
                #depth_values = rearrange(depth_values, 'c h w -> (h w) c')

                rays_o, rays_d = get_rays_torch(self.dir_cam, c2w)
                self.all_rays += [
                    torch.cat(
                        [rays_o, rays_d, self.near * torch.ones_like(rays_o[:, :1]), self.far * torch.ones_like(rays_o[:, :1])], 1
                    )
                ]  # (h*w, 8)

        '''
        flatten all rays/rgb tensor
            * self.all_rgbs[idx] -> (r,g,b)
            * self.all_rays[idx] -> (ox,oy,oz,dx,dy,dz,near,far)
        '''
        self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 8) -> (100x800x800, 8)
        self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3) -> (100x800x800, 3)
        if self.split == "test":
            self.all_depth = torch.cat(self.all_depth, 0)

    def define_transforms(self):
        self.transform = transforms.ToTensor()

    def __len__(self):
        # if self.split == 'train':
        #     return len(self.all_rays)
        # elif self.split == 'val':
        #     raise NotImplementedError
        # elif self.split == 'test':
        #     return 8  # only validate 8 images (to support <=8 gpus)
        # return len(self.meta['frames'])
        return len(self.all_rays)

    def __getitem__(self, idx):
        if self.split ==  "train":
            return {'rays': self.all_rays[idx], 'rgbs': self.all_rgbs[idx]}
        else:
            return {'rays': self.all_rays[idx], 'rgbs': self.all_rgbs[idx], 'depth': self.all_depth[idx]}
        # if self.split == 'train':  # use data in the buffers
        #     sample = {'rays': self.all_rays[idx], 'rgbs': self.all_rgbs[idx]}
        # elif self.split == 'val':
        #     raise NotImplementedError
        # elif self.split == 'test':  
        #     idx = 25 * idx - 1 if idx > 0 else 0
        #     # create data for each image separately
        #     frame = self.meta['frames'][idx]
        #     c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]

        #     img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
        #     img = img.resize((self.image_size,) * 2, Image.LANCZOS)
        #     img = self.transform(img)  # (4, H, W)
        #     valid_mask = (img[-1] > 0).flatten()  # (H*W) valid color area
        #     img = img.view(4, -1).permute(1, 0)  # (H*W, 4) RGBA
        #     img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB

        #     split_name = self.root_dir.split("/")[-1]
        #     depth_path = f"{frame['file_path']}_depth_{depth_index.get(split_name)}.png"
        #     depth_image = Image.open(
        #         os.path.join(self.root_dir, depth_path)).resize(
        #             (self.image_size,) * 2, Image.LANCZOS).convert('L')
        #     depth_image = np.array(depth_image)
        #     #https://github.com/bmild/nerf/issues/107
        #     #https://github.com/bmild/nerf/issues/77
        #     # depth_values has some noise with zero D -> value == 2.0 should be filtered in metric calculation
        #     near = 2.
        #     far = 8.
        #     depth_values = near + (far - near)*(1. - (depth_image / 255.).astype(np.float32))
        #     depth_values = torch.from_numpy(depth_values).unsqueeze(dim=2) # 400, 400, 1

        #     #depth_values = self.transform(depth_values)
        #     #depth_values = rearrange(depth_values, 'c h w -> (h w) c')

        #     rays_o, rays_d = get_rays_torch(self.dir_cam, c2w)

        #     rays = torch.cat(
        #         [rays_o, rays_d, self.near * torch.ones_like(rays_o[:, :1]), self.far * torch.ones_like(rays_o[:, :1])], 1
        #     )  # (H*W, 8)

        #     sample = {'rays': rays, 'rgbs': img, 'c2w': c2w, 'valid_mask': valid_mask, "depth": depth_values}

        # return sample