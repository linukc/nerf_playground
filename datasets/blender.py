""" 
Nerf Sythetic Dataset, combined implimintation. 
(sample batch over all rays in dataset)
"""

import os
import json

import cv2
import torch
import numpy as np
from PIL import Image
from loguru import logger
from torchvision import transforms
from torch.utils.data import Dataset

from datasets.ray_utils import get_ray_dir_cam, get_rays_torch


# pylint: disable=too-many-instance-attributes
class BlenderCombined(Dataset):
    """ Nerf Sythetic Dataset. """

    def __init__(self, root_path: str, image_size: int, split: str="train"):
        self.root_dir = root_path
        self.image_size = image_size
        logger.warning(f"Resize 800x800 values to {self.image_size}x{self.image_size}!")
        self.split = split
        if self.split == "val":
            self.split = "test"
            logger.warning("Use test split in validation!")
            logger.warning("Depth only tested for the lego scene!")

        self.transform = transforms.ToTensor()
        self.read_metainfo()

    def read_metainfo(self):
        """ Combine all rays and images from dataset in one chunk of data. """

        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"),
                  "r", encoding="utf8") as file:
            self.meta = json.load(file)

        # original focal length when W=800
        self.focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])
        # modify focal length to match self.image_size
        self.focal *= self.image_size / 800

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])

        # ray directions for all pixels in camera coordinates
        # same for all images (same H, W, focal)
        self.dir_cam = \
            get_ray_dir_cam(self.image_size, self.image_size, self.focal)

        self.all_rays = []
        self.all_rgbs = []

        if self.split == "test" and "lego" in self.root_dir:
            self.all_depth = []

        if self.split == "train":
            frames = self.meta['frames']
        else:
            frames = self.meta['frames'][::25]
            logger.warning("Use each 25th frame for validation!")

        for frame in frames:
            pose = np.array(frame['transform_matrix'])[:3, :4]
            c2w = torch.FloatTensor(pose)

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            img = Image.open(image_path)
            img = img.resize((self.image_size,) * 2, Image.LANCZOS)
            img = self.transform(img) # (4, h, w)
            img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:]) # blend A to RGB -> (h*w, 3)
            self.all_rgbs += [img]

            rays_o, rays_d = get_rays_torch(self.dir_cam, c2w) # both (h*w, 3)

            self.all_rays += [
                torch.cat(
                    [rays_o, rays_d,
                     self.near * torch.ones_like(rays_o[:, :1]),
                     self.far * torch.ones_like(rays_o[:, :1])], dim=1)] # (h*w, 8)

            if self.split == "test" and "lego" in self.root_dir:
                depth_path = f"{frame['file_path']}_depth_0001.png"
                depth = cv2.imread(os.path.join(self.root_dir, depth_path), 0) # grayscale
                depth = cv2.resize(depth, (400,400), interpolation=cv2.INTER_LANCZOS4)
                #https://github.com/bmild/nerf/issues/107
                #https://github.com/bmild/nerf/issues/77
                depth_values = (self.far + self.near) * (1 - (depth / 255.).astype(np.float32))
                depth_values = torch.from_numpy(depth_values).view(-1, 1)
                self.all_depth += [depth_values]

        self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 8)
        self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)
        if self.split == "test" and "lego" in self.root_dir:
            self.all_depth = torch.cat(self.all_depth, 0)

    def __len__(self):
        return len(self.all_rays)

    def __getitem__(self, idx: int):
        if self.split ==  "test":
            return {'rays': self.all_rays[idx],
                    'rgbs': self.all_rgbs[idx],
                    'depth': self.all_depth[idx]}
        return {'rays': self.all_rays[idx], 'rgbs': self.all_rgbs[idx]}
