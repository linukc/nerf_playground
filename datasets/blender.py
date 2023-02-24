"""Datasets implementation."""

import os
import json

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from loguru import logger
from einops import rearrange
from torchvision import transforms
from torch.utils.data import Dataset


#pylint: disable=too-many-instance-attributes
class BlenderDataset(Dataset):
    """Dataset to import scene from blender metadata.

    Parameters
    ----------
    path: str
        Path to folder with images and jsons.
    split: str
        Flag to decide which split of data to load.
    """

    #pylint: disable=too-many-arguments
    def __init__(self, path: str,
                       image_size: int,
                       bg_color: int,
                       split: str='train',
                       test_each: int=50):

        logger.warning(f"Process images as squares with {image_size}px side!")

        self.path = path
        self.image_size = image_size
        self.bg_color = bg_color
        self.split = split
        self.test_each = test_each

        self._train_rays = []
        self._train_pixels = []
        self._test_rays = []
        self._test_pixels = []
        self._test_depth_values = []

        self.transform = transforms.ToTensor()
        self.read_metadata()

    def read_metadata(self):
        """Reads json file and prepeares arrays for sampling."""

        path = os.path.join(self.path, f"transforms_{self.split}.json")
        with open(path, 'rt', encoding='utf-8') as file:
            self.meta = json.load(file)

        self.orig_image_size = 800
        if self.orig_image_size // self.image_size != 1:
            logger.warning(f"Resize gt_depth 800x800 values to {self.image_size}!")

        self.focal = 0.5 * self.image_size / np.tan(0.5 * self.meta['camera_angle_x'])
        self.focal *= self.image_size / 800  # modify focal length to match image_size

        logger.debug("Calculate camera rays.")
        self.camera_rays_origins, self.camera_rays_dir = \
            self._get_pixels_rays_from_camera_metadata()

        if self.split == "train":
            logger.debug("Load images and camera poses for train.")
            self._read_train_data()
        elif self.split == "test":
            logger.debug("Load images, camera poses and depth for test.")
            self._read_test_data()

    def _get_pixels_rays_from_camera_metadata(self):
        """Calculates rays vectors for each pixel.
        Look at the all conventions here [1].

        Returns
        -------
        rays_origins: torch.array(h, w, 3)
            Vectors bases.
        rays_directions: torch.array(h, w, 3)
            Vectors directions.

        [1] https://www.scratchapixel.com/lessons/3d-basic-rendering/
        ray-tracing-generating-camera-rays/generating-camera-rays.html
        """
        #px_x = (((u + 0.5) / image_size) * 2 - 1) * np.tan(alpha/2)
        #px_y = (-((v + 0.5) / image_size) * 2 + 1) * np.tan(alpha/2)

        linspace = np.linspace(0, self.image_size - 1, self.image_size, dtype=np.float32)
        u_cord, v_coord = np.meshgrid(linspace, linspace)
        rays_dir = np.stack([(u_cord - self.image_size / 2) / self.focal,
                             -(v_coord - self.image_size / 2) / self.focal,
                             -np.ones((self.image_size, self.image_size))], axis=-1)

        return torch.zeros((self.image_size, self.image_size, 3)), \
               torch.FloatTensor(rays_dir / np.expand_dims(np.linalg.norm(rays_dir, axis=-1),
                axis=-1))

    def _get_rays_and_px_from_image(self, c2w, image_path):
        rot_matrix, translate = c2w[:, :3], c2w[:, 3]
        world_rays_origins = (self.camera_rays_origins + translate).view(-1, 3)
        world_rays_dir = (self.camera_rays_dir @ rot_matrix.T).view(-1, 3)
        rays = torch.cat([world_rays_origins,
                            world_rays_dir], dim=1)

        img = Image.open(image_path) #RGBA
        background = Image.new('RGBA', img.size, (self.bg_color,) * 3)
        alpha_composite = Image.alpha_composite(background, img)
        img = alpha_composite.convert("RGB").resize((self.image_size,) * 2, Image.LANCZOS)
        img = self.transform(img)
        img = rearrange(img, 'c h w -> (h w) c')

        return rays, img

    def _read_train_data(self):
        """Fills self.rays and self.pixels arrays."""

        for frame in tqdm(self.meta["frames"]):
            camera_to_world = torch.FloatTensor(frame["transform_matrix"])[:3, :4]
            image_path = os.path.join(self.path, f"{frame['file_path']}.png")
            rays, img = self._get_rays_and_px_from_image(camera_to_world, image_path)
            self._train_rays.append(rays) # h*w 6
            self._train_pixels.append(img) # h*w 3

        # frames_count h*w 6 -> ray_count 6
        self._train_rays = torch.cat(self._train_rays, dim=0)
        # frames_count h*w 3 -> ray_count 3
        self._train_pixels = torch.cat(self._train_pixels, dim=0)

    def _read_test_data(self):
        """Fills self.rays and self.pixels arrays."""

        frames = self.meta["frames"][::self.test_each]
        for frame in tqdm(frames):
            camera_to_world = torch.FloatTensor(frame["transform_matrix"])[:3, :4]
            image_path = os.path.join(self.path, f"{frame['file_path']}.png")
            rays, img = self._get_rays_and_px_from_image(camera_to_world, image_path)
            self._test_rays.append(rays) # h*w 6
            self._test_pixels.append(img) # h*w 3

            depth_image = Image.open(
                os.path.join(self.path, f"{frame['file_path']}_depth_0001.png")).resize(
                    (self.image_size,) * 2, Image.LANCZOS)
            depth_values = np.array(depth_image)[:, :, 0] # ~ 170 max value, looks like cm
            depth_values = self.transform(depth_values)
            depth_values = rearrange(depth_values, 'c h w -> (h w) c')
            self._test_depth_values.append(depth_values) # h*w 1

        self._test_rays = torch.cat(self._test_rays, dim=0) # 4 h*w 6 -> ray_count 6
        self._test_pixels = torch.cat(self._test_pixels, dim=0) # 4 h*w 3 -> ray_count 3
        self._test_depth_values = torch.cat(self._test_depth_values, dim=0) # 4 h*w 1 -> ray_count 1

    def __getitem__(self, idx: int) -> dict:
        """Extracts ray by index.

        Parameters
        ----------
        idx:
            Ray index.

        Returns
        -------
        sample:
            Ray and pixel value.
            dict("ray": torch.tensor([6]),
                 "pixel": torch.tensor(3]))
        """

        if self.split == "train":
            return {"ray": self._train_rays[idx], "pixel": self._train_pixels[idx]}
        if self.split == "test":
            return {"ray": self._test_rays[idx], "gt_pixel": self._test_pixels[idx],
                    "gt_depth": self._test_depth_values[idx]}
        return None

    def __len__(self) -> int:
        """Defines number of rays in scene.

        Returns
        -------
        len:
            Number of rays.
        """

        if self.split == "train":
            return self._train_rays.shape[0]
        if self.split == "test":
            return self._test_rays.shape[0]
        return None
