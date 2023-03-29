""" NeRF model. """

from typing import Tuple, Any

import torch
from torch import nn
from loguru import logger
from omegaconf import DictConfig

from model.nerf_mlp import NeRFMLP
from model.encoding import HighFreqEncoding
from model.volume_render import VolumeRenderer
from model.points_sampler import IntervalSampler
from model.points_sampler import HierarchicalPDFSampler
from model.points_sampler import intervals_to_ray_points


# pylint: disable=too-many-instance-attributes
class NeRFModel(nn.Module):
    """ NeRF Model. See section 5. """

    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg

        if cfg.model.encoding.use:
            self.coords_encoding = HighFreqEncoding(cfg.model.encoding.num_freqs_coords)
            self.viewdir_encoding = HighFreqEncoding(cfg.model.encoding.num_freqs_viewdir)
        else:
            logger.warning("Skip encoding!")

        self.mlp_coarse = NeRFMLP(cfg.model)
        self.mlp_fine = NeRFMLP(cfg.model)

        self.interval_sampler_coarse = IntervalSampler(**cfg.model.interval_sampler)
        if cfg.model.hierarchical_sampler.use:
            self.hsampler = HierarchicalPDFSampler(cfg.model.hierarchical_sampler.num_samples,
                                                   cfg.model.hierarchical_sampler.perturb)
        else:
            fine_interval_sampler_cfg = cfg.model.interval_sampler
            fine_interval_sampler_cfg["num_samples"] = cfg.model.hierarchical_sampler.num_samples
            self.interval_sampler_fine = IntervalSampler(**fine_interval_sampler_cfg)

        self.volume_renderer = VolumeRenderer(**cfg.model.volume_renderer)

    def forward(self, rays: torch.Tensor) -> Tuple[dict, dict]:
        """NeRF forward propogation function.

        Arguments
        ---------
        rays:
            Tensor containing batch of rays origins, rays direcitons, rays near and far.
            [num_rays, 8]

        Returns
        -------
            Tuple of dicts, results of coarse and fine volume rendering.
        """

        ray_origins, ray_dirs = rays[:, 0:3], rays[:, 3:6]
        num_rays = ray_dirs.shape[0]

        coarse_bundle, fine_bundle = {}, {}
        for stage in ("coarse", "fine"):
            if stage == "coarse":
                ray_depth_values = self.interval_sampler_coarse(ray_count=num_rays)
            elif stage == "fine":
                if self.cfg.model.hierarchical_sampler.use:
                    ray_depth_values = self.hsampler(ray_depth_values,
                                                     coarse_bundle['weights'])
                else:
                    ray_depth_values = self.interval_sampler_fine(ray_count=num_rays)

            # [num_rays, num_{stage}_samples]
            ray_points = intervals_to_ray_points(
                point_intervals=ray_depth_values, ray_origins=ray_origins, ray_directions=ray_dirs)

            # Expand rays view dir to match batch size [num_rays, num_samples, 3]
            viewdirs = ray_dirs[..., None, :].expand_as(ray_points).contiguous()

            if self.cfg.model.encoding.use:
                ray_points = self.coords_encoding(ray_points)
                viewdirs = self.viewdir_encoding(viewdirs)

            if stage == "coarse":
                if self.cfg.model.mlp.use_viewdir:
                    coarse_radiance = self.mlp_coarse(xyz=ray_points, viewdirs=viewdirs)
                else:
                    coarse_radiance = self.mlp_coarse(xyz=ray_points)
                coarse_bundle = self.volume_renderer(
                    radiance_field=coarse_radiance,
                    depth_values=ray_depth_values,
                    ray_directions=ray_dirs)

            elif stage == "fine":
                if self.cfg.model.mlp.use_viewdir:
                    fine_radiance = self.mlp_fine(xyz=ray_points, viewdirs=viewdirs)
                else:
                    fine_radiance = self.mlp_fine(xyz=ray_points)
                fine_bundle = self.volume_renderer(
                    radiance_field=fine_radiance,
                    depth_values=ray_depth_values,
                    ray_directions=ray_dirs)

        return coarse_bundle, fine_bundle

    def return_state(self) -> dict:
        """ Return MLP's state dicts. 
        
        Returns
        -------
        state_dict:
            Dictionary with weights for MLPs.
        """

        return {"mlp_coarse_state_dict": self.mlp_coarse.state_dict(),
                "mlp_fine_state_dict": self.mlp_fine.state_dict()}

    def load_state(self, checkpoint_dict: Any):
        """ Load network state dict.
        
        Arguments
        ---------
        checkpoint_dict:
            Loaded state dict.
        """

        self.mlp_coarse.load_state_dict(checkpoint_dict['mlp_coarse_state_dict'])
        self.mlp_fine.load_state_dict(checkpoint_dict['mlp_fine_state_dict'])

    def set_train(self):
        """ Enable train mode. """

        self.mlp_coarse.train()
        self.mlp_fine.train()
        self.volume_renderer.mode = "train"

    def set_eval(self):
        """ Enable eval mode. """

        self.mlp_coarse.eval()
        self.mlp_fine.eval()
        self.volume_renderer.mode = "test"
