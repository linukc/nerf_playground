"""NeRF model"""

import torch
import mcubes #pylint: disable=import-error
from torch import nn
from loguru import logger
from omegaconf import DictConfig

#pylint: disable=import-error
from model.nerf_mlp import NeRFMLP
from model.encoding import HighFreqEncoding
from model.volume_render import VolumeRenderer
from model.points_sampler import IntervalSampler
from model.points_sampler import HierarchicalPDFSampler
from model.points_sampler import intervals_to_ray_points


class NeRFModel(nn.Module): #pylint: disable=too-many-instance-attributes
    """NeRF Model. Section sections 5.2 and 5.3.

    Parameters
    ----------
    See configs/.yaml/model
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()

        if cfg.model.encoding.use:
            self.coords_encoding = HighFreqEncoding(cfg.model.encoding.num_freqs_coords)
            self.viewdir_encoding = HighFreqEncoding(cfg.model.encoding.num_freqs_viewdir)
        else:
            logger.warning("Skip encoding!")

        self.mlp_coarse = NeRFMLP(cfg.model)
        if cfg.model.use_fine_mlp:
            self.mlp_fine = NeRFMLP(cfg.model)
        else:
            logger.warning("Skip fine_mlp!")

        self.interval_sampler = IntervalSampler(**cfg.model.interval_sampler)
        if cfg.model.hierarchical_sampler.use:
            self.hsampler = HierarchicalPDFSampler(cfg.model.hierarchical_sampler.num_fine_samples,
                                                   cfg.model.hierarchical_sampler.perturb)
        self.volume_renderer = VolumeRenderer(**cfg.model.volume_renderer)

        self.cfg = cfg

    def forward(self, rays: torch.Tensor):
        """NeRF Model propogation function.

        Parameters
        ----------
        rays:
            Tensor containing batch of rays origins, rays direcitons, rays near and far.
            Shape num_rays, 8.

        Returns
        -------
            Dict containing the outputs of the rendering results.
        """

        ray_origins, ray_dirs = rays[:, 0:3], rays[:, 3:6]
        num_rays = ray_dirs.shape[0]

        coarse_bundle, fine_bundle = None, None

        renderer_type = ("coarse", "fine")
        if not self.cfg.model.use_fine_mlp:
            renderer_type = ("coarse",)

        for renderer in renderer_type:
            if renderer == "coarse":
                # Generating intervals
                ray_depth_values = self.interval_sampler(ray_count=num_rays)
            elif renderer == "fine":
                if self.cfg.model.hierarchical_sampler.use:
                    ray_depth_values = self.hsampler(ray_depth_values,
                                                     coarse_bundle['weights']) #pylint: disable=unsubscriptable-object
            # [num_rays, self.num_coarse_samples]
            ray_points = intervals_to_ray_points(
                point_intervals=ray_depth_values, ray_origins=ray_origins, ray_directions=ray_dirs)

            # Expand rays view dir to match batch size (ray_count, num_samples, 3)
            viewdirs = ray_dirs[..., None, :].expand_as(ray_points).contiguous()

            if self.cfg.model.encoding.use:
                ray_points = self.coords_encoding(ray_points)
                viewdirs = self.viewdir_encoding(viewdirs)

            if renderer == "coarse":
                coarse_radiance = self.mlp_coarse(xyz=ray_points, viewdirs=viewdirs)
                coarse_bundle = self.volume_renderer(
                    radiance_field=coarse_radiance,
                    depth_values=ray_depth_values,
                    ray_directions=ray_dirs)

            elif renderer == "fine":
                fine_radiance = self.mlp_fine(xyz=ray_points, viewdirs=viewdirs)
                fine_bundle = self.volume_renderer(
                    radiance_field=fine_radiance,
                    depth_values=ray_depth_values,
                    ray_directions=ray_dirs)

        return coarse_bundle, fine_bundle

    def save_state(self, path: str):
        """Return network state dict"""

        if self.cfg.model.use_fine_mlp:
            checkpoint_dict = {
                "mlp_coarse_state_dict": self.mlp_coarse.state_dict(),
                "mlp_fine_state_dict": self.mlp_fine.state_dict(),
            }
        else:
            checkpoint_dict = {
                "mlp_coarse_state_dict": self.mlp_coarse.state_dict()
            }
        torch.save(checkpoint_dict, path)

    def load_state(self, checkpoint_dict):
        """load network state dict"""
        if self.cfg.model.use_fine_mlp:
            self.mlp_coarse.load_state_dict(checkpoint_dict['mlp_coarse_state_dict'])
            self.mlp_fine.load_state_dict(checkpoint_dict['mlp_fine_state_dict'])
        else:
            self.mlp_coarse.load_state_dict(checkpoint_dict['mlp_coarse_state_dict'])

    @torch.no_grad()
    def extract_mesh( #pylint: disable=too-many-locals
        self, iso_level: int=32, sample_resolution: int=128,
        limit: float=1.2, batch_size: int=1024):
        """ Output extracted mesh from radiance field
        Args:
            out_dir(str): Path to store output mesh file.
            iso_level(int): Iso-level value for triangulation
            mesh_name(str): output mesh name(in obj.)
            sample_resolution: (int) Sampling resolution for marching cubes,
                increase it for higher level of detail.
            limit:(float) limits in -xyz to xyz for marching cubes 3D grid.
        """

        # define helper function for batchify 3D grid
        def batchify_grids(*data, batch_size=1024, device="cpu"):
            assert all(sample is None or sample.shape[0] == data[0].shape[0] for sample in data), \
                "Sizes of tensors must match for dimension 0."
            # Data size and current batch offset
            size, batch_offset = (data[0].shape[0], 0)
            while batch_offset < size:
                # Subsample slice
                batch_slice = slice(batch_offset, batch_offset + batch_size)
                # Yield each subsample, and move to available device
                yield [sample[batch_slice].to(device) if sample is not None else sample
                    for sample in data]
                batch_offset += batch_size

        sample_resolution = (sample_resolution, ) * 3

        # Create sample tiles
        grid_xyz = [torch.linspace(-limit, limit, num) for num in sample_resolution]

        # Generate 3D samples and flatten it
        grids3d_flat = torch.stack(torch.meshgrid(*grid_xyz), -1).view(-1, 3).float()

        sigmas_samples = []

        # Batchify 3D grids
        for (sampled_grids, ) in batchify_grids(grids3d_flat, batch_size=batch_size, device='cuda'):
            # Query radiance batch
            sigma_batch = self._forward_sigma(points=sampled_grids)
            # Accumulate radiance
            sigmas_samples.append(sigma_batch.cpu())

        # Output Radiance 3D grid (density)
        sigmas = torch.cat(sigmas_samples, 0).view(*sample_resolution).contiguous().detach().numpy()
        # Density boundaries
        min_a, max_a, std_a = sigmas.min(), sigmas.max(), sigmas.std()

        # Adaptive iso level
        iso_level = min(max(iso_level, min_a + std_a), max_a - std_a)
        print(f"Min density {min_a}, Max density: {max_a}, Mean density {sigmas.mean()}")
        print(f"Querying based on iso level: {iso_level}")

        # Marhcing cubes
        vertices, triangles = mcubes.marching_cubes(sigmas, iso_level)
        return vertices / sample_resolution - .5, triangles

    def _forward_sigma(self, points):
        if self.cfg.model.use_fine_mlp:
            return self.mlp_fine(xyz=self.coords_encoding(points))
        return self.mlp_coarse(xyz=self.coords_encoding(points))
