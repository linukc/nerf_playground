"""Class to sample points along the ray"""

import torch


def intervals_to_ray_points(point_intervals: torch.Tensor,
                            ray_directions: torch.Tensor,
                            ray_origins: torch.Tensor):
    """ Based on ray position('ray_origins') and noraml orientation('ray_directions')
        and intervals('point_intervals'), calculate each point on the ray.

    Parameters
    ----------
        point_intervals: torch.tensor(ray_count, num_samples)
        ray_directions: torch.tensor(ray_count, 3)
        ray_origin: torch.tensor(ray_count, 3)

    Returns
    -------
        ray_points: torch.tensor(ray_count, num_samples, 3)
            Samples points along each ray.
    """

    # https://pytorch.org/docs/stable/notes/broadcasting.html see 3 rules
    ray_points = ray_origins[..., None, :] + \
        ray_directions[..., None, :] * point_intervals[..., None]

    return ray_points

class IntervalSampler(torch.nn.Module):
    """Class to define intervals between samples (points) on the ray."""

    #pylint: disable=too-many-arguments
    def __init__(self, num_samples_coarse: int, perturb: bool, lindisp: bool,
        near_bound:int, far_bound: int):

        super().__init__()

        self.num_samples = num_samples_coarse
        self.perturb = perturb
        self.lindisp = lindisp
        self.near = near_bound
        self.far = far_bound

        # 1 x num_samples
        point_intervals = torch.linspace(0.0, 1.0, self.num_samples,
                                         requires_grad=False, dtype=torch.float32)[None, :]
        self.register_buffer("point_intervals", point_intervals, persistent=False)

    def forward(self, ray_count: int):
        """
        Parameters
        ----------
        ray_count: int
            Number of rays (batch_size).

        Returns
        -------
        point_intervals: torch.Tensor(ray_count, self.num_samples)
            Depths of the sampled points along ray from near to far bounds.
        """

        # Sample in disparity space, as opposed to in depth space. Sampling in disparity is
        # nonlinear when viewed as depth sampling! (The closer to the camera the more samples)
        if not self.lindisp:
            point_intervals = self.point_intervals * (self.far - self.near) + self.near
        else:
            raise NotImplementedError
            # point_intervals = 1.0 / (1.0 / near * (1.0 - self.point_intervals) +
            #   1.0 / far * self.point_intervals)

        point_intervals = point_intervals.expand([ray_count, self.num_samples])

        if self.perturb:
            # Get intervals between samples.
            mids = 0.5 * (point_intervals[..., 1:] + point_intervals[..., :-1])
            upper = torch.cat((mids, point_intervals[..., -1:]), dim=-1)
            lower = torch.cat((point_intervals[..., :1], mids), dim=-1)

            # Stratified samples in those intervals.
            t_rand = torch.rand(
                point_intervals.shape,
                dtype=point_intervals.dtype,
                device=point_intervals.device,
            )
            point_intervals = lower + (upper - lower) * t_rand

        return point_intervals

class HierarchicalPDFSampler(torch.nn.Module):
    """Module that perform Hierarchical sampling (section 5.2)
    Args:
        num_fine_samples (int): Number of depth samples per ray for the fine network.
    """

    def __init__(self, num_fine_samples: int):
        super().__init__()

        self.num_fine_samples = num_fine_samples
        uniform_x = torch.linspace(0.0, 1.0, steps=self.num_fine_samples,
            requires_grad = False, dtype=torch.float32)
        self.register_buffer("uniform_x", uniform_x, persistent=False)

    def forward(self, depth_rays_values_coarse, coarse_weights, perturb=True):
        """
            Inputs:
                depth_rays_values_coarse: (ray_count, num_coarse_samples])
                    Depth values of each sampled point along the ray.
                coarse_weights: (ray_count, num_coarse_samples])
                    Weights assigned to each sampled color of sampled point along the ray.
                perturb:
                    (bool) if True, perform stratified sampling, otherwise perform uniform sampling.
            Outputs:
                depth_values_fine: (ray_count, num_coarse_samples + num_fine_samples)
                    Depths of the hierarchical sampled points along the ray.
        """
        points_on_rays_mid = 0.5 * (depth_rays_values_coarse[..., 1:] + \
            depth_rays_values_coarse[..., :-1])
        interval_samples = self.sample_pdf(points_on_rays_mid,
                                coarse_weights[..., 1:-1],
                                self.uniform_x,
                                perturb=perturb).detach()

        depth_values_fine, _ = torch.sort(torch.cat((depth_rays_values_coarse,
                                                     interval_samples), dim=-1),
                                            dim=-1)
        return depth_values_fine

    #pylint: disable=too-many-locals
    def sample_pdf(self, bins, weights, uniform_x, perturb):
        """Hierarchical sampling (section 5.2) for fine sampling
            implementation by yenchenlin (https://github.com/yenchenlin/nerf-pytorch).
        Inputs:
            bins: [ray_count, num_coarse_samples - 1]
                Points_on_rays_mid.
            weights: [ray_count, num_coarse_samples - 2]
                Weights assigned to each sampled color exclude first and last one.
            uniform_x:
                A one-dimensional tensor of size 'num_fine_samples' whose values are
                evenly spaced from 0 to 1, inclusive.
            perturb: (bool)
                if True, perform stratified sampling, otherwise perform uniform sampling.
        Outputs:
            samples: [ray_count, num_coarse_samples + num_fine_samples]
                Depths of the hierarchical sampled points along the ray.
        """

        weights = weights + 1e-5
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)  # [ray_count, num_coarse_sample-2]
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat(
            [torch.zeros_like(cdf[..., :1]), cdf], dim=-1
        )  # [ray_count, num_coarse_sample-1] = [ray_count, len(bins)]

        # Take uniform samples
        if not perturb:  # cdf.shape[:-1] get cdf shape exclude last one
            uniform_x = uniform_x.expand(list(cdf.shape[:-1]) + \
                [self.num_fine_samples])  # (ray_count, num_fine_samples)
        else:
            uniform_x = torch.rand(
                list(cdf.shape[:-1]) + [self.num_fine_samples],
                dtype=weights.dtype,
                device=weights.device,
            )

        # Invert CDF
        uniform_x = uniform_x.contiguous().detach()
        cdf = cdf.contiguous().detach()

        inds = torch.searchsorted(cdf, uniform_x, right=True)  # (ray_count, num_fine_samples)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack((below, above), dim=-1)  # (ray_count, num_fine_samples, 2)

        # (ray_count, num_fine_samples, num_coarse_sample-1)
        matched_shape = (inds_g.shape[0], inds_g.shape[1], cdf.shape[-1])
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        trans = (uniform_x - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + trans * (bins_g[..., 1] - bins_g[..., 0])

        return samples
