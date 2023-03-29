""" Camera -> rays utils. """

import torch


def get_ray_dir_cam(height: int, width: int, focal: float):
    """ Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems    
    
    Arguments
    ---------
        height:
            Image height.
        width:
            Image width.
        focal: 
            Focal length.
    
    Returns
    -------
        ray_dir_in_cam:
            The direction of the rays in camera coordinate.
            [H, W, 3]
    """

    i, j = torch.meshgrid(torch.linspace(0, width - 1, width),
                          torch.linspace(0, height - 1, height))
    # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    ray_dir_in_cam = \
        torch.stack([(i-width/2)/focal, -(j-height/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)

    return ray_dir_in_cam

def get_rays_torch(ray_dir_in_cam: torch.Tensor, c2w: torch.Tensor):
    """ 
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Arguments
    ---------
        ray_dir_in_cam:
            Precomputed ray directions in camera coordinate.
            [H, W, 3]
        c2w:
            Transformation matrix from camera coordinate to world coordinate.
            [3, 4]
   
    Returns
    -------
        rays_o:
            The origin of the rays in world coordinate.
            [H*W, 3]
        rays_d:
            The normalized direction of the rays in world coordinate.
            [H*W, 3]
    """

    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = ray_dir_in_cam @ c2w[:, :3].T
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d
