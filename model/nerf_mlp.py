"""NeRF MLP"""

import torch
from torch import nn


class NeRFMLP(nn.Module):
    """NeRF MLP. Section Additional Implementation Details.

    Parameters
    ----------
    See configs/model/default.yaml/mlp.
    """

    def __init__(self, cfg_model):
        super().__init__()

        self.skip_connection_layers = list(map(int,
            cfg_model.mlp.skip_connection_layers.split(",")))
        base_layer_num = cfg_model.mlp.base_layer_num
        base_features_size = cfg_model.mlp.base_features_size

        if cfg_model.encoding.use:
            in_features_location = cfg_model.encoding.num_freqs_coords * 3 * 2 + 3
            in_features_direction = cfg_model.encoding.num_freqs_viewdir * 3 * 2 + 3
        else:
            in_features_location = 3
            in_features_direction = 3

        self.mlp_base = nn.ModuleList([
            nn.Sequential(torch.nn.Linear(base_features_size,
                                          base_features_size),
                          torch.nn.ReLU())
                for i in range(base_layer_num)])

        for i in range(base_layer_num):
            if i == 0:
                self.mlp_base[i] = nn.Sequential(
                    torch.nn.Linear(in_features_location,
                                    base_features_size),
                    torch.nn.ReLU())
            elif i in self.skip_connection_layers:
                self.mlp_base[i] = nn.Sequential(
                    torch.nn.Linear(base_features_size + in_features_location,
                                    base_features_size),
                    torch.nn.ReLU())

        self.density = torch.nn.Linear(base_features_size, 1)
        self.dense_mlp = torch.nn.Linear(base_features_size, base_features_size)
        self.color = nn.Sequential(
            torch.nn.Linear(base_features_size+in_features_direction, base_features_size//2),
            torch.nn.ReLU(),
            torch.nn.Linear(base_features_size//2, 3),
            torch.nn.Sigmoid())

    def forward(self, xyz: torch.Tensor, viewdirs: torch.Tensor=None):
        """MLP forward propogation function.

        Parameters
        ----------
        xyz: torch.Tensor
            Input location. (ray_count, num_samples, 3)
        viewdirs: torch.Tensor
            Input direction view. (ray_count, num_samples, 3)

        Returns
        -------
        if not use_viewdir:
            torch.tensor(ray_count, num_samples, 1)
                Density value in particular point.
        else:
            torch.tensor(ray_count, num_samples, 4)
                Color value in particular point + density value.
        """

        input_xyz = xyz
        for i, layer in enumerate(self.mlp_base):
            if i in self.skip_connection_layers:
                xyz = torch.cat((input_xyz, xyz), dim=-1)
            xyz = layer(xyz)

        density = self.density(xyz)
        if viewdirs is None:
            return density

        xyz = self.dense_mlp(xyz)
        color = self.color(torch.cat((xyz, viewdirs), dim=-1))

        return torch.cat((color, density), dim=-1)
