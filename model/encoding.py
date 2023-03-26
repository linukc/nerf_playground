"""Section 5.1: Positional encoding"""

import torch
from einops import rearrange

class HighFreqEncoding(torch.nn.Module):
    """Defines a function that embeds
    (p) = (sin(p2^0), cos(p2^0), · · · , sin(p2^(L-1)), cos(p2^(L-1)))
    Comment abount pi missing: https://github.com/bmild/nerf/issues/12
    """

    def __init__(self, num_freqs):
        super().__init__()

        self.num_freqs = num_freqs

        encod_coeff = torch.zeros((2 * self.num_freqs),
            requires_grad=False, dtype=torch.float32)
        freqs = torch.arange(0, self.num_freqs,
            requires_grad=False, dtype=torch.float32)
        encod_coeff[0::2] = 2**(freqs)
        encod_coeff[1::2] = 2**(freqs)

        self.register_buffer("encod_coeff", encod_coeff, persistent=False)

    def forward(self, in_feat):
        '''
        x: [num_rays, num_points, 3] or [points 2]
        '''
        if len(in_feat.shape) == 3:
            n_rays, n_points, _ = in_feat.shape
            input_features = rearrange(in_feat, "r p f -> (r p) f")
        else:
            input_features = in_feat.clone()

        encoded_features = input_features.unsqueeze(-1) * self.encod_coeff
        encoded_features[:, :, 0::2] = torch.sin(encoded_features[:, :, 0::2])
        encoded_features[:, :, 1::2] = torch.cos(encoded_features[:, :, 1::2])
        # create from b 3 20 with structure
        # x1x2x3...,
        # y1y2y3...,
        # z1z2z3...
        # b 20 3 with structre
        # x1y1z1
        # x2y2z2
        # ...
        encoded_features = torch.transpose(encoded_features, 1, 2).reshape(
            input_features.shape[0], -1)
        encoded_features = torch.cat((input_features, encoded_features), dim=1)

        if len(in_feat.shape) == 3:
            return encoded_features.view(n_rays, n_points, -1)
        return encoded_features
