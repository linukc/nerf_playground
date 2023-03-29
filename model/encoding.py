""" Section 5.1: Positional encoding. """

import torch
from einops import rearrange


class HighFreqEncoding(torch.nn.Module):
    """
    Defines a function that embeds
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
        """ Encoding. """

        n_rays, n_points, _ = in_feat.shape
        input_features = rearrange(in_feat, "r p f -> (r p) f")

        encoded_features = input_features.unsqueeze(-1) * self.encod_coeff
        encoded_features[:, :, 0::2] = torch.sin(encoded_features[:, :, 0::2])
        encoded_features[:, :, 1::2] = torch.cos(encoded_features[:, :, 1::2])
        encoded_features = torch.transpose(encoded_features, 1, 2).reshape(
            input_features.shape[0], -1)
        encoded_features = torch.cat((input_features, encoded_features), dim=1)

        return encoded_features.view(n_rays, n_points, -1)
