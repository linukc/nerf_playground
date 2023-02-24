"""Section 5.1: Positional encoding"""

from math import pi as PI

import torch
from einops import rearrange

class HighFreqEncoding(torch.nn.Module):
    """Defines a function that embeds
    (p) = (sin(πp2^0), cos(πp2^0), · · · , sin(πp2^(L-1)), cos(πp2^(L-1)))"""

    def __init__(self, num_freqs):
        super().__init__()

        self.num_freqs = num_freqs

        encod_coeff = torch.zeros((2 * self.num_freqs),
            requires_grad=False, dtype=torch.float32)
        freqs = torch.arange(0, self.num_freqs,
            requires_grad=False, dtype=torch.float32)
        encod_coeff[0::2] = PI * 2**(freqs)
        encod_coeff[1::2] = PI * 2**(freqs)

        self.register_buffer("encod_coeff", encod_coeff, persistent=False)

    def forward(self, input_features):
        '''
        x: [num_rays, num_points, 3]
        '''
        input_shape = len(input_features.shape)

        if input_shape == 3:
            n_rays, n_points, _ = input_features.shape
            input_features = rearrange(input_features, "r p f -> (r p) f")

        input_features = input_features.unsqueeze(-1) * self.encod_coeff
        input_features[:, :, 0::2] = torch.sin(input_features[:, :, 0::2])
        input_features[:, :, 1::2] = torch.cos(input_features[:, :, 1::2])

        encoded_values = rearrange(input_features, "b f hf -> b (f hf)")

        if input_shape == 3:
            return encoded_values.view(n_rays, n_points, -1)
        return encoded_values
