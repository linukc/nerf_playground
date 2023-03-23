''' Eval script.'''

import hydra
import torch
import trimesh #pylint: disable=import-error
from model.nerf_model import NeRFModel


#pylint: disable = no-value-for-parameter
@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg) -> None:
    """ Entrypoint function."""

    nerf_model = NeRFModel(cfg).to(cfg.training.device)
    nerf_model.eval()

    checkpoint_dict = torch.load('latest.pth',
        map_location=cfg.training.device)
    nerf_model.load_state(checkpoint_dict=checkpoint_dict)

    vertices, triangles = nerf_model.extract_mesh(
        iso_level=32,
        sample_resolution=128,
        limit=1.2)

    mesh = trimesh.Trimesh(vertices, triangles)
    mesh.show()

if __name__ == "__main__":
    main()
