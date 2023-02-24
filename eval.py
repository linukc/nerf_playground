''' Eval script.'''

import hydra
import torch
import trimesh #pylint: disable=import-error
from model.nerf_model import NeRFModel


#pylint: disable = no-value-for-parameter
@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg) -> None:
    """ Entrypoint function."""

    model = NeRFModel(**cfg.model).to(cfg.device)
    model.eval_mode()

    checkpoint_dict = torch.load('checkpoints/checkpoint_i0_s1000.pth', map_location=cfg.device)
    model.load_state(checkpoint_dict=checkpoint_dict)

    vertices, triangles = model.extract_mesh(
        out_dir='meshs',
        mesh_name='test',
        iso_level=32,
        sample_resolution=128,
        limit=1.2
    )

    mesh = trimesh.Trimesh(vertices, triangles)
    mesh.show()

if __name__ == "__main__":
    main()
