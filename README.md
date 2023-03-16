
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/4587e660207746389d51e5f31af0c72a)](https://app.codacy.com/gh/linukc/nerf_playground?utm_source=github.com&utm_medium=referral&utm_content=linukc/nerf_playground&utm_campaign=Badge_Grade_Settings)

To do:
- real image dataset (NDC space ablation study, depth not lay in -1 to 1) see ablation study A chapter
- get rid of pylint disables in the code (redesign functions with too many args and statements).

Run:
```python
python3 -m venv venv  
pip3 install requirements.txt
python3 train.py -m dataset.path=<path_to_scene in blender dataset> (check config before)
```
https://dtransposed.github.io/blog/2022/08/06/NeRF/
https://hmn.wiki/nn/Rendering_equation

Thanks to https://github.com/murumura/NeRF-Simple for the code base.
Feel free to highlight my mistakes and make suggestions, this repo is still under construction.
I am looking forward to build my own project based on the default nerf.
