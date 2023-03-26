[![Codacy Badge](https://app.codacy.com/project/badge/Grade/094f9d7eaabc4e07b03ed8de0526862d)](https://www.codacy.com/gh/linukc/nerf_playground/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=linukc/nerf_playground&amp;utm_campaign=Badge_Grade)

To do:
- все должно быть в одном вандб лог, если я хочу строить один график от другого
- fix - when you rename runs in wandb after for example for graphics you lost track to local folder with logs (he continues to be called by the old name like glowing enigma)
- add config as not a decorator, but an argument
- rewrite dataset duplication
- rewrite analyze - func should be independent and out of context of specific dataset 
- implement correct savings https://discuss.pytorch.org/t/loading-optimizer-dict-starts-training-from-initial-lr/36328 (+shed + optim)
- remove use.fine mlp flag
- get rid of duality between mlp.use_viewdir and forward(..., viewdir=None)
- rewrite model ... some part.use to model.use
- add blender one image_dataset
- align train config options with code
- real image dataset (NDC space ablation study, depth not lay in -1 to 1) see ablation study A chapter
- get rid of pylint disables in the code (redesign functions with too many args and statements).

Run:
```python
python3 -m venv venv  
pip3 install requirements.txt
python3 train.py
```
https://dtransposed.github.io/blog/2022/08/06/NeRF/
https://hmn.wiki/nn/Rendering_equation

Thanks to https://github.com/murumura/NeRF-Simple for the code base.
Feel free to highlight my mistakes and make suggestions, this repo is still under construction.
I am looking forward to build my own project based on the default nerf.
