---
license: mit
library_name: pytorch
tags:
- diffusion
- planning
- compositional-generation
- toy-example
---

# Compositional Visual Planning: toy example

Pretrained weights and training data of the toy example (Figure 2) in
[Compositional Visual Planning via Inference-Time Diffusion Scaling](https://arxiv.org/abs/2603.02646)
([project page](https://comp-visual-planning.github.io/)). A short-horizon diffusion model is trained on
circular arc clips. At test time, three 120° arc generators are composed into a three-petal "flower" by
enforcing agreement at the chunk boundaries at every denoising step.

## Files

| File | Description | Size |
| --- | --- | --- |
| `diffusion_420000.ckpt` | `{"model": ..., "model_ema": ...}` state dicts of the DiT1d denoiser (in_dim 60, emb_dim 256, d_model 384, 12 heads, 6 blocks; x0 prediction, 500 cosine diffusion steps) after 420k gradient steps on 8-token clips | 132 MB |
| `toy.npy` | training data, float64 array of shape (64, 600, 2): 64 rotated flowers, each a closed curve of 600 (x, y) points made of three noisy 120° arcs | 0.6 MB |

## Usage

The toy example code downloads both files when they are missing locally. To download them by hand:

```bash
pip install -U huggingface_hub
hf download comp-visual-planning/toy-example diffusion_420000.ckpt --local-dir ckpts
hf download comp-visual-planning/toy-example toy.npy --local-dir data
```

or from python:

```python
from huggingface_hub import hf_hub_download

ckpt = hf_hub_download("comp-visual-planning/toy-example", "diffusion_420000.ckpt", local_dir="ckpts")
data = hf_hub_download("comp-visual-planning/toy-example", "toy.npy", local_dir="data")
```

Then, from the root of the toy example repository, run `python eval.py method=cvp` (ours) or
`python eval.py method=gsc` (DiffCollage / Generative Skill Chaining baseline).

## Citation

```bibtex
@misc{zhang2026compositionalvisualplanninginferencetime,
      title={Compositional Visual Planning via Inference-Time Diffusion Scaling},
      author={Yixin Zhang and Yunhao Luo and Utkarsh Aashu Mishra and Woo Chul Shin and Yongxin Chen and Danfei Xu},
      year={2026},
      eprint={2603.02646},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2603.02646},
}
```
