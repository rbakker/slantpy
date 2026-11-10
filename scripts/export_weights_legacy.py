"""Fallback only: dump the released checkpoints to .npz with the image's own
(Python 2 / PyTorch 0.2) interpreter, in case a current PyTorch cannot read the
old .pth files directly.

    docker run --rm -v $PWD/scripts:/s -v /path/to/extra/model_dir:/out pipeline/slant:v1 \
        /pythondir/miniconda/bin/python /s/export_weights_legacy.py /extra/model_dir /out

slantpy picks up model_epoch_0027.npz when the .pth next to it is absent.
"""
from __future__ import print_function
import os
import sys

import numpy as np
import torch

src, dst = sys.argv[1], sys.argv[2]
for piece in sorted(os.listdir(src)):
    rel = os.path.join(piece, "finetune_out", "models", "model_epoch_0027")
    pth = os.path.join(src, rel + ".pth")
    if not os.path.exists(pth):
        continue
    state = torch.load(pth, map_location=lambda storage, loc: storage)
    out = os.path.join(dst, rel + ".npz")
    if not os.path.isdir(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out))
    np.savez(out, **dict((k, v.cpu().numpy()) for k, v in state.items()))
    print("wrote", out)
