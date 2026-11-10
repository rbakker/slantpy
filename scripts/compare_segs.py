#!/usr/bin/env python3
"""Agreement between two label images on the same grid.

    python scripts/compare_segs.py a_seg.nii.gz b_seg.nii.gz
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_with_docker import label_agreement, load  # noqa: E402

a, b = load(sys.argv[1]), load(sys.argv[2])
if a.shape != b.shape:
    sys.exit("different grids: %s vs %s" % (a.shape, b.shape))
res, _ = label_agreement(a, b)
for k, v in res.items():
    print("%-24s %s" % (k, ("%.6g" % v) if isinstance(v, float) else v))
