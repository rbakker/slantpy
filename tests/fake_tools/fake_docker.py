#!/usr/bin/env python3
"""A stand-in for the docker CLI, for testing scripts/compare_with_docker.py.
'docker run' produces an OUTPUTS tree with the *original* pipeline's layout,
using slantpy itself plus the stub tools."""
import os
import shutil
import subprocess
import sys

import nibabel as nib
import numpy as np

args = sys.argv[1:]
if args[0] == "stats":
    print("1.5GiB / 8GiB")
elif args[0] == "run":
    mounts = dict(reversed(a.split(":")) for a in (args[i + 1] for i, x in enumerate(args) if x == "-v"))
    inputs, outputs = mounts["/INPUTS/"], mounts["/OUTPUTS/"]
    extra = os.environ["FAKE_SLANT_EXTRA"]
    for f in sorted(os.listdir(inputs)):
        name = f[:-7]
        subprocess.check_call([sys.executable, "-m", "slantpy", "run", os.path.join(inputs, f), outputs,
                               "--extra-dir", extra, "--device", "cpu", "--keep-tiles", "--no-pdf",
                               "--postproc", "legacy", "--forward", "literal"])
        sub = os.path.join(outputs, name)
        ap = os.path.join(sub, "working_dir", "all_piece")
        for t in os.listdir(os.path.join(sub, "tiles")):
            d = os.path.join(ap, t[:-4], "test_out", "seg_output", "epoch_0027", "seg")
            os.makedirs(d)
            cls = np.transpose(np.load(os.path.join(sub, "tiles", t)), (2, 0, 1))
            nib.save(nib.Nifti1Image(cls, np.eye(4)), os.path.join(d, "target_processed_seg.nii.gz"))
        d = os.path.join(ap, "5000_fusion27_mv", "test_out", "seg_output", "epoch_0027", "seg_orig_final")
        os.makedirs(d)
        shutil.move(os.path.join(sub, "target_processed_seg.nii.gz"), d)
        shutil.rmtree(os.path.join(sub, "tiles"))
else:
    sys.exit("fake docker: unsupported " + " ".join(args))
