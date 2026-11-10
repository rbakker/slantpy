#!/usr/bin/env python3
"""Stand-ins for the NiftyReg / ANTs binaries so the glue code can be tested
without them. They honour the same command lines but only do world-coordinate
(identity) resampling."""
import os
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage

name, args = os.path.basename(sys.argv[0]), sys.argv[1:]


def opt(flag):
    return args[args.index(flag) + 1]


def resample(flo_file, ref_file, out_file, order, dtype):
    flo, ref = nib.load(flo_file), nib.load(ref_file)
    v = np.linalg.inv(flo.affine) @ ref.affine
    data = ndimage.affine_transform(np.asanyarray(flo.dataobj).astype(np.float32), v[:3, :3], v[:3, 3],
                                    output_shape=ref.shape[:3], order=order)
    nib.save(nib.Nifti1Image(data.astype(dtype), ref.affine), out_file)


if name == "reg_aladin":
    resample(opt("-flo"), opt("-ref"), opt("-res"), 1, np.int16)
    np.savetxt(opt("-aff"), np.eye(4))
elif name == "reg_transform":
    i = args.index("-invAff")
    np.savetxt(args[i + 2], np.linalg.inv(np.loadtxt(args[i + 1])))
elif name == "reg_resample":
    resample(opt("-flo"), opt("-ref"), opt("-res"), 0, np.uint8)
elif name == "N4BiasFieldCorrection":
    out, bias = opt("-o").strip("[]").split(",")
    img = nib.load(opt("-i"))
    nib.save(nib.Nifti1Image(np.asanyarray(img.dataobj).astype(np.float32), img.affine), out)
    nib.save(nib.Nifti1Image(np.ones(img.shape, np.float32), img.affine), bias)
elif name == "antsRegistration":
    open(opt("-o") + "0GenericAffine.mat", "w").write("fake")
elif name == "antsApplyTransforms":
    assert os.path.exists(opt("-t")) and opt("-n") == "MultiLabel"
    resample(opt("-i"), opt("-r"), opt("-o"), 0, np.float32)
else:
    sys.exit("unknown fake tool " + name)
