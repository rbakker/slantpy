#!/usr/bin/env python3
"""Which native-space segmentation is geometrically consistent with the network's output?

    python scripts/check_native_mapping.py <slantpy_out>/<name>  seg_a.nii.gz [seg_b.nii.gz ...]

<slantpy_out>/<name> is a slantpy subject folder (it holds orig_target.nii.gz,
MNI-registration/, target_processed_seg.nii.gz).

1. Transform check, independent of any segmentation: the original T1 is pushed
   to the MNI grid with slantpy's matrix convention and compared with the image
   reg_aladin itself wrote (target_MNI.nii.gz). A correlation near 1 means the
   inverse-affine mapping used by --postproc direct is the true one.
2. Round trip: each native-space segmentation is pushed back to the MNI grid
   (nearest neighbour, exact forward affine) and compared with the fused labels
   the networks produced there. Higher = less was lost or shifted on the way out.
"""
import glob
import os
import sys

import numpy as np
from scipy import ndimage
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from compare_with_docker import label_agreement  # noqa: E402
from slantpy.postprocess import native_to_mni_voxel, read_niftyreg_affine  # noqa: E402

sub = sys.argv[1]
t1 = nib.load(glob.glob(os.path.join(sub, "orig_target.nii*"))[0])
mni_img = nib.load(os.path.join(sub, "MNI-registration", "target_MNI.nii.gz"))
fused = nib.load(os.path.join(sub, "target_processed_seg.nii.gz"))
aff = read_niftyreg_affine(os.path.join(sub, "MNI-registration", "target_MNI_aff.mtx"))
# MNI voxel -> native voxel
m2n = np.linalg.inv(native_to_mni_voxel(t1.affine, mni_img.affine, aff))

a = ndimage.affine_transform(np.asanyarray(t1.dataobj).astype(np.float32), m2n[:3, :3], m2n[:3, 3],
                             output_shape=mni_img.shape, order=1)
b = np.asanyarray(mni_img.dataobj).astype(np.float32)
print("1. transform check: correlation of our T1->MNI resampling with reg_aladin's = %.5f"
      % np.corrcoef(a.ravel(), b.ravel())[0, 1])
for shift in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):       # sensitivity: what a 1-voxel error would give
    c = np.corrcoef(np.roll(a, 1, axis=shift.index(1)).ravel(), b.ravel())[0, 1]
    print("   (for scale: shifted by one voxel along axis %d it would be %.5f)" % (shift.index(1), c))

ref = np.asanyarray(fused.dataobj)
print("2. round trip to MNI space, agreement with the fused network labels")
for f in sys.argv[2:]:
    seg = np.rint(np.asanyarray(nib.load(f).dataobj)).astype(np.uint8)
    back = ndimage.affine_transform(seg, m2n[:3, :3], m2n[:3, 3], output_shape=ref.shape, order=0)
    r, _ = label_agreement(ref, back)
    print("   %-60s in-brain agreement %.4f  mean Dice %.4f  min Dice %.4f (%s)"
          % (f[-60:], r["agreement_in_brain"], r["mean_dice"], r["min_dice"], r["min_dice_label"]))
