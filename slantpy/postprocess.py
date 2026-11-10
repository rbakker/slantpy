"""Bringing the fused MNI-space segmentation back to the original T1 grid.

``legacy_to_native``  reproduces apply_aladin_affine_via_ants.m command by command.
``direct_to_native``  is the proposed replacement: invert the affine that
                      reg_aladin already produced and resample once, in numpy.
"""
import os
import shutil

import nibabel as nib
import numpy as np
from scipy import ndimage


# --------------------------------------------------------------------------- legacy
def legacy_to_native(seg_mni, t1_mni, t1_orig, out_file, work_dir, tools, mount_dir=None):
    """What the Matlab pipeline does:

    1. reg_aladin   : register the *processed MNI-space* T1 back to the original T1
                      (a second, independent registration)
    2. reg_resample : nearest-neighbour resample the labels with that affine
    3. antsRegistration : register the MNI-space label image to the resampled
                      label image (mean squares on label numbers, rigid then
                      affine) only to obtain the same transform in ITK format
    4. antsApplyTransforms -n MultiLabel with the transform from step 3
    """
    reg_dir = os.path.join(work_dir, "inverse-MNI-registration")
    tmp_dir = os.path.join(work_dir, "temp")
    os.makedirs(reg_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    out_bname = os.path.join(reg_dir, "inverse-MNI-reg")
    aff_mat = out_bname + "0GenericAffine.mat"
    tmp_t1 = os.path.join(reg_dir, "tmp-target.nii.gz")
    tmp_seg = os.path.join(reg_dir, "tmp-target-seg.nii.gz")
    tmp_aff = os.path.join(reg_dir, "tmp-affine.txt")
    log = os.path.join(tmp_dir, "inverse-MNI-reg-output.txt")
    # All inputs must already be inside ``mount_dir`` (the subject's output
    # folder); the result is written there too and moved afterwards, so a
    # containerised tool never needs any other host directory.
    mount_dir = os.path.abspath(mount_dir or work_dir)
    for p in (seg_mni, t1_mni, t1_orig, work_dir):
        if os.path.commonpath([mount_dir, os.path.abspath(p)]) != mount_dir:
            raise ValueError("%s is outside %s" % (p, mount_dir))
    mounts = [mount_dir]
    final_file, out_file = out_file, os.path.join(reg_dir, "result_seg.nii.gz")

    tools.run([tools.niftyreg("reg_aladin"), "-ref", t1_orig, "-flo", t1_mni,
               "-res", tmp_t1, "-aff", tmp_aff], log=log, mounts=mounts)
    tools.run([tools.niftyreg("reg_resample"), "-ref", t1_orig, "-flo", seg_mni,
               "-aff", tmp_aff, "-res", tmp_seg, "-inter", "0"], log=log, mounts=mounts)
    tools.run([tools.ants("antsRegistration"), "-d", "3", "-o", out_bname,
               "-u", "-r", "[%s,%s,1]" % (tmp_seg, seg_mni),
               "-m", "MeanSquares[%s,%s,1,3,Regular,0.5]" % (tmp_seg, seg_mni),
               "-t", "Rigid[0.5]", "-c", "1000x1000x1000", "-f", "6x4x2", "-s", "8x8x8",
               "-m", "MeanSquares[%s,%s,1,3,Regular,0.25]" % (tmp_seg, seg_mni),
               "-t", "Affine[0.5]", "-c", "1000x1000x10", "-f", "4x2x1", "-s", "2x1x0"],
              log=log, mounts=mounts)
    tools.run([tools.ants("antsApplyTransforms"), "-d", "3", "-i", seg_mni, "-o", out_file,
               "-r", t1_orig, "-n", "MultiLabel", "-t", aff_mat], log=log, mounts=mounts)
    shutil.move(out_file, final_file)
    return final_file


# --------------------------------------------------------------------------- direct
def read_niftyreg_affine(path):
    m = np.loadtxt(path)
    if m.shape != (4, 4):
        raise ValueError("%s is not a 4x4 affine" % path)
    return m


def native_to_mni_voxel(orig_affine, mni_affine, aff_ref_to_flo):
    """4x4 matrix taking original-T1 voxel indices to MNI-grid voxel indices.

    reg_aladin's matrix maps world coordinates of the reference (MNI) to world
    coordinates of the floating image (original T1), so
        mni_voxel = inv(A_mni) @ inv(aff) @ A_orig @ orig_voxel
    """
    return np.linalg.inv(mni_affine) @ np.linalg.inv(aff_ref_to_flo) @ orig_affine


def _bbox(mask):
    out = []
    for ax in range(3):
        other = tuple(a for a in range(3) if a != ax)
        idx = np.flatnonzero(mask.any(axis=other))
        out.append((int(idx[0]), int(idx[-1]) + 1))
    return out


def resample_labels(seg, vox2vox, out_shape, mode="linear", sigma=1.0, truncate=4.0):
    """Resample a label volume. ``vox2vox`` maps output voxels to ``seg`` voxels.

    mode='nearest' : nearest neighbour.
    mode='linear'  : every label's indicator is interpolated trilinearly, the
                     largest wins (ties: lowest label).
    mode='gaussian': as 'linear' after smoothing each indicator with a Gaussian
                     of ``sigma`` voxels - an approximation of ANTs' MultiLabel.

    Labels are processed one at a time and only inside the part of the output
    grid they can reach, so memory is two float32 output volumes however many
    labels there are.
    """
    M, off = vox2vox[:3, :3], vox2vox[:3, 3]
    if mode == "nearest":
        return ndimage.affine_transform(seg, M, off, output_shape=out_shape, order=0,
                                        mode="constant", cval=0)
    if mode not in ("linear", "gaussian"):
        raise ValueError("unknown interpolation mode %r" % mode)
    inv = np.linalg.inv(vox2vox)
    pad = 2 + (int(np.ceil(sigma * truncate)) if mode == "gaussian" else 0)
    best = np.zeros(out_shape, np.float32)
    out = np.zeros(out_shape, seg.dtype)
    for lab in np.unique(seg):
        mask = seg == lab
        if lab == 0:
            # background: whole grid, and everything outside the MNI field of view
            src, lo, hi, cval = mask.astype(np.float32), np.zeros(3, int), np.array(out_shape), 1.0
        else:
            bb = _bbox(mask)
            sl = tuple(slice(max(a - pad, 0), min(b + pad, n)) for (a, b), n in zip(bb, seg.shape))
            src = mask[sl].astype(np.float32)
            s0 = np.array([s.start for s in sl], float)
            # corners of the source box -> output voxel space
            corners = np.array([[x, y, z, 1.0] for x in (sl[0].start, sl[0].stop)
                                for y in (sl[1].start, sl[1].stop)
                                for z in (sl[2].start, sl[2].stop)])
            oc = (inv @ corners.T)[:3]
            lo = np.clip(np.floor(oc.min(1)).astype(int) - 1, 0, out_shape)
            hi = np.clip(np.ceil(oc.max(1)).astype(int) + 2, 0, out_shape)
            if np.any(hi <= lo):
                continue
            cval = 0.0
        if mode == "gaussian":
            src = ndimage.gaussian_filter(src, sigma, mode="nearest" if lab == 0 else "constant",
                                          truncate=truncate)
        if lab == 0:
            sub_off = off
        else:
            sub_off = M @ lo + off - s0
        w = ndimage.affine_transform(src, M, sub_off, output_shape=tuple(hi - lo), order=1,
                                     mode="constant", cval=cval, output=np.float32)
        region = tuple(slice(a, b) for a, b in zip(lo, hi))
        better = w > best[region]
        best[region][better] = w[better]
        out[region][better] = lab
    return out


def direct_to_native(seg_mni, aff_file, t1_orig, out_file, mode="linear"):
    """Resample the MNI-space labels onto the original T1 grid using the inverse
    of the forward affine (``target_MNI_aff.mtx``). No registration, no ANTs."""
    seg_img = nib.load(seg_mni)
    seg = np.asanyarray(seg_img.dataobj).astype(np.uint8)
    orig = nib.load(t1_orig)
    shape = tuple(orig.shape[:3])
    v2v = native_to_mni_voxel(orig.affine, seg_img.affine, read_niftyreg_affine(aff_file))
    out = resample_labels(seg, v2v, shape, mode=mode)
    hdr = orig.header.copy()
    hdr.set_data_shape(shape)
    hdr.set_data_dtype(np.uint8)
    hdr.set_slope_inter(1, 0)
    hdr["cal_min"], hdr["cal_max"] = 0, 0
    nib.save(nib.Nifti1Image(out, orig.affine, hdr), out_file)
    return out_file
