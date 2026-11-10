"""Pre-processing: port of preproc_pipline.m (rigidMNI, run_N4correction, process_target).

    T1 --reg_aladin--> MNI305 grid --N4--> intensity normalisation --> target_processed.nii.gz
"""
import json
import os

import nibabel as nib
import numpy as np

from .labels import MNI_SHAPE
from .robustfit import robustfit_huber

ATLAS_VARS = ("brain_seg", "norm_regressand")
ATLAS_QA_VARS = ("model_seg", "meanvec", "eigenvecs")


def load_atlas_model(path, variables=ATLAS_VARS):
    """Read selected variables from atlas_model.mat (v5/v7 or v7.3) or a .npz extract.

    The Matlab code loads every variable in the file, including the full
    atlas data matrix; only ``brain_seg`` and ``norm_regressand`` are used.
    """
    if path.endswith(".npz"):
        with np.load(path) as z:
            return {k: z[k] for k in variables if k in z}
    try:
        from scipy.io import loadmat
        m = loadmat(path, variable_names=list(variables))
        return {k: np.asarray(m[k]) for k in variables if k in m}
    except NotImplementedError:  # Matlab v7.3 = HDF5, stored with reversed axes
        import h5py
        with h5py.File(path, "r") as f:
            return {k: np.transpose(f[k][()]) for k in variables if k in f}


def extract_atlas_model(mat_path, npz_path):
    """One-off: save the two small arrays the pipeline needs to a compressed .npz."""
    m = load_atlas_model(mat_path)
    np.savez_compressed(npz_path, brain_seg=(m["brain_seg"] > 0),
                        norm_regressand=np.asarray(m["norm_regressand"], np.float64).ravel())
    return npz_path


def register_to_mni(target, sub_dir, tools):
    """rigidMNI(): affine (reg_aladin default, 12 dof) registration to the MNI305 template."""
    reg_dir = os.path.join(sub_dir, "MNI-registration")
    tmp_dir = os.path.join(sub_dir, "temp-out")
    os.makedirs(reg_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    log = os.path.join(tmp_dir, "MNI-registration.txt")
    res = os.path.join(reg_dir, "target_MNI.nii.gz")
    aff = os.path.join(reg_dir, "target_MNI_aff.mtx")
    inv = os.path.join(reg_dir, "target_MNI_aff_inv.mtx")
    mounts = [sub_dir]          # ``target`` is the copy inside sub_dir (see pipeline.run_subject)

    if not (os.path.exists(res) and os.path.exists(aff)):
        tools.run([tools.niftyreg("reg_aladin"), "-ref", tools.mni, "-flo", target,
                   "-aff", aff, "-res", res, "-rmask", tools.mni_mask], log=log, mounts=mounts)
        # "fix any obvious problems with the nifty file": store as plain float32
        img = nib.load(res)
        data = np.asanyarray(img.dataobj).astype(np.float32)
        hdr = img.header.copy()
        hdr.set_data_dtype(np.float32)
        hdr.set_slope_inter(1, 0)
        nib.save(nib.Nifti1Image(data, img.affine, hdr), res)
    if not os.path.exists(inv):
        tools.run([tools.niftyreg("reg_transform"), "-ref", res, "-invAff", aff, inv],
                  log=log, mounts=mounts)
    return res, aff, inv


def n4_correct(image, sub_dir, tools):
    """run_N4correction(): N4BiasFieldCorrection with default options, bias field kept."""
    corr_dir = os.path.join(sub_dir, "N4-targets")
    tmp_dir = os.path.join(sub_dir, "temp-out", "N4-target_MNI")
    os.makedirs(corr_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    out = os.path.join(corr_dir, "target_MNI.nii.gz")
    bias = os.path.join(corr_dir, "target_MNI_bias.nii.gz")
    if not (os.path.exists(out) and os.path.exists(bias)):
        tools.run([tools.ants("N4BiasFieldCorrection"), "-d", "3", "-i", image,
                   "-o", "[%s,%s]" % (out, bias)],
                  log=os.path.join(tmp_dir, "N4-txtout.txt"), mounts=[sub_dir])
    return out


def normalise_intensity(data, brain_seg, norm_regressand):
    """Intensity part of process_target(): z-score inside the brain mask, then a
    Huber-robust linear fit of the sorted intensities to the atlas' sorted mean
    intensities. Returns (float32 volume, betas)."""
    mask = np.asarray(brain_seg) > 0
    if mask.shape != data.shape:
        raise ValueError("brain_seg shape %s does not match image shape %s"
                         % (mask.shape, data.shape))
    data = data.astype(np.float64)
    vals = data[mask]
    data = (data - vals.mean()) / vals.std(ddof=1)          # Matlab std() is N-1
    targetvec = np.sort(data[mask])
    betas = robustfit_huber(targetvec, np.asarray(norm_regressand, np.float64).ravel())
    return (data * betas[1] + betas[0]).astype(np.float32), betas


def model_likelihood(normed, atlas_path):
    """The (purely informational) PCA 'likelihood' that process_target() prints."""
    m = load_atlas_model(atlas_path, ATLAS_QA_VARS)
    sel = (m["model_seg"] > 0).ravel(order="F")
    vec = normed.astype(np.float64).ravel(order="F")[sel]
    mean = m["meanvec"].astype(np.float64).ravel()
    ev = m["eigenvecs"].astype(np.float64)
    recon = mean + ev @ (ev.T @ (vec - mean))
    return float(np.exp(np.corrcoef(vec, recon)[0, 1] - 1))


def preprocess(target, sub_dir, tools, atlas_model, n4=True, log=print):
    """Full pre-processing of one T1. Returns the path of target_processed.nii.gz.

    ``n4=False`` skips the N4 bias-field correction, for images that were
    already corrected upstream; the registered image is normalised directly and
    ANTs is not needed at all.
    """
    os.makedirs(sub_dir, exist_ok=True)
    out = os.path.join(sub_dir, "target_processed.nii.gz")
    settings_file = os.path.join(sub_dir, "target_processed.json")
    settings = {"n4": bool(n4)}
    if os.path.exists(out):
        # an existing result made with other settings must not be reused silently
        previous = json.load(open(settings_file)) if os.path.exists(settings_file) else {"n4": True}
        if previous != settings:
            raise RuntimeError(
                "%s was made with N4 bias correction %s, but this run asks for it %s. "
                "Use a new output folder or --clean." % (
                    out, "on" if previous["n4"] else "off", "on" if n4 else "off"))
        log("pre-processing: %s exists, skipping" % out)
        return out
    log("registering to MNI305 (reg_aladin)")
    res, _, _ = register_to_mni(target, sub_dir, tools)
    if n4:
        log("N4 bias field correction")
        corr = n4_correct(res, sub_dir, tools)
    else:
        log("N4 bias field correction skipped (--skip-n4)")
        corr = res

    img = nib.load(corr)
    data = np.asanyarray(img.dataobj)
    if data.shape != MNI_SHAPE:
        raise ValueError("registered image has shape %s, expected %s" % (data.shape, MNI_SHAPE))
    atlas = load_atlas_model(atlas_model)
    normed, betas = normalise_intensity(data, atlas["brain_seg"], atlas["norm_regressand"])
    log("robust (huber) regression parameters: %f %f" % (betas[0], betas[1]))
    hdr = img.header.copy()
    hdr.set_data_dtype(np.float32)
    hdr.set_slope_inter(1, 0)
    tmp = out[:-7] + ".tmp.nii.gz"
    nib.save(nib.Nifti1Image(normed, img.affine, hdr), tmp)
    os.replace(tmp, out)
    with open(settings_file, "w") as fh:
        json.dump(settings, fh)
    return out
