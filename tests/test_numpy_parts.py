import os
import numpy as np
import nibabel as nib
from scipy import ndimage

from slantpy.inference import fuse_majority, tile_input
from slantpy.labels import LABELS, MNI_SHAPE, PIECE_MAP, PIECES, TILE_SHAPE
from slantpy.postprocess import (direct_to_native, native_to_mni_voxel, resample_labels)
from slantpy.preprocess import normalise_intensity
from slantpy.volumes import label_volumes


def test_tiles_cover_volume():
    cover = np.zeros(MNI_SHAPE, int)
    for p in PIECES:
        x0, x1, y0, y1, z0, z1 = PIECE_MAP[p]
        assert (x1 - x0, y1 - y0, z1 - z0) == TILE_SHAPE
        cover[x0:x1, y0:y1, z0:z1] += 1
    assert len(PIECES) == 27 and cover.min() >= 1 and cover.max() == 27


def test_fusion_matches_bruteforce_mode():
    rng = np.random.default_rng(1)
    tiles = [(p, rng.integers(0, 5, TILE_SHAPE).astype(np.uint8)) for p in PIECES]
    fused = fuse_majority(tiles)
    # reference: explicit per-class counting, lowest class wins ties (Matlab mode)
    best = np.zeros(MNI_SHAPE, int)
    ref = np.zeros(MNI_SHAPE, int)
    for c in range(5):
        cnt = np.zeros(MNI_SHAPE, int)
        for p, t in tiles:
            x0, x1, y0, y1, z0, z1 = PIECE_MAP[p]
            cnt[x0:x1, y0:y1, z0:z1] += t == c
        upd = cnt > best
        best[upd], ref[upd] = cnt[upd], c
    lut = np.array((0,) + LABELS)
    assert np.array_equal(fused, lut[ref])


def test_tile_input_layout():
    vol = np.random.default_rng(0).normal(size=MNI_SHAPE).astype(np.float32)
    t = tile_input(vol, "2_3_1")
    assert t.shape == (88, 96, 128) and t.dtype == np.float32
    assert t.min() == 0 and abs(t.max() - 255) < 1e-3
    assert np.argmax(t) == np.ravel_multi_index(
        np.array(np.unravel_index(np.argmax(vol[38:134, 92:220, 0:88]), TILE_SHAPE))[[2, 0, 1]], t.shape)


def test_normalise_intensity_recovers_linear_map():
    rng = np.random.default_rng(2)
    mask = np.zeros((20, 20, 20), bool); mask[4:16, 4:16, 4:16] = True
    ref = np.sort(rng.normal(size=mask.sum()))            # atlas sorted intensities
    img = rng.normal(size=mask.shape)
    img[mask] = rng.permutation(ref) * 37.0 + 500.0       # same distribution, other scale
    out, betas = normalise_intensity(img, mask, ref)
    assert np.allclose(np.sort(out[mask]), ref, atol=1e-4)


def _blobs(shape, n=6, seed=3):
    rng = np.random.default_rng(seed)
    seg = np.zeros(shape, np.uint8)
    g = np.indices(shape)
    for i in range(1, n + 1):
        c = [rng.uniform(0.25, 0.75) * s for s in shape]
        r = rng.uniform(4, 8)
        seg[sum((gi - ci) ** 2 for gi, ci in zip(g, c)) < r * r] = LABELS[i * 7]
    return seg


def test_resample_identity_and_bruteforce():
    seg = _blobs((40, 44, 36))
    assert np.array_equal(resample_labels(seg, np.eye(4), seg.shape, "linear"), seg)
    th = 0.2
    M = np.eye(4)
    M[:3, :3] = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]) * 0.8
    M[:3, 3] = [3.3, 5.1, -1.2]
    shape = (50, 52, 48)
    labs = np.unique(seg)
    for mode in ("linear", "gaussian"):
        w = []
        for l in labs:
            src = (seg == l).astype(np.float32)
            if mode == "gaussian":
                src = ndimage.gaussian_filter(src, 1.0, mode="nearest" if l == 0 else "constant")
            w.append(ndimage.affine_transform(src, M[:3, :3], M[:3, 3], output_shape=shape, order=1,
                                              cval=float(l == 0), output=np.float32))
        ref = labs[np.argmax(np.stack(w), 0)]
        got = resample_labels(seg, M, shape, mode)
        assert (got != ref).mean() < 1e-4, mode
    nn = resample_labels(seg, M, shape, "nearest")
    assert (nn != resample_labels(seg, M, shape, "linear")).mean() < 0.02


def test_direct_to_native_roundtrip(tmp_path):
    """Build an 'MNI' label image the way reg_aladin/reg_resample would
    (ref voxel -> world -> affine -> floating world -> floating voxel) and check
    that direct_to_native brings it back onto the native grid."""
    native_shape, mni_shape = (60, 64, 50), (48, 56, 52)
    A_nat = np.array([[0, -1.1, 0, 30.], [0.9, 0, 0, -25.], [0, 0, 1.2, -28.], [0, 0, 0, 1]])
    A_mni = np.array([[1, 0, 0, -24.], [0, 1, 0, -28.], [0, 0, 1, -26.], [0, 0, 0, 1.]])
    th = 0.15
    aff = np.eye(4)                                       # MNI world -> native world
    aff[:3, :3] = [[np.cos(th), 0, np.sin(th)], [0, 1.05, 0], [-np.sin(th), 0, np.cos(th)]]
    aff[:3, 3] = [2.0, -3.0, 1.5]
    seg_nat = _blobs(native_shape, seed=5)
    V = np.linalg.inv(A_nat) @ aff @ A_mni                # MNI voxel -> native voxel
    seg_mni = ndimage.affine_transform(seg_nat, V[:3, :3], V[:3, 3], output_shape=mni_shape, order=0)
    assert np.allclose(native_to_mni_voxel(A_nat, A_mni, aff), np.linalg.inv(V))
    Vi = np.linalg.inv(V)
    inside = ndimage.affine_transform(np.ones(mni_shape, np.float32), Vi[:3, :3], Vi[:3, 3],
                                      output_shape=native_shape, order=1) > 0.999
    f = lambda n: str(tmp_path / n)
    nib.save(nib.Nifti1Image(seg_mni, A_mni), f("seg_mni.nii.gz"))
    nib.save(nib.Nifti1Image(np.zeros(native_shape, np.int16), A_nat), f("t1.nii.gz"))
    np.savetxt(f("aff.mtx"), aff)
    for mode in ("nearest", "linear", "gaussian"):
        direct_to_native(f("seg_mni.nii.gz"), f("aff.mtx"), f("t1.nii.gz"), f("out.nii.gz"), mode)
        out = nib.load(f("out.nii.gz"))
        got = np.asanyarray(out.dataobj)
        assert out.shape == native_shape and np.allclose(out.affine, A_nat) and got.dtype == np.uint8
        ref = np.where(inside, seg_nat, 0)               # labels outside the MNI box are lost
        dice = 2 * ((got == ref) & (ref > 0)).sum() / ((got > 0).sum() + (ref > 0).sum())
        assert dice > 0.9, (mode, dice)
    vols = label_volumes(f("out.nii.gz"))
    assert len(vols) == 132 and abs(sum(v for _, _, v in vols) - (got > 0).sum() * 1.1 * 0.9 * 1.2) < 1e-3


def test_docker_tools_only_mount_the_subject_folder(tmp_path, monkeypatch):
    """With --tools-docker-image every external command must mount exactly the
    subject's output folder and reference no other host path."""
    import subprocess
    from slantpy import postprocess, preprocess
    from slantpy.tools import Tools
    calls = []

    class Done:
        returncode, stdout = 0, ""

    def fake_run(cmd, **kw):
        calls.append(cmd)
        for a in cmd[cmd.index("img") + 1:]:      # create whatever the tool would have written
            for p in a.strip("[]").split(","):
                if p.startswith(str(tmp_path)) and not os.path.exists(p):
                    if p.endswith(".nii.gz"):
                        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), np.eye(4)), p)
                    elif not p.endswith("inverse-MNI-reg"):
                        np.savetxt(p, np.eye(4))
        return Done()

    monkeypatch.setattr(subprocess, "run", fake_run)
    sub = tmp_path / "out" / "t1"
    sub.mkdir(parents=True)
    t1 = str(sub / "orig_target.nii.gz")
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), np.eye(4)), t1)
    tools = Tools("/nonexistent/extra", docker_image="img")
    res, aff, inv = preprocess.register_to_mni(t1, str(sub), tools)
    preprocess.n4_correct(res, str(sub), tools)
    seg = str(sub / "seg.nii.gz")
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.uint8), np.eye(4)), seg)
    (tmp_path / "out" / "FinalResult").mkdir()
    final = str(tmp_path / "out" / "FinalResult" / "t1_seg.nii.gz")
    postprocess.legacy_to_native(seg, res, t1, final, str(sub / "inv_reg"), tools, mount_dir=str(sub))
    assert os.path.exists(final) and len(calls) == 7
    for cmd in calls:
        mounts = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-v"]
        assert mounts == ["%s:%s" % (sub, sub)], cmd
        for a in cmd[cmd.index("img") + 1:]:
            for p in a.replace("[", ",").replace("]", ",").split(","):
                if p.startswith("/"):
                    assert p.startswith(str(sub)) or p.startswith("/extra/"), p
