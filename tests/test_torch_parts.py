import os
import shutil
import stat
import sys
from collections import OrderedDict

import nibabel as nib
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from slantpy.labels import LABELS, MNI_SHAPE, N_CLASSES  # noqa: E402
from slantpy.model import UNet3D, load_weights  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def old_style_state(seed=0):
    """Random weights with the key set of a PyTorch-0.2 checkpoint (no num_batches_tracked)."""
    torch.manual_seed(seed)
    m = UNet3D(1, N_CLASSES)
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm3d):
            mod.weight.data.uniform_(0.5, 1.5)
            mod.bias.data.normal_(0, 0.2)
    return OrderedDict((k, v) for k, v in m.state_dict().items() if "num_batches_tracked" not in k)


def test_checkpoint_keys_and_loading(tmp_path):
    state = old_style_state()
    assert "ec0.1.running_mean" in state and "dc0.0.weight" in state and "dc9.0.weight" in state
    torch.save(state, str(tmp_path / "m.pth"))
    np.savez(str(tmp_path / "m.npz"), **{k: v.numpy() for k, v in state.items()})
    a = load_weights(UNet3D(), str(tmp_path / "m.pth"))
    b = load_weights(UNet3D(), str(tmp_path / "m.npz"))
    for (k, v), (_, w) in zip(a.state_dict().items(), b.state_dict().items()):
        assert torch.equal(v, w), k
    bad = dict(state); bad.pop("ec3.0.weight")
    torch.save(bad, str(tmp_path / "bad.pth"))
    with pytest.raises(RuntimeError):
        load_weights(UNet3D(), str(tmp_path / "bad.pth"))


def test_lean_forward_equals_literal_and_uses_batch_statistics():
    m = UNet3D()
    m.load_state_dict(old_style_state(1), strict=False)
    m.eval()                                    # must still normalise with batch statistics
    x = torch.rand(1, 1, 16, 24, 32) * 255
    with torch.inference_mode():
        a, b = m.forward(x), m.forward_lean(x)
        assert a.shape == (1, 133, 16, 24, 32)
        assert torch.allclose(a, b, rtol=1e-3, atol=1e-3 * float(a.abs().max()))
        assert (a.argmax(1) == b.argmax(1)).float().mean() > 0.999
        for slab in (4, 5, 16, 64):             # slab-wise tail == whole-volume tail
            assert (m.labels_lean(x, slab) == b.argmax(1)).float().mean() > 0.9995, slab
        assert torch.equal(m.labels(x), a.argmax(1).to(torch.uint8))
        # reference built from primitives: conv -> instance statistics -> relu
        conv, bn = m.ec0[0], m.ec0[1]
        y = conv(x)
        mu, var = y.mean((0, 2, 3, 4), keepdim=True), y.var((0, 2, 3, 4), unbiased=False, keepdim=True)
        ref = torch.relu((y - mu) / torch.sqrt(var + bn.eps) * bn.weight.view(1, -1, 1, 1, 1)
                         + bn.bias.view(1, -1, 1, 1, 1))
        assert torch.allclose(m.ec0(x), ref, atol=1e-4)
        m.batch_stats = False
        m.eval()
        assert not torch.allclose(m.ec0(x), ref, atol=1e-4)      # running stats give something else
        assert torch.allclose(m.forward(x), m.forward_lean(x), rtol=1e-3, atol=1e-3)


def make_fixture(tmp_path):
    extra = tmp_path / "extra"
    fma = extra / "full-multi-atlas"
    for d in ("niftyreg/bin", "ANTs-bin/bin", "MNI", "atlas-processing"):
        (fma / d).mkdir(parents=True)
    fake = os.path.join(HERE, "fake_tools", "fake_tool.py")
    for d, names in (("niftyreg/bin", ("reg_aladin", "reg_transform", "reg_resample")),
                     ("ANTs-bin/bin", ("N4BiasFieldCorrection", "antsRegistration", "antsApplyTransforms"))):
        for n in names:
            shutil.copy(fake, str(fma / d / n))
            os.chmod(str(fma / d / n), 0o755)
    rng = np.random.default_rng(0)
    g = np.indices(MNI_SHAPE)
    brain = sum(((gi - s / 2) / (s * 0.4)) ** 2 for gi, s in zip(g, MNI_SHAPE)) < 1
    mni_aff = np.diag([1., 1, 1, 1]); mni_aff[:3, 3] = [-86, -126, -72]
    template = (brain * 100 + rng.normal(0, 5, MNI_SHAPE)).astype(np.float32)
    nib.save(nib.Nifti1Image(template, mni_aff), str(fma / "MNI" / "average305_t1_tal_lin.nii.gz"))
    nib.save(nib.Nifti1Image(brain.astype(np.uint8), mni_aff), str(fma / "MNI" / "average305_t1_tal_lin_mask.nii.gz"))
    np.savez(str(fma / "atlas-processing" / "atlas_model.npz"), brain_seg=brain,
             norm_regressand=np.sort(rng.normal(size=int(brain.sum()))))
    wdir = extra / "model_dir" / "2_2_2" / "finetune_out" / "models"
    wdir.mkdir(parents=True)
    torch.save(old_style_state(2), str(wdir / "model_epoch_0027.pth"))

    # a 'native' T1 on a different grid (2 mm, other orientation and name with brackets)
    nat_aff = np.array([[0, 0, 2., -80], [-2., 0, 0, 100], [0, 2., 0, -70], [0, 0, 0, 1]])
    nat = np.zeros((100, 78, 86), np.float32)
    from scipy import ndimage
    v = np.linalg.inv(mni_aff) @ nat_aff
    nat = ndimage.affine_transform(template, v[:3, :3], v[:3, 3], output_shape=nat.shape, order=1)
    (tmp_path / "in").mkdir()
    t1 = str(tmp_path / "in" / "t1(a).nii.gz")
    nib.save(nib.Nifti1Image(nat.astype(np.int16), nat_aff), t1)

    return extra, t1, nat, nat_aff


def test_end_to_end_with_fake_tools(tmp_path):
    """Whole driver, both post-processing paths, with stub binaries, random
    weights and a single tile (a full 27-tile run needs the real model files)."""
    from slantpy.pipeline import main
    extra, t1, nat, nat_aff = make_fixture(tmp_path)
    if True:
        outs = {}
        for mode, fwd in (("legacy", "literal"), ("direct", "lean")):
            out = str(tmp_path / ("out_" + mode))
            assert main(["run", t1, out, "--extra-dir", str(extra), "--postproc", mode, "--forward", fwd,
                         "--device", "cpu", "--keep-tiles", "--pieces", "2_2_2"]) == 0
            seg = nib.load(os.path.join(out, "FinalResult", "t1a_seg.nii.gz"))
            assert seg.shape == nat.shape and np.allclose(seg.affine, nat_aff)
            outs[mode] = np.asanyarray(seg.dataobj)
            assert set(np.unique(outs[mode]).astype(int)) <= set((0,) + LABELS)
            txt = open(os.path.join(out, "FinalVolTxt", "t1a_label_volumes.txt")).read().splitlines()
            assert len(txt) == 133 and txt[1].startswith("3rd Ventricle,4,")
            assert os.path.exists(os.path.join(out, "FinalPDF", "t1a_result.pdf"))
            assert os.path.exists(os.path.join(out, "t1a", "tiles", "2_2_2.npy"))
        # same network output, same (identity) transform: the two routes must agree closely
        assert (outs["legacy"] > 0).any()
        assert (outs["legacy"] == outs["direct"]).mean() > 0.97


def test_gpu_choice_and_cpu_fallback(monkeypatch, tmp_path):
    import slantpy.inference as inf
    assert inf.choose_gpu([0.2, 23.0], 4.0) == 1          # GPU 0 busy, GPU 1 free
    assert inf.choose_gpu([0.2, 3.0], 4.0) is None        # both too busy -> CPU
    assert inf.choose_gpu([], 4.0) is None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(inf, "gpu_free_gb", lambda: [0.2, 23.0])
    assert inf.pick_device(4.0, log=lambda m: None) == "cuda:1"
    monkeypatch.setattr(inf, "gpu_free_gb", lambda: [0.2, 1.0])
    assert inf.pick_device(4.0, log=lambda m: None) == "cpu"

    # automatic mode: a GPU that runs out of memory mid-run must hand over to the CPU.
    # Simulated on a CPU-only machine by pretending the chosen device is a GPU.
    wdir = tmp_path / "model_dir" / "2_2_2" / "finetune_out" / "models"
    wdir.mkdir(parents=True)
    torch.save(old_style_state(3), str(wdir / "model_epoch_0027.pth"))
    real_device, state = torch.device, {"fail": True}

    class FakeCuda:
        type = "cuda"
        def __str__(self):
            return "cuda:1"

    monkeypatch.setattr(inf, "pick_device", lambda need, log: "cuda:1")
    monkeypatch.setattr(torch, "device", lambda d: FakeCuda() if str(d).startswith("cuda") else real_device(d))
    monkeypatch.setattr(torch.nn.Module, "to", lambda self, *a, **k: self)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    orig_to = torch.Tensor.to

    def tensor_to(self, dev, *a, **k):
        if isinstance(dev, FakeCuda):
            raise torch.OutOfMemoryError("simulated")
        return orig_to(self, dev, *a, **k)

    monkeypatch.setattr(torch.Tensor, "to", tensor_to)
    msgs = []
    vol = np.random.default_rng(0).normal(size=MNI_SHAPE).astype(np.float32)
    out = list(inf.segment_tiles(vol, str(tmp_path / "model_dir"), pieces=("2_2_2",), log=msgs.append))
    assert out[0][1].shape == (96, 128, 88)
    assert any("continuing on the CPU" in m for m in msgs) and "(cpu)" in msgs[-1]
