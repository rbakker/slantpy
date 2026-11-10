# slantpy

**SLANT whole-brain segmentation without Matlab, ported from
[MASILab/SLANTbrainSeg](https://github.com/MASILab/SLANTbrainSeg), the original
implementation by Yuankai Huo and colleagues at the MASI Lab, Vanderbilt
University.** A T1-weighted MRI goes in, 132 BrainCOLOR regions come out, on the
original image grid.

The method is described in Huo et al., *3D whole brain segmentation using
spatially localized atlas network tiles*, NeuroImage 2019. slantpy replaces the
original's compiled Matlab and Python 2 / PyTorch 0.2 code with Python 3; it is
an independent port, not affiliated with the original authors. It uses the
original trained networks and reproduces the original pipeline's result in MNI space almost voxel for voxel,
in a fraction of the time and memory, and with a simpler, more accurate way
back to the original image.

Documentation: <https://rbakker.github.io/slantpy/> ·
[design notes](docs/design-notes.md) · [credits](NOTICE.md)

|  | original docker image | slantpy |
|---|---|---|
| Runtime needs | Matlab runtime, Python 2, PyTorch 0.2 | Python 3, current PyTorch |
| Peak memory, 8 CPU cores | 28.7 GB | 1.8 GB |
| One scan, same 8 CPU cores | 26.4 min | 4.5 min (2.7 min with a GPU) |
| Back to native space | two more registrations, then ANTs | one resampling with the inverse affine |

Measured on one test scan on one machine; see [Validation](#validation).

## How it works

1. **Register** the T1 to the MNI305 template (NiftyReg `reg_aladin`, affine).
2. **Bias-correct** (ANTs `N4BiasFieldCorrection`; optional, see `--skip-n4`) and
   **normalise** intensities to the atlas with a robust linear fit.
3. **Segment** 27 overlapping tiles, each with its own 3-D U-Net.
4. **Fuse** the tiles by majority vote.
5. **Resample** the labels to the original image with the inverse of the
   affine from step 1, and write the volume table and a QA page.

Steps 1 and 2 call the external programs; everything else is Python.

## What you need

* Python 3.9 or newer with numpy, scipy, nibabel, PyTorch (matplotlib for the
  QA page, h5py only for a v7.3 `atlas_model.mat`).
* The SLANT model files: 27 network weight files, `atlas_model.mat` and the
  MNI305 template. They are **not** in this repository. `slantpy extract`
  copies them out of the SLANT docker image (see Install).
* `reg_aladin` and `N4BiasFieldCorrection`: from the docker image, or from the
  conda packages `niftyreg` and `ants`.

## Install

    pip install git+https://github.com/rbakker/slantpy.git

### Get the model files

The trained networks are distributed by the SLANT authors inside their docker
image (see the [SLANT README](https://github.com/MASILab/SLANTbrainSeg)).
Pull it once and let slantpy copy out what it needs:

    docker pull masidocker/public:deep_brain_seg_v1_1_0_CPU
    slantpy extract masidocker/public:deep_brain_seg_v1_1_0_CPU /data/resources/SLANT

The GPU build `masidocker/public:deep_brain_seg_v1_1_0` works as well; the
model files are the same. After this step docker is no longer needed, unless
you choose to run the registration tools inside the image. `extract` creates

    /data/resources/SLANT/
        model_dir/<tile>/finetune_out/models/model_epoch_0027.pth     27 tiles
        full-multi-atlas/MNI/average305_t1_tal_lin*.nii.gz
        full-multi-atlas/atlas-processing/atlas_model.npz
        full-multi-atlas/niftyreg/, full-multi-atlas/ANTs-bin/        binaries from the image

A conda environment file and a Snakemake rule are in [`workflow/`](workflow).

## Run

    slantpy run t1.nii.gz out/ --extra-dir /data/resources/SLANT

Results:

    out/FinalResult/t1_seg.nii.gz             labels on the original grid (uint8)
    out/FinalVolTxt/t1_label_volumes.txt      volume per region, mm3
    out/FinalPDF/t1_result.pdf                QA page
    out/t1/                                   intermediate files; a rerun resumes from them

Useful options:

| option | meaning |
|---|---|
| `--device cpu` / `cuda:1` | Default: the GPU with the most free memory if it has enough, otherwise the CPU. |
| `--threads N` | CPU threads for PyTorch. |
| `--interp linear` / `nearest` / `gaussian` | Label interpolation to native space (default `linear`). |
| `--skip-n4` | Skip the N4 bias-field correction, for images already corrected upstream. ANTs is then not needed. |
| `--postproc legacy --forward literal` | Behave like the original pipeline (see below). |
| `--tools-docker-image IMAGE` | Run `reg_aladin` and N4 inside the SLANT docker image instead of on the host. |
| `--half` | Mixed precision on GPU; not bit-identical. |
| `--no-pdf`, `--keep-tiles`, `--pieces 2_2_2` | Skip the QA page; keep per-tile labels; run single tiles for debugging. |

The input file is never modified and only the subject's output folder is ever
written to (or mounted into a container).

## Differences from the original pipeline

By default slantpy changes two things:

* **Network evaluation** (`--forward lean`). Same layers and weights, but the
  last three layers are evaluated in slabs and no concatenated tensor is built.
  Peak memory for one tile on CPU: 1.7 GB instead of 4.2 GB.
* **Back to native space** (`--postproc direct`). The original registers the
  MNI-space image back to the T1 a second time, then runs an ANTs registration
  between two label images to convert that transform, then resamples. slantpy
  inverts the affine it already has and resamples once.

`--postproc legacy --forward literal` reproduces the original chain command by
command, for comparison.

**Results from the two are not interchangeable.** On a test scan the default
output and the docker output differed in 7 % of brain voxels (regional volumes
by 1.9 % on average, total brain volume by 0.2 %). Do not mix them within one
study.

## Validation

One T1, docker image against slantpy with the same binaries:

| stage | agreement with the docker image |
|---|---|
| registration to MNI | identical |
| network input | within 1 part in 100,000 |
| 27 tiles | 5 voxels in 10 million differ |
| fused labels, MNI space | 2 voxels of 5.9 million differ |
| native space, `--postproc legacy` | 97.2 % of brain voxels, mean Dice 0.973 |
| native space, default | 93.1 % of brain voxels, mean Dice 0.936 |

To judge which native-space result is better, each was pushed back to MNI space
with the exact forward affine and compared with the network's labels there:

| native-space result | in-brain agreement | worst region (Dice) |
|---|---|---|
| docker image | 0.903 | 0.817 |
| slantpy default | 0.950 | 0.914 |

### Run time and memory

The same scan with both pipelines restricted to the same 8 CPU cores:

| | docker image | slantpy `--postproc legacy --forward literal` | slantpy default |
|---|---|---|---|
| total | 26.4 min | 7.4 min | 4.5 min |
| peak memory | 28.7 GB | 4.2 GB | 1.8 GB |
| pre-processing | 267 s | 117 s | 118 s |
| segmentation, 27 tiles | 728 s | 144 s | 143 s |
| back to native space | 450 s | 182 s | 5 s |
| volume table | 138 s | included above | included above |

With a GPU the default configuration took 2.7 min in total. Memory is the peak of
the main process; the short-lived registration and N4 processes are not included.

This is a single scan, not a validation study. Run it on your own data:

    python scripts/compare_with_docker.py t1.nii.gz --workdir /scratch/cmp
    python scripts/check_native_mapping.py /scratch/cmp/improved/t1/t1 <segmentations...>

Other scripts: `benchmark_memory.py` (one tile, both forward modes),
`compare_segs.py` (Dice between two label images), `make_observer_zip.py`
(package for a 3-D viewer), `export_weights_legacy.py` (fallback if PyTorch
cannot read the 2018 checkpoints).

## Known quirks kept from the original

* The networks run in training mode at test time, so batch normalisation uses
  each tile's own statistics. The weights were validated that way.
* Each tile is scaled by its own minimum and maximum.
* Ties in the majority vote go to the lowest label number.

More in the [design notes](docs/design-notes.md).

## Tests

    python -m pytest tests

Synthetic data, random weights and stub registration programs; no model files
needed.

## Credits and license

SLANT was created by Yuankai Huo and colleagues at the MASI Lab, Vanderbilt
University. slantpy is an independent port, not affiliated with or endorsed by
them. If you use it, cite the SLANT papers and the
[SLANT repository](https://github.com/MASILab/SLANTbrainSeg), as listed in
[NOTICE.md](NOTICE.md).

BSD 3-Clause, see [LICENSE](LICENSE). Copyright (c) 2018-2026 Vanderbilt
University; copyright (c) 2026 TU Eindhoven.

For research use only; not for clinical use.
