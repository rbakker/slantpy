# Design notes: memory use and post-processing

Why slantpy's defaults differ from the original pipeline, and what else could
be changed. Items marked **[default]** are what `slantpy run` does; the
original behaviour stays available through the option shown. Numbers marked
*measured* come from `scripts/benchmark_memory.py` on a 2-core CPU machine with
random weights; rerun the script on your hardware.

## 1. Memory

### 1.1 The network is where the memory goes

One tile is 88 × 96 × 128 = 1.08 million voxels, and the decoder carries
133 to 197 channels at that resolution (0.6 to 0.85 GB per tensor).

| | peak memory, one tile, CPU | time |
|---|---|---|
| original image (PyTorch 0.2), per its README | "50 GB might be required" | – |
| original image, measured on 8 CPU cores (whole run) | 28.7 GB | – |
| slantpy `--forward literal` (same layers, current PyTorch) | 4.2 GB *measured* | 22 s |
| slantpy `--forward lean` | 1.7 GB *measured* (0.65 GB of it is Python + PyTorch) | 22 s |

The two slantpy modes labelled 1 voxel out of 1,081,344 differently in that test.

Why the original needs so much: PyTorch 0.2 computes 3-D (transposed)
convolutions on CPU by unfolding the input into a column buffer of
channels × 27 × voxels floats. For the 197-channel layer that is about 23 GB
for a single layer (my estimate from the layer sizes, not a measurement).
Current PyTorch does not do this, so **the port alone removes most of the
problem**.

What `--forward lean` **[default]** adds, without changing the function computed
(`--forward literal` evaluates the layers as written):

1. *Last three layers in slabs.* `dc3`, `dc2`, `dc1` and the arg-max contain no
   normalisation, so they can be evaluated 16 slices at a time with a 2-voxel
   halo. No 133-channel full-resolution tensor is ever allocated. This is the
   step that takes the peak from 4.2 to 1.7 GB.
2. *No concatenation.* A convolution over `cat(a, b)` equals the sum of two
   convolutions; the 197-channel tensor is never built.
3. *Transposed 3×3×3 convolutions written as ordinary convolutions* (flipped
   kernel, channels swapped), which use the better optimised code path.
4. In-place ReLU, skip connections freed as soon as they are consumed,
   `torch.inference_mode()`.
5. One process and one model object for all 27 tiles instead of 27 Python
   start-ups; tile results stay in memory (1 MB each) instead of 81 NIfTI files.

On a GPU the same applies to GPU memory. `--half` (mixed precision) roughly
halves it again but is not bit-identical; check it with the comparison script
before adopting it.

Because floating-point sums are evaluated in a different order, lean and
literal can disagree on voxels where two classes are practically tied.

### 1.2 Pre-processing

* `process_target` loads twelve variables from `atlas_model.mat`, including the
  whole atlas data matrix, and uses two (`brain_seg`, `norm_regressand`).
  slantpy reads only those two; `slantpy extract` stores them as a
  small `.npz`.
* Every Matlab NIfTI read gunzips to `/tmp` first, and the T1 is copied twice
  (`orig_target.nii.gz`, `working_dir/deep_learning/`). slantpy reads compressed
  files directly and keeps one working copy.

### 1.3 Fusion

Matlab stacks 27 full volumes and calls `mode` once per voxel (5.9 million
calls). That is slow rather than large. In slantpy votes are counted per
16-voxel slab (73 MB peak, a few seconds) with the same tie rule.

## 2. Post-processing

### 2.1 What the current code does

The fused labels live on the MNI305 grid. To bring them to the original T1,
`apply_aladin_affine_via_ants.m` runs:

1. `reg_aladin` again, now from the normalised MNI-space T1 to the original T1;
2. `reg_resample` (nearest neighbour) of the labels with that affine;
3. `antsRegistration` (rigid, then affine, mean squares, with `-u`) between the
   MNI-space **label image** and the resampled **label image**;
4. `antsApplyTransforms -n MultiLabel` with the transform from step 3.

Steps 2 and 3 exist only to obtain the step-1 affine in ANTs' file format.

Problems:

* The inverse transform is already known: `target_MNI_aff.mtx` from
  pre-processing (the pipeline even writes `target_MNI_aff_inv.mtx` and never
  uses it). Estimating it twice more means the labels go back through a
  transform that is close to, but not, the inverse of the one that brought the
  image in.
* Mean squares on label numbers is not a meaningful similarity: the difference
  between label 4 and label 207 counts 40,000 times more than between 4 and 5.
  `-u` additionally histogram-matches the two label images.
* Two extra registrations cost minutes and depend on thread count.

### 2.2 Replacement: `--postproc direct` **[default]**

    native voxel → native world → inv(target_MNI_aff) → MNI world → MNI voxel

One resampling in numpy/scipy, no registration, no ANTs, output stored as
uint8 with the original T1's header. Labels are handled one at a time inside
their bounding box, so memory is two float32 volumes of the native grid
regardless of the number of labels.

Interpolation (`--interp`):

* `linear` (default): each label's mask is interpolated trilinearly, the largest
  wins. No smoothing beyond what resampling requires.
* `nearest`: fastest, blockier borders.
* `gaussian`: masks smoothed with a 1-voxel Gaussian first. This approximates
  ANTs `MultiLabel`; it is not the identical kernel. It rounds off thin
  structures, which is also true of the current output.

`--postproc legacy` runs the original four commands instead.

### 2.3 What a comparison on one scan showed

One T1, docker image against slantpy with the same binaries
(`scripts/compare_with_docker.py`, `scripts/check_native_mapping.py`):

* Up to the fused labels in MNI space the port and the docker image agree in
  all but 2 of 5.9 million voxels.
* Pushing each native-space result back to the MNI grid with the exact forward
  affine and comparing with the network's labels there:

  | native-space result | in-brain agreement | mean Dice | worst region |
  |---|---|---|---|
  | docker image | 0.903 | 0.907 | 0.817 |
  | slantpy, `--postproc legacy` | 0.897 | 0.901 | 0.802 |
  | slantpy, `--postproc direct` | 0.950 | 0.950 | 0.914 |

* The direct mapping reproduces `reg_aladin`'s own resampled image with a
  correlation of 0.999 (a one-voxel error would give about 0.97).
* The docker image and slantpy's legacy mode, fed inputs that differ in 2 label
  voxels, disagree in 2.8 % of brain voxels in native space: the legacy chain
  is sensitive to details that should not matter.
* Direct and docker results differ in 6.9 % of brain voxels; regional volumes
  differ by 1.9 % on average (up to 11 %), total brain volume by 0.2 %. Do not
  mix results of the two within one study.

* Like for like on the same 8 CPU cores: docker image 26.4 min and 28.7 GB peak;
  slantpy default 4.5 min and 1.8 GB; slantpy with legacy settings 7.4 min and
  4.2 GB. Per stage (docker / slantpy default): pre-processing 267 / 118 s,
  segmentation 728 / 143 s, back to native space 450 / 5 s, volume table 138 s / included in the 5 s.
* With 8 tool threads instead of 1, slantpy's legacy mode and the docker image
  agreed in 98.3 % of brain voxels instead of 97.2 %: the legacy chain's result
  depends on the thread count. The direct result did not change.

This is one scan, and the round-trip test shares its affine with the direct
method, so treat it as supporting evidence rather than a validation study.

If you prefer to keep ANTs for the final resampling, the NiftyReg matrix can be
converted to an ITK transform analytically (change of sign for the x and y
axes, RAS to LPS); no registration is needed for that either.

### 2.4 Further options (not implemented)

* **Resample votes, not labels.** After fusion each voxel has up to 27 votes.
  Interpolating the per-label vote fractions to native space and taking the
  arg-max there gives smoother, less biased borders than interpolating a hard
  label map. The label-by-label loop in `resample_labels` can take vote
  fractions instead of 0/1 masks with no extra memory.
* **Fuse probabilities instead of votes.** Averaging the tiles' soft-max output
  uses the networks' confidence and removes the arbitrary tie rule (currently
  the lowest label number wins). This changes results and needs validation.
* **Segment in native resolution** is not possible with these weights; the
  networks are tied to the 1 mm MNI305 grid.

## 3. Things in the original worth knowing (reproduced, not changed)

* The network is tested in training mode (`Trainer.validate` calls
  `model.train()`), so batch normalisation uses each tile's own statistics.
  The stored running averages are never used. Switching to `eval()` gives
  different, unvalidated results.
* The last layer ends in a ReLU before the arg-max. Where every class is zero
  the arg-max is a tie; current PyTorch returns class 0 (background) there,
  PyTorch 0.2 did not guarantee which index it returned. If the comparison
  shows tile differences, look here first.
* Each tile is scaled by its own minimum and maximum, so one very bright voxel
  changes the input of the whole tile.
* "rigidMNI" is a 12-parameter affine registration (`reg_aladin` default).
* `clean_sublist.m` compares strings of different length with `~=`, which is an
  error in Matlab; a file name with brackets stops the original pipeline.
  slantpy strips brackets from the output name and leaves the input alone.
* `run_deep_brain_seg.sh` starts with `rm -r /OUTPUTS/*`. slantpy does this
  only with `--clean`.
* `preproc_pipline.m` in the archive contains a stray Python line
  (`print('SHAPES', ...)`), so the `.m` source no longer matches the compiled
  binary in the image.
