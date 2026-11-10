# Credits and third-party material

## SLANT

slantpy is a port of **SLANT** (Spatially Localized Atlas Network Tiles) by
Yuankai Huo and colleagues at the MASI Lab, Vanderbilt University:
<https://github.com/MASILab/SLANTbrainSeg> (BSD 3-Clause, copyright Vanderbilt
University). slantpy is not affiliated with or endorsed by Vanderbilt
University or the original authors.

Derived from the original sources:

| here | original |
|---|---|
| `slantpy/model.py` | `python/torchsrc/models/Unet3D.py`, test-time behaviour of `python/torchsrc/trainer.py` |
| `slantpy/inference.py` | `python/test.py`, `python/torchsrc/imgloaders/imgloader_CT_3D.py`, `matlab/project_space.m`, `reassign_braincolor.m`, `get_MV_fused_image.m` |
| `slantpy/preprocess.py` | `matlab/preproc_pipline.m`, `run_N4correction.m` |
| `slantpy/postprocess.py` (`legacy_to_native`) | `matlab/apply_aladin_affine_via_ants.m` |
| `slantpy/volumes.py`, `slantpy/labels.py`, `slantpy/report.py` | `matlab/generate_volume_stat.m`, `save_txt_file.m`, `get_piece_map.m`, `make_light_PDFviews.m`, `generate_brianseg_pdf_xiong_light.m` |

If you use slantpy, cite the SLANT papers and the SLANT repository:

* Yuankai Huo, Zhoubing Xu, Yunxi Xiong, Katherine Aboud, Parasanna
  Parvathaneni, Shunxing Bao, Camilo Bermudez, Susan M. Resnick, Laurie E.
  Cutting, and Bennett A. Landman. "3D whole brain segmentation using spatially
  localized atlas network tiles." NeuroImage, 2019.
  <https://www.sciencedirect.com/science/article/pii/S1053811919302307>
* Yuankai Huo, Zhoubing Xu, Katherine Aboud, Parasanna Parvathaneni, Shunxing
  Bao, Camilo Bermudez, Susan M. Resnick, Laurie E. Cutting, and Bennett A.
  Landman. "Spatially Localized Atlas Network Tiles Enables 3D Whole Brain
  Segmentation." MICCAI 2018. <https://arxiv.org/abs/1806.00546>

* MASI Lab, Vanderbilt University. SLANTbrainSeg: deep whole brain high resolution segmentation. Software repository, <https://github.com/MASILab/SLANTbrainSeg>.

## Where the model files come from

The networks and atlas files are distributed by the original authors inside the
SLANT docker image, as described in the SLANT repository's README:

    docker pull masidocker/public:deep_brain_seg_v1_1_0        # GPU build
    docker pull masidocker/public:deep_brain_seg_v1_1_0_CPU    # CPU build

`slantpy extract <image> <folder>` copies the weights, `atlas_model.mat`, the
MNI305 template and the NiftyReg/ANTs binaries out of whichever of these you
have. Either build works; the model files are the same.

## Not included in this repository

The trained network weights, `atlas_model.mat` and the MNI305 template are the
work of their respective authors and are **not** redistributed here. They are
taken from a copy of the SLANT docker image that the user already has
(`slantpy extract`). Their terms of use are those of their owners.

## External programs called at run time

* NiftyReg (`reg_aladin`, `reg_transform`, `reg_resample`) - University College London, BSD license.
* ANTs (`N4BiasFieldCorrection`, and in `--postproc legacy` `antsRegistration`,
  `antsApplyTransforms`) - Apache 2.0 license.

## Label protocol

The 132 regions follow the BrainCOLOR protocol
(<http://braincolor.mindboggle.info/protocols/>).

## Research use only

Like the original, this software is for research. It has not been reviewed or
approved for clinical use.
