"""End-to-end driver: the Python replacement of run_deep_brain_seg.sh.

Outputs keep the names of the original pipeline:
    <out>/FinalResult/<name>_seg.nii.gz
    <out>/FinalVolTxt/<name>_label_volumes.txt
    <out>/FinalPDF/<name>_result.pdf
    <out>/<name>/...   intermediate files (registration, N4, normalised image, MNI-space labels)
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

import nibabel as nib
import numpy as np

from . import postprocess, preprocess
from .inference import fuse_majority, segment_tiles
from .labels import PIECES
from .tools import Tools
from .volumes import write_volume_txt


def subject_name(path):
    name = os.path.basename(path)
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            name = name[:-len(ext)]
            break
    return name.replace("(", "").replace(")", "")     # clean_sublist.m, without renaming the input


def find_atlas_model(extra_dir):
    base = os.path.join(extra_dir, "full-multi-atlas", "atlas-processing", "atlas_model")
    for ext in (".npz", ".mat"):
        if os.path.exists(base + ext):
            return base + ext
    raise FileNotFoundError("atlas model not found: %s.mat" % base)


def run_subject(t1, out_dir, extra_dir, tools, postproc="direct", interp="linear",
                forward="lean", device=None, half=False, threads=None, pdf=True,
                atlas_model=None, keep_tiles=False, pieces=None, min_gpu_free_gb=None,
                log=print):
    t1 = os.path.abspath(t1)
    name = subject_name(t1)
    sub_dir = os.path.join(out_dir, name)
    final_dir = os.path.join(out_dir, "FinalResult")
    txt_dir = os.path.join(out_dir, "FinalVolTxt")
    for d in (sub_dir, final_dir, txt_dir):
        os.makedirs(d, exist_ok=True)
    final_seg = os.path.join(final_dir, "%s_seg.nii.gz" % name)
    timing = {}

    # Work on a copy inside the output folder (the original pipeline's
    # orig_target.nii.gz). External tools, and a container running them, then
    # only ever see the subject's output folder, never the input location.
    src = t1
    t1 = os.path.join(sub_dir, "orig_target" + (".nii.gz" if src.endswith(".gz") else ".nii"))
    if not os.path.exists(t1):
        shutil.copyfile(src, t1)

    t0 = time.time()
    processed = preprocess.preprocess(t1, sub_dir, tools,
                                      atlas_model or find_atlas_model(extra_dir), log=log)
    timing["preprocessing"] = time.time() - t0

    t0 = time.time()
    seg_mni = os.path.join(sub_dir, "target_processed_seg.nii.gz")
    if not os.path.exists(seg_mni):
        img = nib.load(processed)
        vol = np.asanyarray(img.dataobj)
        tiles = []
        for piece, cls in segment_tiles(vol, os.path.join(extra_dir, "model_dir"), device=device,
                                        lean=(forward == "lean"), half=half, threads=threads,
                                        pieces=pieces or PIECES,
                                        min_gpu_free_gb=min_gpu_free_gb, log=log):
            tiles.append((piece, cls))
            if keep_tiles:      # class indices 0..132, (x, y, z), for debugging / comparison
                os.makedirs(os.path.join(sub_dir, "tiles"), exist_ok=True)
                np.save(os.path.join(sub_dir, "tiles", piece + ".npy"), cls)
        fused = fuse_majority(tiles)
        hdr = img.header.copy()
        hdr.set_data_dtype(np.uint8)
        hdr.set_slope_inter(1, 0)
        nib.save(nib.Nifti1Image(fused, img.affine, hdr), seg_mni)
    timing["segmentation"] = time.time() - t0

    t0 = time.time()
    if not os.path.exists(final_seg):
        if postproc == "legacy":
            postprocess.legacy_to_native(seg_mni, processed, t1, final_seg,
                                         os.path.join(sub_dir, "inv_reg"), tools, mount_dir=sub_dir)
        else:
            aff = os.path.join(sub_dir, "MNI-registration", "target_MNI_aff.mtx")
            postprocess.direct_to_native(seg_mni, aff, t1, final_seg, mode=interp)
    write_volume_txt(final_seg, os.path.join(txt_dir, "%s_label_volumes.txt" % name))
    timing["postprocessing"] = time.time() - t0

    if pdf:
        try:
            from .report import write_pdf
            os.makedirs(os.path.join(out_dir, "FinalPDF"), exist_ok=True)
            write_pdf(processed, seg_mni, os.path.join(out_dir, "FinalPDF", "%s_result.pdf" % name), name)
        except ImportError:
            log("matplotlib not installed, skipping the PDF report")
    for k, v in timing.items():
        log("*******%s time: %.1f seconds" % (k, v))
    return final_seg


def extract_resources(image, dest, binaries=True, docker_cmd="docker"):
    """Copy model weights, atlas model, MNI template (and the binaries) out of the SLANT image."""
    fma = "full-multi-atlas"
    wanted = ["model_dir", fma + "/MNI", fma + "/atlas-processing/atlas_model.mat"]
    if binaries:
        wanted += [fma + "/niftyreg", fma + "/ANTs-bin"]
    cid = subprocess.check_output([docker_cmd, "create", image, "/bin/true"],
                                  universal_newlines=True).strip()
    try:
        for rel in wanted:
            target = os.path.join(dest, rel)
            if os.path.exists(target):
                print("exists, skipping:", target)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            print("copying /extra/%s" % rel)
            subprocess.check_call([docker_cmd, "cp", "-L", "%s:/extra/%s" % (cid, rel), target])
    finally:
        subprocess.call([docker_cmd, "rm", cid], stdout=subprocess.DEVNULL)
    mat = os.path.join(dest, fma, "atlas-processing", "atlas_model.mat")
    npz = mat[:-4] + ".npz"
    if not os.path.exists(npz):
        preprocess.extract_atlas_model(mat, npz)
        print("wrote", npz)
    return dest


def main(argv=None):
    ap = argparse.ArgumentParser(prog="slantpy", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="segment one T1 or every *.nii.gz in a directory")
    r.add_argument("input", help="T1 image (.nii/.nii.gz) or a directory of them")
    r.add_argument("output", help="output directory")
    r.add_argument("--extra-dir", required=True,
                   help="host copy of the image's /extra (see 'slantpy extract')")
    r.add_argument("--tools-docker-image", default=None,
                   help="run NiftyReg/ANTs inside this image instead of from --extra-dir")
    r.add_argument("--postproc", choices=("direct", "legacy"), default="direct",
                   help="direct (default): invert the forward affine and resample once; "
                        "legacy: the original reg_aladin + ANTs chain")
    r.add_argument("--interp", choices=("linear", "nearest", "gaussian"), default="linear",
                   help="label interpolation for --postproc direct")
    r.add_argument("--forward", choices=("lean", "literal"), default="lean",
                   help="lean (default): the network evaluated with less memory; literal: layer by layer "
                        "as in the original")
    r.add_argument("--device", default=None,
                   help="cpu, cuda, cuda:1 ... Default: the GPU with the most free memory if it has "
                        "enough, otherwise the CPU; falls back to the CPU on a GPU out-of-memory error")
    r.add_argument("--min-gpu-free-gb", type=float, default=None,
                   help="free GPU memory required for the automatic choice (default 4 lean / 8 literal)")
    r.add_argument("--half", action="store_true", help="mixed precision on GPU (not bit-identical)")
    r.add_argument("--threads", type=int, default=None, help="CPU threads for PyTorch")
    r.add_argument("--tool-threads", type=int, default=1,
                   help="threads for N4/ANTs/NiftyReg (the original image uses 1 for reproducibility)")
    r.add_argument("--atlas-model", default=None, help="atlas_model.mat or its .npz extract")
    r.add_argument("--no-pdf", action="store_true")
    r.add_argument("--keep-tiles", action="store_true", help="save each tile's raw network labels")
    r.add_argument("--clean", action="store_true", help="empty the output directory first")
    r.add_argument("--pieces", nargs="+", default=os.environ.get("SLANTPY_PIECES", "").split() or None,
                   help="debugging: only run these tiles, e.g. 2_2_2 (default: all 27)")

    e = sub.add_parser("extract", help="copy weights/atlas/binaries out of the SLANT docker image")
    e.add_argument("image")
    e.add_argument("dest")
    e.add_argument("--no-binaries", action="store_true")

    a = ap.parse_args(argv)
    if a.cmd == "extract":
        extract_resources(a.image, a.dest, binaries=not a.no_binaries)
        return 0

    out = os.path.realpath(a.output)     # real path: it is bind-mounted when tools run in docker
    if a.clean and os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)
    if os.path.isdir(a.input):
        inputs = sorted(glob.glob(os.path.join(a.input, "*.nii.gz")))
    else:
        inputs = [a.input]
    if not inputs:
        ap.error("no *.nii.gz found in %s" % a.input)
    tools = Tools(a.extra_dir, a.tools_docker_image, threads=a.tool_threads)
    for t1 in inputs:
        print("=== %s" % t1)
        run_subject(t1, out, os.path.abspath(a.extra_dir), tools, postproc=a.postproc,
                    interp=a.interp, forward=a.forward, device=a.device, half=a.half,
                    threads=a.threads, pdf=not a.no_pdf, atlas_model=a.atlas_model,
                    keep_tiles=a.keep_tiles, pieces=a.pieces,
                    min_gpu_free_gb=a.min_gpu_free_gb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
