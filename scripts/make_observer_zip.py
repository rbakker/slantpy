#!/usr/bin/env python3
"""Build a tractogram.observer zip package (datapackage.json manifest) with the
original T1, the docker segmentation, the slantpy 'mimic' segmentation and a
difference map (docker label where the two differ, 0 where they agree).

    python scripts/make_observer_zip.py --workdir /scratch/slant_cmp --name 1966-ANT1W-201

or with explicit files:

    python scripts/make_observer_zip.py --t1 t1.nii.gz --docker-seg a.nii.gz --mimic-seg b.nii.gz \
        --name sub01 --out sub01_slant.zip [--extra improved=c.nii.gz]

Spec: https://tractogram.observer/docs/zip-package-spec.md
"""
import argparse
import glob
import gzip
import io
import json
import os
import sys
import zipfile

import nibabel as nib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def nifti_gz_bytes(data, ref):
    """uint16 label volume on the grid of ``ref``, as .nii.gz bytes."""
    hdr = ref.header.copy()
    hdr.set_data_shape(data.shape)
    hdr.set_data_dtype(np.uint16)
    hdr.set_slope_inter(1, 0)
    hdr["cal_min"], hdr["cal_max"] = 0, 0
    raw = nib.Nifti1Image(data.astype(np.uint16), ref.affine, hdr).to_bytes()
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as fh:
        fh.write(raw)
    return buf.getvalue()


def load_seg(path, ref):
    img = nib.load(path)
    if tuple(img.shape[:3]) != tuple(ref.shape[:3]):
        sys.exit("%s has grid %s, the T1 has %s" % (path, img.shape, ref.shape))
    if not np.allclose(img.affine, ref.affine, atol=1e-2):
        print("warning: %s has a slightly different affine than the T1; the T1's is used" % path)
    return np.rint(np.asanyarray(img.dataobj)).astype(np.uint16).reshape(ref.shape[:3])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", help="work directory of compare_with_docker.py")
    ap.add_argument("--name", required=True, help="subject name (input file name without .nii.gz)")
    ap.add_argument("--t1")
    ap.add_argument("--docker-seg")
    ap.add_argument("--mimic-seg")
    ap.add_argument("--extra", action="append", default=[], metavar="LABEL=PATH",
                    help="further segmentation to include, e.g. improved=/path/seg.nii.gz (repeatable)")
    ap.add_argument("--lut", default=os.path.join(HERE, "slant_itksnap.txt"),
                    help="ITK-SNAP label description file")
    ap.add_argument("--out", help="zip to write (default <name>_slant_observer.zip)")
    a = ap.parse_args()

    n, w = a.name, a.workdir
    if w:
        t1s = glob.glob(os.path.join(w, "reference", n, "INPUTS", n + ".nii*"))
        a.t1 = a.t1 or (t1s[0] if t1s else None)
        a.docker_seg = a.docker_seg or os.path.join(w, "reference", n, "OUTPUTS", "FinalResult", n + "_seg.nii.gz")
        a.mimic_seg = a.mimic_seg or os.path.join(w, "mimic", n, "FinalResult", n + "_seg.nii.gz")
    for what, p in (("T1", a.t1), ("docker segmentation", a.docker_seg), ("mimic segmentation", a.mimic_seg),
                    ("lookup table", a.lut)):
        if not p or not os.path.exists(p):
            sys.exit("%s not found: %s" % (what, p))

    t1 = nib.load(a.t1)
    ref = nib.Nifti1Image(np.zeros(t1.shape[:3], np.uint8), t1.affine, t1.header)
    docker, mimic = load_seg(a.docker_seg, ref), load_seg(a.mimic_seg, ref)
    differ = docker != mimic
    diff = np.where(differ, docker, 0)
    hidden = int((differ & (docker == 0)).sum())
    print("voxels that differ: %d (%.2f%% of the voxels labelled by either)"
          % (differ.sum(), 100.0 * differ.sum() / max(((docker > 0) | (mimic > 0)).sum(), 1)))
    print("  of these, %d are background in the docker result and therefore 0 (invisible) in the "
          "difference map" % hidden)

    t1_name = "%s_T1w.nii.gz" % n
    segs = [("seg-docker", "SLANT docker", docker), ("seg-mimic", "slantpy mimic", mimic),
            ("seg-diff", "docker label where docker and mimic differ", diff)]
    for item in a.extra:
        label, path = item.split("=", 1)
        segs.append(("seg-" + label, "slantpy " + label, load_seg(path, ref)))

    resources = [{"name": "t1", "path": t1_name, "title": "T1w (original)", "role": "anat"}]
    for key, title, _ in segs:
        resources.append({"name": key, "path": "%s_%s.nii.gz" % (n, key), "title": title,
                          "role": "label", "labels": "slant-lut"})
    resources.append({"name": "slant-lut", "path": "slant_itksnap.txt"})
    manifest = {"name": "%s-slant-comparison" % n.lower(), "title": "SLANT docker vs slantpy: %s" % n,
                "description": "Original T1, docker segmentation, slantpy mimic segmentation and a map "
                               "with the docker label wherever the two differ (0 where they agree).",
                "space": "native", "resources": resources, "view": {"bounds": t1_name, "from": "left"}}

    out = a.out or "%s_slant_observer.zip" % n
    with zipfile.ZipFile(out, "w") as z:
        # already-compressed files are stored, text is deflated (per the spec)
        if a.t1.endswith(".gz"):
            z.write(a.t1, t1_name, compress_type=zipfile.ZIP_STORED)
        else:
            with open(a.t1, "rb") as fh:
                buf = io.BytesIO()
                with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as g:
                    g.write(fh.read())
            z.writestr(zipfile.ZipInfo(t1_name), buf.getvalue(), compress_type=zipfile.ZIP_STORED)
        for key, _, data in segs:
            z.writestr(zipfile.ZipInfo("%s_%s.nii.gz" % (n, key)), nifti_gz_bytes(data, ref),
                       compress_type=zipfile.ZIP_STORED)
        z.write(a.lut, "slant_itksnap.txt", compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("datapackage.json", json.dumps(manifest, indent=2), compress_type=zipfile.ZIP_DEFLATED)
    print("wrote %s (%.1f MB)" % (out, os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main()
