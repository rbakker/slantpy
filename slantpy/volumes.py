"""Label volume table: port of generate_volume_stat.m / save_txt_file.m."""
import nibabel as nib
import numpy as np

from .labels import LABEL_NAMES, LABELS


def label_volumes(seg_file):
    img = nib.load(seg_file)
    seg = np.asanyarray(img.dataobj)
    voxel = float(np.prod(img.header.get_zooms()[:3]))
    vals, counts = np.unique(seg, return_counts=True)
    n = {float(v): int(c) for v, c in zip(vals, counts)}
    return [(name, lab, n.get(float(lab), 0) * voxel) for name, lab in zip(LABEL_NAMES, LABELS)]


def write_volume_txt(seg_file, txt_file):
    with open(txt_file, "w") as fh:
        fh.write("Label Name (BrainCOLOR),Label Number (BrainCOLOR),Label Volume (mm^3)\n")
        for name, lab, vol in label_volumes(seg_file):
            fh.write("%s,%d,%.2f\n" % (name, lab, vol))
    return txt_file
