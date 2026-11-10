"""One-page QA PDF, after generate_brianseg_pdf_xiong_light.m (needs matplotlib).

Same content as the Matlab report (three orthogonal slices of the processed
MNI-space image, plain and with the segmentation at 60 % opacity, plus an info
box); the layout is close to, not identical with, the original.
"""
import datetime

import nibabel as nib
import numpy as np

from .labels import CMAP


def _overlay(img, seg, alpha, lim, cmap_idx):
    g = np.clip((img - lim[0]) / (lim[1] - lim[0]), 0, 1)
    rgb = np.repeat(g[..., None], 3, axis=2)
    m = seg > 0
    rgb[m] = alpha * cmap_idx[seg[m]] + (1 - alpha) * rgb[m]
    return rgb


def write_pdf(image_file, seg_file, pdf_file, subject=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = np.asanyarray(nib.load(image_file).dataobj).astype(np.float64)
    seg = np.asanyarray(nib.load(seg_file).dataobj).astype(np.int64)
    # consecutive label numbers, colours spread over the colour map as in Matlab
    ul, seg = np.unique(seg, return_inverse=True)
    seg = seg.reshape(img.shape)
    cmap = np.asarray(CMAP)
    cmap_idx = cmap[np.round(np.linspace(0, len(cmap) - 1, len(ul))).astype(int)]
    lim = np.percentile(img[seg > 0], [1, 99]) if (seg > 0).any() else (img.min(), img.max())
    nx, ny, nz = img.shape
    # the Matlab report shows the sagittal slice 10 voxels off the mid-line
    cuts = [("Axial", lambda a: np.rot90(a[:, :, nz // 2])),
            ("Coronal", lambda a: np.rot90(a[:, ny // 2, :])),
            ("Sagittal", lambda a: np.rot90(a[min(nx // 2 + 10, nx - 1), :, :]))]

    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("SLANT Deep Whole Brain Segmentation Result", fontsize=20, y=0.95)
    for row, (alpha, name) in enumerate(((0.0, "Raw Image"), (0.6, "Slicewise\nSegmentation"))):
        for col, (title, cut) in enumerate(cuts):
            ax = fig.add_axes([0.10 + col * 0.29, 0.62 - row * 0.27, 0.27, 0.25])
            ax.imshow(_overlay(cut(img), cut(seg), alpha, lim, cmap_idx), interpolation="nearest")
            ax.set_axis_off()
            if row == 0:
                ax.set_title(title, fontsize=10)
        fig.text(0.06, 0.745 - row * 0.27, name, rotation=90, ha="center", va="center", fontsize=12)
    ax = fig.add_axes([0.10, 0.16, 0.85, 0.14])
    ax.set_xticks([]); ax.set_yticks([])
    for y, (k, v) in zip((0.85, 0.65, 0.45, 0.25), (
            ("Project", "DeepWholeBrain"), ("Subject", subject), ("Experiment", "SLANT (slantpy)"),
            ("Date of run", datetime.datetime.now().strftime("%c")))):
        ax.text(0.02, y, k + ":", fontsize=9, fontweight="bold", va="center")
        ax.text(0.20, y, v, fontsize=9, va="center")
    fig.text(0.06, 0.23, "INFO", rotation=90, ha="center", va="center", fontsize=12)
    fig.savefig(pdf_file)
    plt.close(fig)
    return pdf_file
