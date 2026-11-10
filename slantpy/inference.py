"""Tile-wise network inference and majority-vote fusion.

Replaces the 27 separate ``python test.py --piece=...`` processes, the
per-tile NIfTI files, and the Matlab functions project_space.m,
reassign_braincolor.m and get_MV_fused_image.m.
"""
import os

import numpy as np

from .labels import LABELS, MNI_SHAPE, N_CLASSES, PIECE_MAP, PIECES


def weight_path(model_dir, piece, epoch=27):
    base = os.path.join(model_dir, piece, "finetune_out", "models", "model_epoch_%04d" % epoch)
    for ext in (".pth", ".npz"):
        if os.path.exists(base + ext):
            return base + ext
    raise FileNotFoundError("no weights for tile %s: %s.pth" % (piece, base))


def tile_input(volume, piece):
    """Crop one tile and scale it to 0..255 with its own min/max (as the data loader does).
    Returns a float32 array ordered (z, x, y), the network's layout."""
    x0, x1, y0, y1, z0, z1 = PIECE_MAP[piece]
    t = np.asarray(volume[x0:x1, y0:y1, z0:z1], dtype=np.float32)
    lo, hi = t.min(), t.max()
    if not hi > lo:
        raise ValueError("tile %s is constant; is the input image empty?" % piece)
    t = (t - lo) / (hi - lo) * np.float32(255.0)
    return np.ascontiguousarray(np.transpose(t, (2, 0, 1)))


# Free GPU memory wanted before a GPU is used automatically. Estimates with
# headroom (CPU peaks are 1.7 / 4.2 GB per tile; cuDNN adds workspace on top).
GPU_NEED_GB = {"lean": 4.0, "literal": 8.0}


def choose_gpu(free_gb, need_gb):
    """Index of the GPU with the most free memory if it has ``need_gb``, else None."""
    if not free_gb:
        return None
    best = max(range(len(free_gb)), key=lambda i: free_gb[i])
    return best if free_gb[best] >= need_gb else None


def gpu_free_gb():
    """Free memory (GB) of every GPU PyTorch can see; a GPU that cannot even be
    queried (e.g. completely full) counts as 0."""
    import torch
    free = []
    for i in range(torch.cuda.device_count()):
        try:
            free.append(torch.cuda.mem_get_info(i)[0] / 1e9)
        except Exception:
            free.append(0.0)
    return free


def pick_device(need_gb, log=print):
    """'cuda:N' for the freest GPU with enough memory, otherwise 'cpu'."""
    import torch
    if not torch.cuda.is_available():
        return "cpu"
    free = gpu_free_gb()
    idx = choose_gpu(free, need_gb)
    log("free GPU memory (GB): %s; need %.1f -> using %s" % (
        ", ".join("cuda:%d %.1f" % (i, f) for i, f in enumerate(free)), need_gb,
        "cpu" if idx is None else "cuda:%d" % idx))
    return "cpu" if idx is None else "cuda:%d" % idx


def segment_tiles(volume, model_dir, device=None, lean=True, half=False, epoch=27,
                  pieces=PIECES, threads=None, min_gpu_free_gb=None, log=print):
    """Yield ``(piece, classes)``; ``classes`` is uint8 (x, y, z) with network class
    indices 0..132 for that tile.

    ``device=None`` picks the GPU with the most free memory if it has enough,
    else the CPU, and moves to the CPU if a GPU runs out of memory part-way
    (someone else may start a job). An explicit ``device`` is used as given.
    """
    import torch
    from .model import UNet3D, load_weights

    if volume.shape != MNI_SHAPE:
        raise ValueError("expected a %s volume, got %s" % (MNI_SHAPE, volume.shape))
    if threads:
        torch.set_num_threads(int(threads))
    auto = device is None
    if auto:
        need = min_gpu_free_gb if min_gpu_free_gb is not None else GPU_NEED_GB["lean" if lean else "literal"]
        device = pick_device(need, log)
    device = torch.device(device)
    oom = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError
    model = UNet3D(1, N_CLASSES)
    try:
        model.to(device)
    except oom:
        if not auto:
            raise
        log("GPU out of memory while loading the model, using the CPU")
        device = torch.device("cpu")
    model.train()                       # see model.py: the original tests in train mode
    labels = model.labels_lean if lean else model.labels

    def run(piece):
        x = torch.from_numpy(tile_input(volume, piece))[None, None].to(device)
        use_amp = bool(half) and device.type == "cuda"
        with torch.inference_mode(), torch.autocast(device.type, enabled=use_amp):
            return labels(x)[0].cpu().numpy()                         # (z, x, y)

    for piece in pieces:
        load_weights(model, weight_path(model_dir, piece, epoch))
        try:
            cls = run(piece)
        except oom:
            if not (auto and device.type == "cuda"):
                raise
            log("GPU out of memory at tile %s, continuing on the CPU" % piece)
            device = torch.device("cpu")
            model.to(device)
            torch.cuda.empty_cache()
            cls = run(piece)
        log("tile %s done (%s)" % (piece, device))
        yield piece, np.transpose(cls, (1, 2, 0))                     # -> (x, y, z)


def fuse_majority(tiles, shape=MNI_SHAPE, slab=16):
    """Majority vote over overlapping tiles; ties go to the lowest label, exactly
    like Matlab ``mode``. ``tiles``: iterable of (piece, uint8 class array).
    Returns a uint8 volume of BrainCOLOR label numbers.

    Votes are counted per x-slab so the peak is slab*220*156*133 bytes (73 MB for
    16) rather than a 27-volume stack that is looped over voxel by voxel.
    """
    tiles = list(tiles)
    lut = np.array((0,) + LABELS, dtype=np.uint8)
    out = np.zeros(shape, dtype=np.uint8)
    for s0 in range(0, shape[0], slab):
        s1 = min(s0 + slab, shape[0])
        counts = np.zeros((s1 - s0,) + tuple(shape[1:]) + (N_CLASSES,), dtype=np.uint8)
        for piece, cls in tiles:
            x0, x1, y0, y1, z0, z1 = PIECE_MAP[piece]
            a, b = max(x0, s0), min(x1, s1)
            if a >= b:
                continue
            idx = cls[a - x0:b - x0, :, :, None].astype(np.intp)
            view = counts[a - s0:b - s0, y0:y1, z0:z1]
            np.put_along_axis(view, idx, np.take_along_axis(view, idx, -1) + 1, -1)
        out[s0:s1] = lut[counts.argmax(-1)]
    return out
