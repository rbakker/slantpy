#!/usr/bin/env python3
"""Peak memory and time of one tile through the network, literal vs lean.

    python scripts/benchmark_memory.py                 # random weights, CPU or GPU
    python scripts/benchmark_memory.py --weights /extra/model_dir/2_2_2/finetune_out/models/model_epoch_0027.pth

Each mode runs in its own process so the peaks do not contaminate each other.
With real weights it also reports how many voxels the two modes label differently.
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def one(mode, device, weights, half, out):
    import resource
    import numpy as np
    import torch
    from slantpy.model import UNet3D, load_weights
    torch.manual_seed(0)
    dev = torch.device(device)
    model = UNet3D()
    if weights:
        load_weights(model, weights)
    model = model.to(dev).train()
    x = (torch.rand(1, 1, 88, 96, 128, generator=torch.Generator().manual_seed(1)) * 255).to(dev)
    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    t0 = time.time()
    with torch.inference_mode(), torch.autocast(dev.type, enabled=half and dev.type == "cuda"):
        cls = (model.labels_lean if mode == "lean" else model.labels)(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    res = {"mode": mode, "seconds": time.time() - t0, "rss_before_gb": base,
           "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}
    if dev.type == "cuda":
        res["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    np.save(out, cls.to(torch.uint8).cpu().numpy())
    print(json.dumps(res))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--modes", nargs="+", default=["literal", "lean"])
    ap.add_argument("--_child", default=None)
    ap.add_argument("--_out", default=None)
    a = ap.parse_args()
    if a._child:
        one(a._child, a.device, a.weights, a.half, a._out)
        sys.exit(0)
    import numpy as np
    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    outs = {}
    for mode in a.modes:
        out = "/tmp/slantpy_bench_%s_%d.npy" % (mode, os.getpid())
        cmd = [sys.executable, __file__, "--_child", mode, "--_out", out, "--device", device]
        cmd += (["--weights", a.weights] if a.weights else []) + (["--half"] if a.half else [])
        p = subprocess.run(cmd, stdout=subprocess.PIPE, universal_newlines=True)
        if p.returncode != 0:
            print("%-8s failed (exit %d; out of memory?)" % (mode, p.returncode))
            continue
        r = json.loads(p.stdout.strip().splitlines()[-1])
        outs[mode] = np.load(out); os.remove(out)
        print("%-8s %6.1f s   peak RSS %5.2f GB (%.2f GB before the forward pass)%s" % (
            mode, r["seconds"], r["peak_rss_gb"], r["rss_before_gb"],
            "   peak GPU %.2f GB" % r["peak_gpu_gb"] if "peak_gpu_gb" in r else ""))
    if len(outs) == 2:
        a_, b_ = outs.values()
        print("voxels labelled differently: %d of %d" % ((a_ != b_).sum(), a_.size))
