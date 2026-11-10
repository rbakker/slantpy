#!/usr/bin/env python3
"""Run the current docker-based SLANT and the new Python pipeline on the same T1
and report how well they agree, stage by stage.

    python scripts/compare_with_docker.py /data/sub01/t1.nii.gz --workdir /scratch/slant_cmp

What it does
  1. reference : runs the docker image exactly like the existing wrapper
                 (INPUTS/OUTPUTS mounts, /extra/run_deep_brain_seg.sh) and keeps
                 the whole OUTPUTS tree.
  2. resources : copies weights, atlas model, MNI template (and optionally the
                 NiftyReg/ANTs binaries) out of the image, once.
  3. new       : runs ``python -m slantpy run`` for each requested variant
                   mimic    = literal network + legacy reg_aladin/ANTs post-processing
                   improved = lean network + direct inverse-affine post-processing
  4. compare   : affine, registered image, N4 image, normalised image, every tile,
                 fused MNI-space labels, final native-space labels, label volumes,
                 wall time and peak memory.

Nothing is deleted; rerunning skips steps whose outputs already exist
(use --force to redo them).
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

import nibabel as nib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from slantpy.labels import LABEL_NAMES, LABELS, PIECES  # noqa: E402
from slantpy.pipeline import subject_name  # noqa: E402

VARIANTS = {
    "mimic": ["--postproc", "legacy", "--forward", "literal"],
    "improved": ["--postproc", "direct", "--interp", "linear", "--forward", "lean"],
}
REF_FUSED = ("working_dir/all_piece/5000_fusion27_mv/test_out/seg_output/epoch_0027/"
             "seg_orig_final/target_processed_seg.nii.gz")
REF_TILE = "working_dir/all_piece/%s/test_out/seg_output/epoch_0027/seg/target_processed_seg.nii.gz"


# ----------------------------------------------------------------------------- running
def _parse_mem(text):
    m = re.match(r"\s*([\d.]+)\s*([KMGT]i?B|B)", text)
    if not m:
        return 0.0
    scale = {"B": 1e-9, "KiB": 2 ** 10 / 1e9, "MiB": 2 ** 20 / 1e9, "GiB": 2 ** 30 / 1e9,
             "TiB": 2 ** 40 / 1e9, "KB": 1e-6, "MB": 1e-3, "GB": 1.0, "TB": 1e3}
    return float(m.group(1)) * scale[m.group(2)]


class DockerMemWatch(threading.Thread):
    """Poll ``docker stats`` for a named container and remember the peak (GB)."""

    def __init__(self, name, docker="docker", every=2.0):
        super().__init__(daemon=True)
        self.name, self.docker, self.every = name, docker, every
        self.peak, self._halt = 0.0, threading.Event()

    def run(self):
        while not self._halt.is_set():
            try:
                out = subprocess.run([self.docker, "stats", "--no-stream", "--format",
                                      "{{.MemUsage}}", self.name], stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, universal_newlines=True, timeout=20).stdout
                self.peak = max(self.peak, _parse_mem(out))
            except Exception:
                pass
            self._halt.wait(self.every)

    def stop(self):
        self._halt.set()
        self.join(timeout=30)
        return self.peak


def parse_cpu_list(text):
    """'0-3,8,10-11' -> [0, 1, 2, 3, 8, 10, 11]"""
    cpus = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            cpus += list(range(int(a), int(b) + 1))
        elif part:
            cpus.append(int(part))
    return sorted(set(cpus))


def format_cpu_list(cpus):
    return ",".join(str(c) for c in sorted(cpus))


def _cpu_times():
    out = {}
    with open("/proc/stat") as fh:
        for line in fh:
            m = re.match(r"cpu(\d+)\s+(.*)", line)
            if m:
                v = [int(x) for x in m.group(2).split()]
                out[int(m.group(1))] = (sum(v[:8]), v[3] + v[4])        # total, idle + iowait
    return out


def _siblings(cpu):
    try:
        with open("/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list" % cpu) as fh:
            return tuple(parse_cpu_list(fh.read()))
    except OSError:
        return (cpu,)


def least_used_cpus(n, seconds=2.0, usage=None, siblings=_siblings, allowed=None):
    """The ``n`` least busy CPUs, one per physical core, measured over ``seconds``.
    A physical core is as busy as its busiest hyper-thread, so the idle sibling
    of a loaded core is not picked before a completely idle core."""
    if usage is None:
        a = _cpu_times()
        time.sleep(seconds)
        b = _cpu_times()
        usage = {c: 1.0 - (b[c][1] - a[c][1]) / max(b[c][0] - a[c][0], 1) for c in b if c in a}
    if allowed is None:
        allowed = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else set(usage)
    cores = {}
    for c in sorted(usage):
        if c in allowed:
            cores.setdefault(siblings(c), []).append(c)
    ranked = sorted(cores.items(), key=lambda kv: (max(usage.get(c, 0.0) for c in kv[0] if c in usage), kv[1][0]))
    return sorted(members[0] for _, members in ranked[:n])


def resolve_cpuset(spec, work):
    """'auto', 'auto:N' or an explicit list such as '0-7'. An automatic choice is
    written to <workdir>/cpuset.txt and reused, so every run of a comparison
    gets the same cores."""
    if not spec.startswith("auto"):
        return parse_cpu_list(spec)
    memo = os.path.join(work, "cpuset.txt")
    if os.path.exists(memo):
        cpus = parse_cpu_list(open(memo).read())
        print("[cpuset] reusing %s from %s (delete the file to choose again)" % (format_cpu_list(cpus), memo))
        return cpus
    n = int(spec.split(":")[1]) if ":" in spec else 8
    cpus = least_used_cpus(n)
    if len(cpus) < n:
        print("[cpuset] only %d physical cores available, using all of them" % len(cpus))
    with open(memo, "w") as fh:
        fh.write(format_cpu_list(cpus) + "\n")
    print("[cpuset] least used cores: %s" % format_cpu_list(cpus))
    return cpus


def stage_times(log_file):
    """{'preprocessing': seconds, ...} from the '*******<stage> time: N seconds' lines
    that both pipelines print."""
    try:
        text = open(log_file).read()
    except OSError:
        return {}
    return {k.strip(): float(v) for k, v in re.findall(r"\*{7}\s*(.*?) time: ([\d.]+) seconds", text)}


def _stats_from_log(log_file):
    """Run time from the '*******... time: N seconds' lines the original script prints
    (used when an earlier invocation did not get to save run_stats.json)."""
    try:
        secs = [float(x) for x in re.findall(r"\*{7}.*time: ([\d.]+) seconds", open(log_file).read())]
    except OSError:
        return {}
    return {"seconds": sum(secs)} if secs else {}


def run_reference(t1, ref_dir, image, docker, gpu, force, cpus=None):
    """The existing wrapper's docker call, with OUTPUTS kept."""
    name = subject_name(t1)
    inputs, outputs = os.path.join(ref_dir, "INPUTS"), os.path.join(ref_dir, "OUTPUTS")
    final = os.path.join(outputs, "FinalResult", "%s_seg.nii.gz" % name)
    stats_file = os.path.join(ref_dir, "run_stats.json")
    if os.path.exists(final) and not force:
        print("[reference] found %s, not rerunning docker" % final)
        if os.path.exists(stats_file):
            return outputs, json.load(open(stats_file))
        return outputs, _stats_from_log(os.path.join(ref_dir, "docker_log.txt"))
    for d in (inputs, outputs):
        os.makedirs(d, exist_ok=True)
    shutil.copy(t1, os.path.join(inputs, name + ".nii.gz"))
    cname = "slant_ref_%d" % os.getpid()
    cmd = [docker, "run", "--rm", "--name", cname]
    if gpu:
        cmd += ["--gpus", "all" if gpu == "all" else "device=%s" % gpu]
    if cpus:
        cmd += ["--cpuset-cpus", format_cpu_list(cpus)]
    cmd += ["-v", "%s:/INPUTS/" % inputs, "-v", "%s:/OUTPUTS/" % outputs, image, "/bin/bash", "-c",
            "/extra/run_deep_brain_seg.sh; chmod -R 777 /INPUTS/; chmod -R 777 /OUTPUTS/"]
    print("[reference] " + " ".join(cmd))
    watch = DockerMemWatch(cname, docker)
    t0 = time.time()
    watch.start()
    with open(os.path.join(ref_dir, "docker_log.txt"), "w") as log:
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
    stats = {"seconds": time.time() - t0, "peak_mem_gb": watch.stop()}
    if rc != 0 or not os.path.exists(final):
        sys.exit("[reference] docker run failed, see %s" % os.path.join(ref_dir, "docker_log.txt"))
    json.dump(stats, open(stats_file, "w"))
    return outputs, stats


def run_new(t1, out_dir, extra_dir, variant_args, common_args, force, cpus=None):
    name = subject_name(t1)
    final = os.path.join(out_dir, "FinalResult", "%s_seg.nii.gz" % name)
    stats_file = os.path.join(out_dir, "run_stats.json")
    if os.path.exists(final) and not force:
        print("[new] found %s, not rerunning" % final)
        return json.load(open(stats_file)) if os.path.exists(stats_file) else {}
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, "-m", "slantpy", "run", t1, out_dir, "--extra-dir", extra_dir,
           "--keep-tiles"] + variant_args + common_args
    print("[new] " + " ".join(cmd))
    # run through a tiny wrapper so we get the child's peak RSS (python + native tools)
    pin = "import os;os.sched_setaffinity(0,{%s});" % format_cpu_list(cpus) if cpus else ""
    wrapper = (pin + "import resource,subprocess,sys,json,time;t=time.time();"
               "rc=subprocess.call(sys.argv[2:]);"
               "r=resource.getrusage(resource.RUSAGE_CHILDREN);"
               "json.dump({'seconds':time.time()-t,'peak_mem_gb':r.ru_maxrss/1e6},open(sys.argv[1],'w'));"
               "sys.exit(rc)")
    env = dict(os.environ, PYTHONPATH=os.path.dirname(HERE) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    if cpus:        # containers started for the external tools do not inherit our affinity
        env["SLANTPY_DOCKER_RUN_ARGS"] = "--cpuset-cpus=%s" % format_cpu_list(cpus)
    with open(os.path.join(out_dir, "log.txt"), "w") as log:
        rc = subprocess.call([sys.executable, "-c", wrapper, stats_file] + cmd,
                             stdout=log, stderr=subprocess.STDOUT, env=env)
    if rc != 0:
        sys.exit("[new] run failed, see %s" % os.path.join(out_dir, "log.txt"))
    return json.load(open(stats_file))


# ----------------------------------------------------------------------------- comparing
def load(path):
    return np.asanyarray(nib.load(path).dataobj)


def image_diff(a_file, b_file):
    if not (os.path.exists(a_file) and os.path.exists(b_file)):
        return None
    a, b = load(a_file).astype(np.float64), load(b_file).astype(np.float64)
    if a.shape != b.shape:
        return {"shape_mismatch": [list(a.shape), list(b.shape)]}
    d = np.abs(a - b)
    rng = float(a.max() - a.min()) or 1.0
    return {"max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
            "max_abs_diff_rel_to_range": float(d.max() / rng),
            "correlation": float(np.corrcoef(a.ravel(), b.ravel())[0, 1])}


def label_agreement(a, b):
    """Voxel agreement and per-label Dice between two integer label arrays."""
    a = np.rint(a).astype(np.int64).ravel()
    b = np.rint(b).astype(np.int64).ravel()
    n = int(max(a.max(), b.max())) + 1
    na, nb = np.bincount(a, minlength=n), np.bincount(b, minlength=n)
    inter = np.bincount(a[a == b], minlength=n)
    dice = {}
    for lab in LABELS:
        if lab < n and na[lab] + nb[lab] > 0:
            dice[lab] = 2.0 * inter[lab] / (na[lab] + nb[lab])
    fg = (a > 0) | (b > 0)
    vals = np.array(list(dice.values())) if dice else np.array([np.nan])
    worst = min(dice, key=dice.get) if dice else None
    return {"voxels_different": int((a != b).sum()),
            "agreement_all_voxels": float((a == b).mean()),
            "agreement_in_brain": float((a[fg] == b[fg]).mean()) if fg.any() else 1.0,
            "mean_dice": float(np.nanmean(vals)), "median_dice": float(np.nanmedian(vals)),
            "min_dice": float(np.nanmin(vals)),
            "min_dice_label": None if worst is None else "%d %s" % (worst, LABEL_NAMES[LABELS.index(worst)]),
            "labels_compared": len(dice)}, dice


def read_volumes(txt):
    out = {}
    with open(txt) as fh:
        next(fh)
        for line in fh:
            name, lab, vol = line.rstrip("\n").rsplit(",", 2)
            out[int(lab)] = float(vol)
    return out


def compare(ref_out, new_out, name, report_dir, tag):
    ref_sub, new_sub = os.path.join(ref_out, name), os.path.join(new_out, name)
    res = {}

    fa = os.path.join(ref_sub, "MNI-registration", "target_MNI_aff.mtx")
    fb = os.path.join(new_sub, "MNI-registration", "target_MNI_aff.mtx")
    if os.path.exists(fa) and os.path.exists(fb):
        res["affine_to_MNI"] = {"max_abs_diff": float(np.abs(np.loadtxt(fa) - np.loadtxt(fb)).max())}
    for key, rel in (("registered_T1", "MNI-registration/target_MNI.nii.gz"),
                     ("N4_corrected", "N4-targets/target_MNI.nii.gz"),
                     ("normalised_T1 (network input)", "target_processed.nii.gz")):
        d = image_diff(os.path.join(ref_sub, rel), os.path.join(new_sub, rel))
        if d:
            res[key] = d

    # individual tiles: reference stores (z, x, y) class indices with an identity affine
    diffs, n_tiles = [], 0
    for p in PIECES:
        fr, fn = os.path.join(ref_sub, REF_TILE % p), os.path.join(new_sub, "tiles", p + ".npy")
        if os.path.exists(fr) and os.path.exists(fn):
            r = np.transpose(load(fr), (1, 2, 0))
            diffs.append(float((r != np.load(fn)).mean()))
            n_tiles += 1
    if n_tiles:
        res["tiles (network output)"] = {
            "tiles_compared": n_tiles, "mean_fraction_voxels_different": float(np.mean(diffs)),
            "worst_tile": PIECES[int(np.argmax(diffs))], "worst_fraction_different": float(np.max(diffs))}

    fr, fn = os.path.join(ref_sub, REF_FUSED), os.path.join(new_sub, "target_processed_seg.nii.gz")
    if os.path.exists(fr) and os.path.exists(fn):
        res["fused_labels_MNI_space"], _ = label_agreement(load(fr), load(fn))

    fr = os.path.join(ref_out, "FinalResult", "%s_seg.nii.gz" % name)
    fn = os.path.join(new_out, "FinalResult", "%s_seg.nii.gz" % name)
    a, b = load(fr), load(fn)
    if a.shape != b.shape:
        res["final_labels_native_space"] = {"shape_mismatch": [list(a.shape), list(b.shape)]}
        dice = {}
    else:
        res["final_labels_native_space"], dice = label_agreement(a, b)

    vr = read_volumes(os.path.join(ref_out, "FinalVolTxt", "%s_label_volumes.txt" % name))
    vn = read_volumes(os.path.join(new_out, "FinalVolTxt", "%s_label_volumes.txt" % name))
    rel = [abs(vn[l] - vr[l]) / vr[l] * 100 for l in LABELS if vr.get(l, 0) > 0]
    res["label_volumes"] = {"mean_abs_diff_percent": float(np.mean(rel)) if rel else 0.0,
                            "max_abs_diff_percent": float(np.max(rel)) if rel else 0.0,
                            "total_brain_volume_diff_percent":
                                float(abs(sum(vn.values()) - sum(vr.values())) / sum(vr.values()) * 100)}

    csv_file = os.path.join(report_dir, "%s_%s_per_label.csv" % (name, tag))
    with open(csv_file, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "name", "volume_reference_mm3", "volume_new_mm3", "volume_diff_percent",
                    "dice_final_native"])
        for lab, lname in zip(LABELS, LABEL_NAMES):
            r, n = vr.get(lab, 0.0), vn.get(lab, 0.0)
            w.writerow([lab, lname, "%.2f" % r, "%.2f" % n,
                        "%.3f" % ((n - r) / r * 100) if r else "", "%.4f" % dice[lab] if lab in dice else ""])
    res["per_label_csv"] = csv_file
    return res


def print_report(name, tag, res, ref_stats, new_stats, ref_label=""):
    print("\n" + "=" * 78)
    print("%s : docker reference  vs  slantpy '%s'" % (name, tag))
    print("=" * 78)
    for key, val in res.items():
        if not isinstance(val, dict) or key in ("reference_run", "new_run", "stage_seconds"):
            continue
        print(key)
        for k, v in val.items():
            print("    %-34s %s" % (k, ("%.6g" % v) if isinstance(v, float) else v))
    print("resources")
    for label, s in (("docker reference" + ref_label, ref_stats), ("slantpy " + tag, new_stats)):
        if s:
            mem = "peak memory %.1f GB" % s["peak_mem_gb"] if s.get("peak_mem_gb") else "peak memory not recorded"
            print("    %-34s %6.1f min, %s" % (label, s.get("seconds", 0) / 60, mem))
    st = res.get("stage_seconds")
    if st and (st["reference"] or st["new"]):
        print("time per stage (seconds)              docker    slantpy")
        for k in list(st["reference"]) + [k for k in st["new"] if k not in st["reference"]]:
            fmt = lambda d: ("%9.0f" % d[k]) if k in d else "        -"
            print("    %-30s %s  %s" % (k, fmt(st["reference"]), fmt(st["new"])))
    if res.get("cpuset"):
        print("both pipelines were restricted to CPUs %s (%d threads)" % (res["cpuset"], res["cpuset"].count(",") + 1))
    print("per-label table: %s" % res.get("per_label_csv"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("t1", nargs="+", help="T1 image(s), .nii.gz")
    ap.add_argument("--workdir", required=True, help="where all runs and reports are kept")
    ap.add_argument("--image", default="pipeline/slant:v1", help="docker image of the current pipeline")
    ap.add_argument("--docker", default="docker", help="docker executable")
    ap.add_argument("--gpu", nargs="?", const="all", default=None, metavar="DEVICE",
                    help="run the reference container on a GPU: '--gpu=all' or '--gpu=1' for "
                         "device 1 (write it with '=' so it cannot swallow the T1 argument). Needs the GPU build of the image and the NVIDIA container toolkit. "
                         "This reference is kept in <workdir>/reference_gpu, next to the CPU one.")
    ap.add_argument("--cpuset", default=None, metavar="SPEC",
                    help="CPU-against-CPU timing: restrict the docker reference and the slantpy runs to "
                         "the same cores. 'auto' picks the 8 least used physical cores, 'auto:N' picks N, "
                         "or give a list such as 0-7. Implies --device cpu for slantpy unless --device is "
                         "given. These runs are kept in separate folders (suffix _cpuset).")
    ap.add_argument("--extra-dir", default=None,
                    help="where the image's resources are / will be extracted (default <workdir>/extra)")
    ap.add_argument("--tools", choices=("docker", "native"), default="docker",
                    help="run NiftyReg/ANTs for the new pipeline inside the image (default, always "
                         "works) or natively from the extracted binaries")
    ap.add_argument("--variants", nargs="+", default=["mimic", "improved"], choices=sorted(VARIANTS))
    ap.add_argument("--device", default=None,
                    help="device for the new pipeline (cpu, cuda:1 ...); default: freest GPU with "
                         "enough memory, else CPU")
    ap.add_argument("--threads", type=int, default=None, help="PyTorch CPU threads for the new pipeline")
    ap.add_argument("--share-preprocessing", action="store_true",
                    help="start the new runs from the reference's registration/N4/normalised image, "
                         "to compare the network and post-processing in isolation")
    ap.add_argument("--force", action="store_true", help="redo runs even if outputs exist")
    a = ap.parse_args()

    work = os.path.abspath(a.workdir)
    extra = os.path.abspath(a.extra_dir or os.path.join(work, "extra"))
    report_dir = os.path.join(work, "report")
    os.makedirs(report_dir, exist_ok=True)

    if not os.path.isdir(os.path.join(extra, "model_dir")):
        from slantpy.pipeline import extract_resources
        print("[resources] extracting from %s to %s" % (a.image, extra))
        extract_resources(a.image, extra, binaries=(a.tools == "native"), docker_cmd=a.docker)

    cpus = resolve_cpuset(a.cpuset, work) if a.cpuset else None
    suffix = ("_gpu" if a.gpu else "") + ("_cpuset" if cpus else "")
    if cpus:
        a.device = a.device or "cpu"
        a.threads = a.threads or len(cpus)

    common = ["--no-pdf"]
    if cpus:
        common += ["--tool-threads", str(len(cpus))]
    if a.tools == "docker":
        common += ["--tools-docker-image", a.image]
    if a.device:
        common += ["--device", a.device]
    if a.threads:
        common += ["--threads", str(a.threads)]

    summary = {}
    for t1 in a.t1:
        t1 = os.path.abspath(t1)
        name = subject_name(t1)
        ref_dir = os.path.join(work, "reference" + suffix, name)
        ref_out, ref_stats = run_reference(t1, ref_dir, a.image, a.docker, a.gpu, a.force, cpus)
        for tag in a.variants:
            new_out = os.path.join(work, tag + ("_sharedpre" if a.share_preprocessing else "")
                                   + ("_cpuset" if cpus else ""), name)
            if a.force and os.path.isdir(new_out):
                shutil.rmtree(new_out)
            if a.share_preprocessing and not os.path.isdir(os.path.join(new_out, name)):
                dst = os.path.join(new_out, name)
                os.makedirs(dst, exist_ok=True)
                shutil.copy(os.path.join(ref_out, name, "target_processed.nii.gz"), dst)
                for d in ("MNI-registration", "N4-targets"):
                    shutil.copytree(os.path.join(ref_out, name, d), os.path.join(dst, d))
            new_stats = run_new(t1, new_out, extra, VARIANTS[tag], common, a.force, cpus)
            res = compare(ref_out, new_out, name, report_dir, tag)
            res["reference_run"], res["new_run"] = ref_stats, new_stats
            res["stage_seconds"] = {"reference": stage_times(os.path.join(ref_dir, "docker_log.txt")),
                                    "new": stage_times(os.path.join(new_out, "log.txt"))}
            if cpus:
                res["cpuset"] = format_cpu_list(cpus)
            print_report(name, tag, res, ref_stats, new_stats, " (GPU)" if a.gpu else " (CPU)")
            summary["%s/%s" % (name, tag)] = res
    out = os.path.join(report_dir, "summary.json")
    json.dump(summary, open(out, "w"), indent=2)
    print("\nsummary written to %s" % out)


if __name__ == "__main__":
    main()
