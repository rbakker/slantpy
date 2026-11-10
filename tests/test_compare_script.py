import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("torch")
from test_torch_parts import make_fixture  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def test_compare_script_with_fake_docker(tmp_path):
    extra, t1, _, _ = make_fixture(tmp_path)
    docker = str(tmp_path / "docker")
    with open(docker, "w") as fh:
        fh.write("#!/bin/sh\nexec %s %s \"$@\"\n" % (sys.executable, os.path.join(HERE, "fake_tools", "fake_docker.py")))
    os.chmod(docker, 0o755)
    env = dict(os.environ, FAKE_SLANT_EXTRA=str(extra), SLANTPY_PIECES="2_2_2",
               PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    work = str(tmp_path / "work")
    p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "compare_with_docker.py"), t1,
                        "--workdir", work, "--docker", docker, "--extra-dir", str(extra),
                        "--tools", "native", "--cpuset", "auto:2"],
                       env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    print(p.stdout)
    assert p.returncode == 0, p.stdout
    s = json.load(open(os.path.join(work, "report", "summary.json")))
    mimic, improved = s["t1a/mimic"], s["t1a/improved"]
    # the 'reference' here is slantpy itself, so the mimic run must reproduce it exactly
    assert mimic["affine_to_MNI"]["max_abs_diff"] == 0
    assert mimic["normalised_T1 (network input)"]["max_abs_diff"] == 0
    assert mimic["tiles (network output)"]["tiles_compared"] == 1
    assert mimic["tiles (network output)"]["worst_fraction_different"] == 0
    assert mimic["fused_labels_MNI_space"]["voxels_different"] == 0
    assert mimic["final_labels_native_space"]["voxels_different"] == 0
    assert mimic["label_volumes"]["max_abs_diff_percent"] == 0
    assert mimic["reference_run"]["peak_mem_gb"] > 1 and mimic["new_run"]["peak_mem_gb"] > 0.5
    assert improved["fused_labels_MNI_space"]["agreement_all_voxels"] > 0.999
    assert improved["final_labels_native_space"]["agreement_all_voxels"] > 0.97
    assert os.path.exists(improved["per_label_csv"])
    # --cpuset: both sides pinned to the same cores, choice remembered, stage times reported
    cpus = open(os.path.join(work, "cpuset.txt")).read().strip()
    assert mimic["cpuset"] == cpus and "--cpuset-cpus " + cpus in open(
        os.path.join(work, "reference_cpuset", "t1a", "docker_log.txt")).read() + p.stdout
    assert set(mimic["stage_seconds"]["new"]) == {"preprocessing", "segmentation", "postprocessing"}
    assert os.path.isdir(os.path.join(work, "mimic_cpuset")) and "time per stage" in p.stdout

