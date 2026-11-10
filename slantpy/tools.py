"""Running the external registration / bias-correction binaries.

The pipeline still needs three external programs that have no drop-in Python
equivalent: NiftyReg (reg_aladin, reg_transform, reg_resample) and ANTs
(N4BiasFieldCorrection, and for the legacy post-processing antsRegistration
and antsApplyTransforms). They ship in the SLANT image under
/extra/full-multi-atlas. Two ways to run them:

* natively (``docker_image=None``): from a copy of that folder on the host, or,
  if the binaries are not there, from PATH (conda packages ``niftyreg``, ``ants``)
* inside the original image (``docker_image='pipeline/slant:v1'``); every
  directory that holds an argument file is mounted at the same path.
"""
import os
import shlex
import shutil
import subprocess


class ToolError(RuntimeError):
    pass


class Tools:
    def __init__(self, extra_dir, docker_image=None, threads=1, docker_cmd="docker"):
        self.docker_image = docker_image
        self.docker_cmd = docker_cmd
        self.threads = int(threads)
        # where the tools (and the MNI template they read) live, as seen by the tools
        self.root = "/extra" if docker_image else os.path.abspath(extra_dir)
        fma = os.path.join(self.root, "full-multi-atlas")
        self.niftyreg_bin = os.path.join(fma, "niftyreg", "bin")
        self.ants_bin = os.path.join(fma, "ANTs-bin", "bin")
        self.mni = os.path.join(fma, "MNI", "average305_t1_tal_lin.nii.gz")
        self.mni_mask = os.path.join(fma, "MNI", "average305_t1_tal_lin_mask.nii.gz")

    def _find(self, bundled_dir, name):
        """The binary bundled under --extra-dir if it is there, otherwise the one
        on PATH (e.g. niftyreg / ants installed in the conda environment)."""
        bundled = os.path.join(bundled_dir, name)
        if self.docker_image or os.path.exists(bundled):
            return bundled
        return shutil.which(name) or bundled

    def niftyreg(self, name):
        return self._find(self.niftyreg_bin, name)

    def ants(self, name):
        return self._find(self.ants_bin, name)

    def _env(self):
        # The released image pins ITK to one thread so N4 is reproducible.
        return {
            "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": str(self.threads),
            "OMP_NUM_THREADS": str(self.threads),
        }

    def run(self, cmd, log=None, mounts=()):
        """Run ``cmd`` (list of str). ``mounts``: existing host directories the
        command reads or writes; in docker mode they are bind-mounted at the same
        path. The pipeline only ever passes the subject's output folder."""
        cmd = [str(c) for c in cmd]
        env = self._env()
        if self.docker_image:
            full = [self.docker_cmd, "run", "--rm"]
            # extra "docker run" arguments, e.g. SLANTPY_DOCKER_RUN_ARGS="--cpuset-cpus=0-7"
            full += shlex.split(os.environ.get("SLANTPY_DOCKER_RUN_ARGS", ""))
            if hasattr(os, "getuid"):
                full += ["-u", "%d:%d" % (os.getuid(), os.getgid())]
            for k, v in env.items():
                full += ["-e", "%s=%s" % (k, v)]
            for m in sorted({os.path.abspath(m) for m in mounts}):
                if not os.path.isdir(m):       # never let the docker daemon create host paths
                    raise ToolError("refusing to mount a directory that does not exist: %s" % m)
                full += ["-v", "%s:%s" % (m, m)]
            full += [self.docker_image] + cmd
            run_env = None
        else:
            if not os.path.exists(cmd[0]):
                raise ToolError("external tool not found: %s" % cmd[0])
            full = cmd
            run_env = dict(os.environ, **env)
        proc = subprocess.run(full, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              env=run_env, universal_newlines=True)
        if log:
            with open(log, "a") as fh:
                fh.write("$ %s\n%s\n" % (" ".join(shlex.quote(c) for c in full), proc.stdout))
        if proc.returncode != 0:
            raise ToolError("command failed (exit %d): %s\n%s" % (
                proc.returncode, " ".join(shlex.quote(c) for c in full), proc.stdout[-2000:]))
        return proc.stdout
