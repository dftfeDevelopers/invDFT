#!/usr/bin/env python3
"""
install.py -- build DFT-FE (and optionally invDFT) together with all of
its dependencies on an HPC machine.

Machine profiles (modules, compilers, GPU/ELPA flags, tested versions) live in
machines/<name>.json: OLCF Frontier, NERSC Perlmutter, UMich Great Lakes, and
"generic" for any other Linux cluster. A --config file can name a profile,
extend one ({"base": "frontier", ...}) or define one inline. See README.md.

Examples
--------
  # Frontier, AMD GPUs, also build invDFT (turns on PETSc/SLEPc automatically)
  python3 install.py --machine frontier --prefix /lustre/orion/<proj>/scratch/$USER/dftfe --invdft

  # Great Lakes, CPU only, dependencies in a separate prefix, MKL, with dftd4
  python3 install.py --machine greatlakes --prefix $HOME/dftfe \\
      --prefix-dependencies /scratch/<acct>/$USER/dftfe-deps --blas mkl --dftd4

  # Show the plan and every command without running anything
  python3 install.py --machine perlmutter --prefix $PSCRATCH/dftfe --dry-run

  # Be prompted for the main settings
  python3 install.py --interactive

  # All settings (and optionally the machine profile) in one file
  python3 install.py --config my_frontier.json

Layout
------
  <prefix-dependencies>/            installed dependencies (lib/, include/, dealii*/, p4est/, ...)
  <prefix-dependencies>/src|build|logs   sources, build trees and logs of the dependencies;
                                    deleted after a successful run unless --keep-*-dep is given
  <prefix>/dftfe                    DFT-FE git checkout; binaries in build/<release|debug>/{real,complex}
  <prefix>/invDFT                   invDFT git checkout; binary in build/<release|debug>/real
  <prefix>/env.sh                   module loads + library paths; source it before running

A rerun resumes: packages that are already installed with the same
configuration are skipped (see --force / --only to rebuild).
"""

import argparse
import copy
import datetime
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time

if sys.version_info < (3, 6):
    sys.exit("install.py requires Python 3.6 or newer")

SCRIPT = "install.py"
STATE_DIR = ".install_dftfe"
OWNER_MARKER = ".install_dftfe_owned"

# ---------------------------------------------------------------------------
# Versions and sources
# ---------------------------------------------------------------------------

DEFAULT_VERSIONS = {
    "openblas": "0.3.28",
    "blis": "3.0.1",
    "libflame": "5.2.0",
    "alglib": "4.06.0",
    "libxc": "6.2.2",
    "spglib": "02159eef6e7349535049a43fe2272bb634c77945",
    "p4est": "2.8.7",
    "kokkos": "4.3.00",
    "scalapack": "2.2.2",
    "elpa": "2025.01.001",
    "petsc": "3.21.1",
    "slepc": "3.21.1",
    "dealii": "9.6.2",
    "dftd3": "0.6.0",
    "dftd4": "3.7.0",
}

SOURCES = {
    "openblas": "https://github.com/OpenMathLib/OpenBLAS/releases/download/v{v}/OpenBLAS-{v}.tar.gz",
    "blis": "https://github.com/amd/blis/archive/refs/tags/{v}.tar.gz",
    "libflame": "https://github.com/flame/libflame/archive/refs/tags/{v}.tar.gz",
    "alglib": "https://www.alglib.net/translator/re/alglib-{v}.cpp.gpl.tgz",
    "libxc": "https://gitlab.com/libxc/libxc/-/archive/{v}/libxc-{v}.tar.gz",
    "spglib": "https://github.com/spglib/spglib.git",
    "p4est": "https://p4est.github.io/release/p4est-{v}.tar.gz",
    "kokkos": "https://github.com/kokkos/kokkos/archive/refs/tags/{v}.tar.gz",
    "scalapack": "https://github.com/Reference-ScaLAPACK/scalapack/archive/refs/tags/v{v}.tar.gz",
    "elpa": "https://elpa.mpcdf.mpg.de/software/tarball-archive/Releases/{v}/elpa-{v}.tar.gz",
    "petsc": "https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-{v}.tar.gz",
    "slepc": "https://slepc.upv.es/download/distrib/slepc-{v}.tar.gz",
    "dealii": "https://github.com/dealii/dealii/releases/download/v{v}/dealii-{v}.tar.gz",
    "dftd3": "https://github.com/dftd3/simple-dftd3/archive/refs/tags/v{v}.tar.gz",
    "dftd4": "https://github.com/dftd4/dftd4/archive/refs/tags/v{v}.tar.gz",
}

DFTFE_REPOS = {
    "github": "https://github.com/dftfeDevelopers/dftfe.git",
    "bitbucket": "https://bitbucket.org/dftfedevelopers/dftfe.git",
}
INVDFT_REPO = "https://github.com/dftfeDevelopers/invDFT.git"

BLAS_CHOICES = ["openblas", "blis+flame", "mkl", "libsci"]
GPU_VENDORS = ["nvidia", "amd", "intel"]
# Names accepted by --use-existing that are never built by this script.
EXTERNALS = ["boost", "libxml2", "dccl"]

# ---------------------------------------------------------------------------
# GPU defaults (per vendor); machine profiles override these.
#
# Strings here and in machine profiles may contain {march}, {arch}, {cc}, {cxx}, {fc}, {mpicc}, {mpicxx},
# {mpifc}; these are filled in by the installer. $VARS are left for the shell
# to expand at build time, after the modules have been loaded.
# ---------------------------------------------------------------------------

ELPA_CPU = {
    "cc": "{mpicc}", "cxx": "{mpicxx}", "fc": "{mpifc}",
    "cflags": "{march} -O2 -fPIC",
    "cxxflags": "-std=c++17 {march} -O2 -fPIC",
    "fcflags": "-ffree-line-length-none {march} -O2 -fPIC",
    "libs": "",
    "configure": "",
}

GPU_DEFAULTS = {
    "nvidia": {
        "lang": "cuda",
        "cxx_flags": "{march} -fPIC",
        "cxx_flags_release": "-O2",
        "device_flags": "-arch=sm_{arch}",
        "shared_linker_flags": "",
        "elpa": dict(ELPA_CPU, configure=(
            "--enable-nvidia-gpu --enable-gpu-streams=nvidia "
            "--with-NVIDIA-GPU-compute-capability=sm_{arch} "
            "--with-cuda-path=$CUDA_HOME --with-cuda-sdk-path=$CUDA_HOME "
            "--without-threading-support-check-during-build")),
    },
    "amd": {
        "lang": "hip",
        "cxx_flags": "{march} -fPIC -I$ROCM_PATH/include",
        "cxx_flags_release": "-O2",
        "device_flags": "{march} -O2 -munsafe-fp-atomics -I$ROCM_PATH/include",
        "shared_linker_flags": "-L$ROCM_PATH/lib -lamdhip64",
        "elpa": dict(
            ELPA_CPU, cxx="hipcc",
            cxxflags=("-std=c++17 {march} -O2 -fPIC --offload-arch={arch} "
                      "-I$ROCM_PATH/include -I$ROCM_PATH/include/rocsolver"),
            libs="-L$ROCM_PATH/lib -lamdhip64 -lrocblas -lrocsolver",
            configure="--enable-amd-gpu --enable-hipcub"),
    },
    "intel": {
        "lang": "sycl",
        "cxx_flags": "{march} -fPIC",
        "cxx_flags_release": "-O2",
        "device_flags": ('--intel -fsycl -O2 -fsycl-targets=spir64_gen '
                         '-Xsycl-target-backend "-device {arch}" -fp-model=precise'),
        "shared_linker_flags": "",
        "elpa": None,  # ELPA is built CPU-only for Intel GPUs
    },
}

# ---------------------------------------------------------------------------
# Machine profiles
#
# Profiles live in machines/<name>.json next to this script. A profile may also
# be given inline in a --config file as "machine": {...}, either complete or as
# {"base": "<name or path>", ...overrides...}. Keys missing from a profile take
# the values below; nested dicts are merged key by key, lists are replaced.
# ---------------------------------------------------------------------------

MACHINES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "machines")

PROFILE_SKELETON = {
    "description": "",
    "detect": {},             # {"lmod_system_name": [...], "env": {VAR: value}, "hostname_regex": [...]}
    "cray": False,            # Cray PE: unload cray-libsci unless --blas libsci, add CRAY_LD_LIBRARY_PATH
    "compilers": {"cc": "gcc", "cxx": "g++", "fc": "gfortran",
                  "mpicc": "mpicc", "mpicxx": "mpicxx", "mpifc": "mpif90"},
    "march": "-march=native",
    "modules": [],            # always loaded
    "cpu_modules": [],        # loaded for CPU-only builds
    "gpu_modules": {},        # vendor -> modules for GPU builds
    "dccl_modules": {},       # vendor -> modules for --dccl
    "blas_modules": {},       # blas kind -> modules
    "gpu_exports": {},        # environment variables exported for GPU builds
    "defaults": {},           # defaults for user settings (gpu, gpu_vendor, gpu_arch, blas, ...)
    "versions": {},           # package version overrides
    "dccl_prefix": {},        # vendor -> NCCL/RCCL prefix
    "elpa_cpu_configure": "", # extra ELPA configure options (e.g. SIMD kernels to disable)
    "blis_config": "auto",    # BLIS configuration name
    "gpu": {},                # vendor -> overrides of GPU_DEFAULTS[vendor]
}


def deep_merge(base, over):
    if isinstance(base, dict) and isinstance(over, dict):
        out = dict(base)
        for k, v in over.items():
            out[k] = deep_merge(base.get(k), v)
        return out
    return copy.deepcopy(over)


def available_machines():
    try:
        return sorted(f[:-5] for f in os.listdir(MACHINES_DIR) if f.endswith(".json"))
    except OSError:
        return []


def _profile_path(spec):
    if spec.endswith(".json") or os.sep in spec:
        return absp(spec)
    return os.path.join(MACHINES_DIR, spec + ".json")


def _read_profile(spec, seen=()):
    """Return the profile named/located by spec (str) or given inline (dict), unmerged with the skeleton."""
    if isinstance(spec, str):
        path = _profile_path(spec)
        if path in seen:
            raise InstallError("machine profile %s is its own base" % path)
        if not os.path.exists(path):
            raise InstallError("unknown machine '%s' (no %s); available: %s"
                               % (spec, path, ", ".join(available_machines()) or "none"))
        try:
            with open(path) as f:
                data = json.load(f)
        except ValueError as e:
            raise InstallError("cannot parse machine profile %s: %s" % (path, e))
        seen = seen + (path,)
    elif isinstance(spec, dict):
        data = spec
    else:
        raise InstallError("'machine' must be a name, a path to a .json file, or a dict")
    unknown = sorted(k for k in data if k not in PROFILE_SKELETON and k != "base" and not k.startswith("_"))
    if unknown:
        raise InstallError("unknown machine profile key(s): %s" % ", ".join(unknown))
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    if "base" in data:
        base = data.pop("base")
        # A base path inside a profile file is relative to that file.
        if (seen and isinstance(base, str) and (base.endswith(".json") or os.sep in base)
                and not os.path.isabs(os.path.expanduser(base))):
            base = os.path.join(os.path.dirname(seen[-1]), base)
        return deep_merge(_read_profile(base, seen), data)
    return data


def load_profile(spec):
    """Return (display name, full profile)."""
    prof = deep_merge(PROFILE_SKELETON, _read_profile(spec))
    if isinstance(spec, str):
        name = os.path.basename(spec)[:-5] if spec.endswith(".json") else spec
    elif "base" in spec:
        name = "%s (modified)" % os.path.basename(str(spec["base"])).replace(".json", "")
    else:
        name = "custom"
    if not prof["description"]:
        prof["description"] = "machine profile from the config file"
    return name, prof

# Settings that can come from the command line or a --config file.
DEFAULTS = {
    "machine": "auto",
    "prefix": None,
    "prefix_dependencies": None,
    "jobs": None,
    "gpu": False,
    "gpu_vendor": None,
    "gpu_arch": None,
    "elpa_gpu": None,
    "blas": "openblas",
    "petsc": None,
    "dftd3": False,
    "dftd4": False,
    "dccl": False,
    "gpu_aware_mpi": False,
    "int64": None,  # derived: off with --invdft, on otherwise
    "higher_quad_psp": False,
    "real_only": False,
    "build_type": "Release",
    "dftfe_repo": "github",
    "dftfe_branch": None,
    "dftfe_src": None,
    "invdft": False,
    "invdft_repo": INVDFT_REPO,
    "invdft_branch": "invGKS",
    "invdft_src": None,
    "git_pull": False,
    "modules": None,
    "add_module": [],
    "cc": None, "cxx": None, "fc": None,
    "mpicc": None, "mpicxx": None, "mpifc": None,
    "cpu_arch_flags": None,
    "device_flags": None,
    "cxx_flags": None,
    "versions": {},
    "extra_args": {},
    "use_existing": {},
    "only": None,
    "skip": [],
    "force": [],
    "dry_run": False,
    "emit_script": None,
    "fetch_only": False,
    "keep_src_dep": False,
    "keep_build_dep": False,
    "keep_logs_dep": False,
    "verbose": False,
    "yes": False,
}
# Settings that describe a single invocation; not saved in the resolved config.
TRANSIENT = {"only", "force", "dry_run", "emit_script", "fetch_only", "yes", "verbose"}


class InstallError(Exception):
    pass


class Cfg(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


class Step(object):
    """One shell snippet, run with bash in directory `cwd`."""
    __slots__ = ("cwd", "cmd")

    def __init__(self, cwd, cmd):
        self.cwd = cwd
        self.cmd = cmd


# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------

_SAFE = re.compile(r"[A-Za-z0-9_\-+=./:,@%^]+")


def q(s):
    """Quote for bash, keeping $VAR expansion (values are double-quoted)."""
    s = str(s)
    if s and _SAFE.fullmatch(s):
        return s
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`") + '"'


def cmdline(prog, args):
    parts = [prog] + [q(a) for a in args]
    if len(parts) <= 3:
        return " ".join(parts)
    return " \\\n    ".join(parts)


def onoff(flag):
    return "ON" if flag else "OFF"


def absp(path):
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def fetch_tarball(url, dest):
    tmp = dest + ".tmp"
    archive = tmp + "/archive"
    return "\n".join([
        "if [ ! -d %s ]; then" % q(dest),
        "  rm -rf %s && mkdir -p %s" % (q(tmp), q(tmp)),
        "  curl -fsSL --retry 3 -o %s %s" % (q(archive), q(url)),
        "  tar xf %s -C %s --strip-components=1" % (q(archive), q(tmp)),
        "  rm -f %s" % q(archive),
        "  mv %s %s" % (q(tmp), q(dest)),
        "fi",
    ])


def fresh_dir(path, copy_from=None):
    """Shell lines that recreate `path` (optionally as a copy of a source tree) and cd into it."""
    lines = ["rm -rf %s" % q(path)]
    if copy_from:
        lines.append("cp -r %s %s" % (q(copy_from), q(path)))
    else:
        lines.append("mkdir -p %s" % q(path))
    lines.append("cd %s" % q(path))
    return lines


# ---------------------------------------------------------------------------
# Packages
# ---------------------------------------------------------------------------

class Package(object):
    name = None
    base = None          # key into versions/sources when it differs from name
    dependency = True    # False for DFT-FE and invDFT

    def __init__(self, ctx):
        self.ctx = ctx

    @property
    def key(self):
        return self.base or self.name

    @property
    def version(self):
        return self.ctx.versions.get(self.key, "")

    def deps(self):
        return []

    def location(self):
        return self.ctx.D

    def srcdir(self):
        return os.path.join(self.ctx.src, "%s-%s" % (self.key, self.version))

    def builddir(self):
        return os.path.join(self.ctx.bld, self.name)

    def url(self):
        return SOURCES[self.key].format(v=self.version)

    def fetch_steps(self):
        return [Step(self.ctx.src, fetch_tarball(self.url(), self.srcdir()))]

    def build_steps(self, clean=True):
        raise NotImplementedError

    def logfile(self):
        return os.path.join(self.ctx.logs, self.name + ".log")

    def stampfile(self):
        root = self.ctx.D if self.dependency else self.ctx.P
        return os.path.join(root, STATE_DIR, "stamps", self.name + ".json")

    def runtime_libdirs(self):
        # Built packages install into lib/ (CMAKE_INSTALL_LIBDIR=lib); <deps>/lib64 is added
        # globally in env.sh. Existing installations may use either.
        loc = self.ctx.loc[self.name]
        if self.name in self.ctx.use_existing:
            return [os.path.join(loc, "lib"), os.path.join(loc, "lib64")]
        return [os.path.join(loc, "lib")]

    def pkgconfig_dirs(self):
        return [os.path.join(d, "pkgconfig") for d in self.runtime_libdirs()]

    # -- helpers --
    def cmake_steps(self, args, install=True):
        c = self.ctx
        args = list(args) + ["-DCMAKE_POLICY_VERSION_MINIMUM=3.5"] + c.extra(self)
        b = self.builddir()
        lines = fresh_dir(b) + [cmdline("cmake", args + [self.srcdir()]),
                                "cmake --build . --parallel %s" % c.j]
        if install:
            lines.append("cmake --install .")
        return [Step(c.bld, "\n".join(lines))]

    def cmake_prefix_args(self):
        return ["-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_INSTALL_PREFIX=" + self.location(),
                "-DCMAKE_INSTALL_LIBDIR=lib"]


class OpenBLAS(Package):
    name = "openblas"

    def build_steps(self, clean=True):
        c = self.ctx
        make_args = ["CC=" + c.cc["cc"], "FC=" + c.cc["fc"], "USE_OPENMP=0", "USE_THREAD=0",
                     "NO_AFFINITY=1"] + c.extra(self)
        lines = fresh_dir(self.builddir(), copy_from=self.srcdir()) + [
            cmdline("make -j%s" % c.j, make_args),
            cmdline("make", ["PREFIX=" + self.location()] + make_args + ["install"]),
        ]
        return [Step(c.bld, "\n".join(lines))]


class BLIS(Package):
    name = "blis"

    def build_steps(self, clean=True):
        c = self.ctx
        args = ["--prefix=" + self.location(), "--enable-shared", "--enable-threading=no",
                "CC=" + c.cc["cc"], "CXX=" + c.cc["cxx"], "FC=" + c.cc["fc"],
                "CFLAGS=-O2"] + c.extra(self) + [c.prof.get("blis_config", "auto")]
        lines = fresh_dir(self.builddir()) + [
            cmdline(q(os.path.join(self.srcdir(), "configure")), args),
            "make -j%s" % c.j,
            "make install",
        ]
        return [Step(c.bld, "\n".join(lines))]


class LibFlame(Package):
    name = "libflame"

    def deps(self):
        return ["blis"]

    def build_steps(self, clean=True):
        c = self.ctx
        args = ["--prefix=" + self.location(), "--enable-dynamic-build", "--enable-lapack2flame",
                "--enable-max-arg-list-hack", "CC=" + c.cc["cc"], "CXX=" + c.cc["cxx"],
                "FC=" + c.cc["fc"], "CFLAGS=-O2"] + c.extra(self)
        lines = fresh_dir(self.builddir(), copy_from=self.srcdir()) + [
            cmdline("./configure", args),
            "make -j%s" % c.j,
            "make install",
        ]
        return [Step(c.bld, "\n".join(lines))]


class Alglib(Package):
    name = "alglib"

    def location(self):
        return os.path.join(self.ctx.D, "alglib")

    def build_steps(self, clean=True):
        c = self.ctx
        loc = self.location()
        flags = ["-shared", "-fPIC", "-O2"] + c.extra(self)
        lines = [
            "rm -rf %s && mkdir -p %s" % (q(loc), q(loc)),
            "%s -o %s %s *.cpp" % (c.cc["cxx"], q(os.path.join(loc, "libAlglib.so")),
                                   " ".join(q(f) for f in flags)),
            "cp *.h %s/" % q(loc),
        ]
        return [Step(os.path.join(self.srcdir(), "src"), "\n".join(lines))]

    def runtime_libdirs(self):
        return [self.ctx.loc[self.name]]

    def pkgconfig_dirs(self):
        return []


class Libxc(Package):
    name = "libxc"

    def build_steps(self, clean=True):
        c = self.ctx
        return self.cmake_steps(self.cmake_prefix_args() + [
            "-DCMAKE_C_COMPILER=" + c.cc["cc"], "-DCMAKE_C_FLAGS=-O2 -fPIC",
            "-DCMAKE_CXX_COMPILER=" + c.cc["cxx"], "-DCMAKE_CXX_FLAGS=-O2 -fPIC",
            "-DBUILD_SHARED_LIBS=ON", "-DBUILD_TESTING=OFF"])


class Spglib(Package):
    name = "spglib"

    def srcdir(self):
        return os.path.join(self.ctx.src, "spglib-" + self.version[:10])

    def fetch_steps(self):
        d = self.srcdir()
        cmd = "\n".join([
            "if [ ! -d %s ]; then" % q(d + "/.git"),
            "  rm -rf %s" % q(d),
            "  git clone %s %s" % (q(SOURCES["spglib"]), q(d)),
            "  git -C %s checkout %s" % (q(d), q(self.version)),
            "fi",
        ])
        return [Step(self.ctx.src, cmd)]

    def build_steps(self, clean=True):
        c = self.ctx
        return self.cmake_steps(self.cmake_prefix_args() + [
            "-DCMAKE_C_COMPILER=" + c.cc["cc"], "-DCMAKE_CXX_COMPILER=" + c.cc["cxx"],
            "-DCMAKE_C_FLAGS=-O2 -fPIC", "-DSPGLIB_WITH_TESTS=OFF"])


class P4est(Package):
    """FAST and DEBUG builds, as deal.II expects (replaces the p4est-setup*.sh scripts)."""
    name = "p4est"

    def location(self):
        return os.path.join(self.ctx.D, "p4est")

    def build_steps(self, clean=True):
        c = self.ctx
        steps = []
        for variant, cflags, extra in (("FAST", "-fPIC -O2", []),
                                       ("DEBUG", "-fPIC -O0 -g", ["--enable-debug"])):
            b = os.path.join(self.builddir(), variant)
            args = ["CC=" + c.cc["mpicc"], "CXX=" + c.cc["mpicxx"], "FC=" + c.cc["mpifc"],
                    "F77=" + c.cc["mpifc"], "--enable-mpi", "--enable-shared",
                    "--disable-vtk-binary", "--without-blas"] + extra + [
                    "--prefix=" + os.path.join(self.location(), variant),
                    "CFLAGS=" + cflags, "CPPFLAGS=-DSC_LOG_PRIORITY=SC_LP_ESSENTIAL"] + c.extra(self)
            lines = fresh_dir(b) + [
                cmdline(q(os.path.join(self.srcdir(), "configure")), args),
                "make -j%s" % c.j,
                'if ! grep -q "P4EST_HAVE_ZLIB *1" $(find . -name p4est_config.h); then',
                '  echo "ERROR: p4est was built without zlib, which deal.II requires." >&2',
                '  echo "Install/load zlib or add -L/-I paths via --extra-args p4est=..." >&2',
                "  exit 1",
                "fi",
                "make install",
            ]
            steps.append(Step(c.bld, "\n".join(lines)))
        return steps

    def runtime_libdirs(self):
        return [os.path.join(self.ctx.loc[self.name], "FAST", "lib")]

    def pkgconfig_dirs(self):
        return []


class Kokkos(Package):
    name = "kokkos"

    def build_steps(self, clean=True):
        c = self.ctx
        return self.cmake_steps(self.cmake_prefix_args() + [
            "-DCMAKE_C_COMPILER=" + c.cc["mpicc"], "-DCMAKE_C_FLAGS=-O2 -fPIC",
            "-DCMAKE_CXX_COMPILER=" + c.cc["mpicxx"], "-DCMAKE_CXX_FLAGS=-O2 -fPIC",
            "-DCMAKE_CXX_STANDARD=17"])


class ScaLAPACK(Package):
    name = "scalapack"

    def deps(self):
        return list(self.ctx.blas_pkgs)

    def build_steps(self, clean=True):
        c = self.ctx
        b = c.blas
        return self.cmake_steps(self.cmake_prefix_args() + [
            "-DBUILD_SHARED_LIBS=ON", "-DBUILD_STATIC_LIBS=OFF", "-DBUILD_TESTING=OFF",
            "-DCMAKE_C_COMPILER=" + c.cc["mpicc"], "-DCMAKE_Fortran_COMPILER=" + c.cc["mpifc"],
            "-DCMAKE_C_FLAGS=-fPIC %s -Wno-error=implicit-function-declaration" % c.march,
            "-DCMAKE_Fortran_FLAGS=-fPIC %s -fallow-argument-mismatch" % c.march,
            "-DUSE_OPTIMIZED_LAPACK_BLAS=ON",
            "-DBLAS_LIBRARIES=" + ";".join(b["blas_f"]),
            "-DLAPACK_LIBRARIES=" + ";".join(b["lapack_f"])])


class ELPA(Package):
    name = "elpa"

    def deps(self):
        d = list(self.ctx.blas_pkgs)
        if self.ctx.cfg.blas != "libsci":
            d.append("scalapack")
        return d

    def build_steps(self, clean=True):
        c = self.ctx
        use_gpu = c.gpu is not None and c.cfg.elpa_gpu and c.gpu.get("elpa")
        e = c.gpu["elpa"] if use_gpu else c.T(ELPA_CPU)
        libs = " ".join(x for x in (e["libs"], c.scalapack_link, c.blas["f_link"]) if x)
        args = ["CC=" + e["cc"], "CXX=" + e["cxx"], "FC=" + e["fc"],
                "CFLAGS=" + e["cflags"], "CXXFLAGS=" + e["cxxflags"], "FCFLAGS=" + e["fcflags"],
                "LIBS=" + libs, "--prefix=" + self.location(), "--enable-shared",
                "--enable-c-tests=no", "--enable-cpp-tests=no", "--enable-option-checking=fatal"]
        args += shlex.split(c.T(c.prof.get("elpa_cpu_configure", "")))
        args += shlex.split(e["configure"])
        args += c.extra(self)
        lines = fresh_dir(self.builddir()) + [
            cmdline(q(os.path.join(self.srcdir(), "configure")), args),
            "make -j%s" % c.j,
            "make install",
        ]
        return [Step(c.bld, "\n".join(lines))]


class PETSc(Package):
    base = "petsc"

    def __init__(self, ctx, scalar):
        Package.__init__(self, ctx)
        self.scalar = scalar
        self.name = "petsc-" + scalar

    def deps(self):
        return list(self.ctx.blas_pkgs)

    def location(self):
        return os.path.join(self.ctx.D, self.name)

    def build_steps(self, clean=True):
        c = self.ctx
        src = self.srcdir()
        arch = "arch-dftfe-" + self.scalar
        args = ["PETSC_ARCH=" + arch, "--prefix=" + self.location(), "--with-debugging=no",
                "--with-64-bit-indices=true", "--with-shared-libraries=1", "--with-x=0",
                "--with-cc=" + c.cc["mpicc"], "--with-cxx=" + c.cc["mpicxx"],
                "--with-fc=" + c.cc["mpifc"], "--with-scalar-type=" + self.scalar]
        if self.scalar == "complex":
            args.append("--with-fortran-kernels=true")
        if c.blas["petsc"]:
            args.append(c.blas["petsc"])
        args += ["COPTFLAGS=-O2", "CXXOPTFLAGS=-O2", "FOPTFLAGS=-O2"] + c.extra(self)
        make = "make PETSC_DIR=%s PETSC_ARCH=%s" % (q(src), arch)
        lines = ["rm -rf %s" % arch, "unset PETSC_DIR PETSC_ARCH",
                 cmdline("./configure", args), make + " all", make + " install"]
        return [Step(src, "\n".join(lines))]


class SLEPc(Package):
    base = "slepc"

    def __init__(self, ctx, scalar):
        Package.__init__(self, ctx)
        self.scalar = scalar
        self.name = "slepc-" + scalar

    def deps(self):
        return ["petsc-" + self.scalar]

    def location(self):
        return os.path.join(self.ctx.D, self.name)

    def build_steps(self, clean=True):
        c = self.ctx
        b = self.builddir()
        petsc = c.loc["petsc-" + self.scalar]
        make = "make SLEPC_DIR=%s PETSC_DIR=%s" % (q(b), q(petsc))
        lines = fresh_dir(b, copy_from=self.srcdir()) + [
            "export PETSC_DIR=%s SLEPC_DIR=%s" % (q(petsc), q(b)),
            "unset PETSC_ARCH",
            cmdline("./configure", ["--prefix=" + self.location()] + c.extra(self)),
            make,
            make + " install",
        ]
        return [Step(c.bld, "\n".join(lines))]


class DealII(Package):
    base = "dealii"

    def __init__(self, ctx, scalar=None):
        Package.__init__(self, ctx)
        self.scalar = scalar
        self.name = "dealii-" + scalar if scalar else "dealii"

    def deps(self):
        d = ["p4est", "kokkos"] + list(self.ctx.blas_pkgs)
        if self.scalar:
            d += ["petsc-" + self.scalar, "slepc-" + self.scalar]
        return d

    def location(self):
        return os.path.join(self.ctx.D, self.name)

    def build_steps(self, clean=True):
        c = self.ctx
        cc = c.cc
        args = [
            "-DCMAKE_BUILD_TYPE=Release", "-DDEAL_II_CXX_FLAGS_RELEASE=-O2",
            "-DCMAKE_INSTALL_PREFIX=" + self.location(),
            "-DCMAKE_CXX_STANDARD=17",
            "-DCMAKE_C_COMPILER=" + cc["cc"], "-DCMAKE_CXX_COMPILER=" + cc["cxx"],
            "-DCMAKE_Fortran_COMPILER=" + cc["fc"],
            "-DMPI_C_COMPILER=" + cc["mpicc"], "-DMPI_CXX_COMPILER=" + cc["mpicxx"],
            "-DMPI_Fortran_COMPILER=" + cc["mpifc"],
            "-DCMAKE_CXX_FLAGS=%s -std=c++17" % c.march, "-DCMAKE_C_FLAGS=" + c.march,
            "-DDEAL_II_WITH_MPI=ON", "-DDEAL_II_WITH_64BIT_INDICES=ON",
            "-DDEAL_II_WITH_COMPLEX_VALUES=ON",
            "-DDEAL_II_WITH_P4EST=ON", "-DP4EST_DIR=" + c.loc["p4est"],
            "-DKOKKOS_DIR=" + c.loc["kokkos"],
            "-DDEAL_II_WITH_LAPACK=ON", "-DLAPACK_FOUND=true",
            "-DLAPACK_LIBRARIES=" + ";".join(c.blas["lapack_c"]),
            "-DDEAL_II_WITH_TBB=OFF", "-DDEAL_II_WITH_TASKFLOW=OFF",
            "-DDEAL_II_COMPONENT_EXAMPLES=OFF",
            "-DDEAL_II_FORCE_BUNDLED_BOOST=OFF", "-DDEAL_II_ALLOW_PLATFORM_INTROSPECTION=OFF",
        ]
        if "boost" in c.loc:
            args.append("-DBOOST_DIR=" + c.loc["boost"])
        if self.scalar:
            args += ["-DDEAL_II_WITH_PETSC=ON", "-DPETSC_DIR=" + c.loc["petsc-" + self.scalar],
                     "-DDEAL_II_WITH_SLEPC=ON", "-DSLEPC_DIR=" + c.loc["slepc-" + self.scalar]]
        return self.cmake_steps(args)


class Dftd(Package):
    """simple-dftd3 (name 'dftd3') and dftd4; each gets its own prefix since both bundle mctc-lib."""

    def __init__(self, ctx, name):
        Package.__init__(self, ctx)
        self.name = name

    def deps(self):
        return list(self.ctx.blas_pkgs)

    def location(self):
        return os.path.join(self.ctx.D, self.name)

    def build_steps(self, clean=True):
        c = self.ctx
        return self.cmake_steps(self.cmake_prefix_args() + [
            "-DCMAKE_Fortran_COMPILER=" + c.cc["fc"], "-DCMAKE_C_COMPILER=" + c.cc["cc"],
            "-DBLAS_LIBRARIES=" + ";".join(c.blas["blas_f"]),
            "-DLAPACK_LIBRARIES=" + ";".join(c.blas["lapack_f"]),
            "-DBUILD_SHARED_LIBS=ON", "-DWITH_OpenMP=OFF"])


def gpu_cmake_args(ctx):
    """CMake arguments shared by DFT-FE and invDFT."""
    c = ctx.cfg
    g = ctx.gpu
    if g:
        cxx_flags, release = g["cxx_flags"], g["cxx_flags_release"]
    else:
        cxx_flags, release = ctx.march + " -fPIC", "-O2"
    if c.cxx_flags:
        cxx_flags = c.cxx_flags
    prefix_path = [ctx.loc["elpa"]]
    for name in ("dftd3", "dftd4"):
        if name in ctx.loc:
            prefix_path.append(ctx.loc[name])
    dccl = bool(g and c.dccl)
    if dccl:
        prefix_path.append(ctx.dccl_prefix)
    args = ["-DCMAKE_CXX_STANDARD=17",
            "-DCMAKE_CXX_COMPILER=" + ctx.cc["mpicxx"],
            "-DCMAKE_CXX_FLAGS=" + cxx_flags,
            "-DCMAKE_CXX_FLAGS_RELEASE=" + release,
            "-DCMAKE_BUILD_TYPE=" + c.build_type,
            "-DALGLIB_DIR=" + ctx.loc["alglib"],
            "-DLIBXC_DIR=" + ctx.loc["libxc"],
            "-DSPGLIB_DIR=" + ctx.loc["spglib"],
            "-DXML_LIB_DIR=" + ctx.xml_lib,
            "-DXML_INCLUDE_DIR=" + ctx.xml_inc,
            "-DCMAKE_PREFIX_PATH=" + ";".join(prefix_path),
            "-DWITH_DCCL=" + onoff(dccl)]
    if not g:
        return args + ["-DWITH_GPU=OFF"]
    lang = g["lang"].upper()
    args += ["-DWITH_GPU=ON", "-DGPU_LANG=" + g["lang"], "-DGPU_VENDOR=" + c.gpu_vendor,
             "-DWITH_GPU_AWARE_MPI=" + onoff(c.gpu_aware_mpi),
             "-DCMAKE_%s_FLAGS=%s" % (lang, c.device_flags or g["device_flags"])]
    if g["lang"] in ("cuda", "hip"):
        args.append("-DCMAKE_%s_ARCHITECTURES=%s" % (lang, c.gpu_arch))
    if g["shared_linker_flags"]:
        args.append("-DCMAKE_SHARED_LINKER_FLAGS=" + g["shared_linker_flags"])
    if dccl:
        p = ctx.dccl_prefix
        if g["lang"] == "cuda":
            args += ["-DNCCL_INCLUDE_DIR=%s/include" % p, "-DNCCL_LIB_DIR=%s/lib" % p]
        else:
            args += ["-DRCCL_INCLUDE_DIR=%s/include/rccl" % p, "-DRCCL_LIB_DIR=%s/lib" % p]
    return args


class GitProject(Package):
    """Common logic for DFT-FE and invDFT: git checkout under --prefix, in-tree build directory."""
    dependency = False
    repo = None
    branch = None

    @property
    def version(self):
        return self.branch

    def location(self):
        return self.srcdir()

    def builddir(self):
        return os.path.join(self.ctx.loc[self.name], "build", self.ctx.cfg.build_type.lower())

    def logfile(self):
        return os.path.join(self.ctx.loc[self.name], "build", "install_%s.log" % self.name)

    def runtime_libdirs(self):
        return []

    def pkgconfig_dirs(self):
        return []

    def fetch_steps(self):
        src = self.srcdir()
        lines = ["if [ ! -e %s ]; then" % q(src),
                 "  git clone -b %s %s %s" % (q(self.branch), q(self.repo), q(src)),
                 "fi"]
        if self.ctx.cfg.git_pull:
            lines += ["if [ -d %s ]; then" % q(src + "/.git"),
                      "  cd %s" % q(src),
                      "  git fetch origin",
                      "  git checkout %s" % q(self.branch),
                      "  git pull --ff-only origin %s" % q(self.branch),
                      "fi"]
        return [Step(os.path.dirname(src), "\n".join(lines))]

    def branch_check(self):
        src = self.ctx.loc[self.name]
        return Step(src, "\n".join([
            'cur=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)',
            'if [ "$cur" != %s ]; then' % q(self.branch),
            '  echo "WARNING: %s is on branch $cur, expected %s" >&2' % (src, self.branch),
            "fi"]))

    def cmake_build(self, kind, args, clean):
        c = self.ctx
        src = c.loc[self.name]
        d = os.path.join(self.builddir(), kind)
        lines = ["rm -rf %s" % q(d)] if clean else []
        lines += ["mkdir -p %s" % q(d), "cd %s" % q(d),
                  cmdline("cmake", args + c.extra(self) + [src]),
                  "cmake --build . --parallel %s" % c.j]
        return Step(src, "\n".join(lines))


class DFTFE(GitProject):
    name = "dftfe"

    def __init__(self, ctx):
        GitProject.__init__(self, ctx)
        repo = ctx.cfg.dftfe_repo
        self.repo = DFTFE_REPOS.get(repo, repo)
        self.branch = ctx.cfg.dftfe_branch

    def srcdir(self):
        return absp(self.ctx.cfg.dftfe_src) if self.ctx.cfg.dftfe_src else os.path.join(self.ctx.P, "dftfe")

    def deps(self):
        c = self.ctx
        d = list(c.dealii_names) + ["alglib", "libxc", "spglib", "elpa"]
        d += [n for n in ("dftd3", "dftd4") if c.cfg[n]]
        return d

    def build_steps(self, clean=True):
        c = self.ctx
        steps = [self.branch_check()]
        kinds = ["real"] if c.cfg.real_only else ["real", "complex"]
        for kind in kinds:
            args = gpu_cmake_args(c) + [
                "-DDEAL_II_DIR=" + c.dealii_dir(kind),
                "-DWITH_MDI=OFF", "-DMDI_PATH=", "-DWITH_TORCH=OFF",
                "-DWITH_CUSTOMIZED_DEALII=OFF", "-DWITH_TESTING=OFF", "-DMINIMAL_COMPILE=OFF",
                "-DHIGHERQUAD_PSP=" + onoff(c.cfg.higher_quad_psp),
                "-DBUILD_SHARED_LIBS=ON", "-DUSE_64BIT_INT=" + onoff(c.cfg.int64),
                "-DWITH_COMPLEX=" + onoff(kind == "complex")]
            steps.append(self.cmake_build(kind, args, clean))
        return steps


class InvDFT(GitProject):
    name = "invdft"

    def __init__(self, ctx):
        GitProject.__init__(self, ctx)
        self.repo = ctx.cfg.invdft_repo
        self.branch = ctx.cfg.invdft_branch

    def srcdir(self):
        return absp(self.ctx.cfg.invdft_src) if self.ctx.cfg.invdft_src else os.path.join(self.ctx.P, "invDFT")

    def deps(self):
        return ["dftfe", self.ctx.dealii_names[0], "alglib", "libxc", "spglib", "elpa"]

    def build_steps(self, clean=True):
        c = self.ctx
        dftfe_src = c.loc["dftfe"]
        dftfe_real = os.path.join(dftfe_src, "build", c.cfg.build_type.lower(), "real")
        # Newer DFT-FE generates include/dftfe/config.h inside its build tree.
        includes = [os.path.join(dftfe_src, "include"), os.path.join(dftfe_real, "include")]
        args = gpu_cmake_args(c) + [
            "-DDEAL_II_DIR=" + c.dealii_dir("real"),
            "-DDFTFE_INSTALL_PATH=" + dftfe_real,
            "-DDFTFE_INCLUDE_PATH=" + ";".join(includes),
            "-DUSE_64BIT_INT=" + onoff(c.cfg.int64),
            "-DWITH_COMPLEX=OFF"]
        return [self.branch_check(), self.cmake_build("real", args, clean)]


# ---------------------------------------------------------------------------
# Machine detection, configuration
# ---------------------------------------------------------------------------

def detect_machine(env=None):
    """First profile in machines/ whose "detect" rules match this host, else "generic"."""
    env = os.environ if env is None else env
    sysname = env.get("LMOD_SYSTEM_NAME", "").lower()
    host = (socket.gethostname() + " " + socket.getfqdn()).lower()
    for name in available_machines():
        try:
            rules = _read_profile(name).get("detect", {})
        except InstallError:
            continue
        if (sysname and sysname in [s.lower() for s in rules.get("lmod_system_name", [])]
                or any(env.get(k, "").lower() == str(v).lower() for k, v in rules.get("env", {}).items())
                or any(re.search(rx, host) for rx in rules.get("hostname_regex", []))):
            return name
    return "generic"


def default_jobs():
    return min(os.cpu_count() or 4, 16)


def parse_kv(items, what):
    out = {}
    for item in items or []:
        if isinstance(item, dict):
            out.update(item)
            continue
        if "=" not in item:
            raise InstallError("%s expects NAME=VALUE, got '%s'" % (what, item))
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def parse_list(items):
    out = []
    for item in items or []:
        out += [x.strip() for x in item.split(",") if x.strip()]
    return out


def normalize_arch(vendor, arch):
    arch = str(arch).strip()
    if vendor == "nvidia":
        for pre in ("sm_", "compute_"):
            if arch.startswith(pre):
                arch = arch[len(pre):]
        if not arch.isdigit():
            raise InstallError("NVIDIA --gpu-arch must look like 80 or sm_80, got '%s'" % arch)
    elif vendor == "amd" and not arch.startswith("gfx"):
        raise InstallError("AMD --gpu-arch must look like gfx90a, got '%s'" % arch)
    return arch


def load_config_file(path):
    path = absp(path)
    try:
        with open(path) as f:
            data = json.load(f)
    except ValueError as e:
        raise InstallError("cannot parse %s: %s" % (path, e))
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    unknown = sorted(set(data) - set(DEFAULTS))
    if unknown:
        raise InstallError("unknown keys in %s: %s" % (path, ", ".join(unknown)))

    # Profile paths inside a config file are relative to the config file.
    def rel(spec):
        if (isinstance(spec, str) and (spec.endswith(".json") or os.sep in spec)
                and not os.path.isabs(os.path.expanduser(spec))):
            return os.path.join(os.path.dirname(path), spec)
        return spec
    m = data.get("machine")
    if isinstance(m, dict) and "base" in m:
        data["machine"] = dict(m, base=rel(m["base"]))
    elif m is not None:
        data["machine"] = rel(m)
    return data


def resolve_config(cli, file_cfg):
    """Return (cfg, notes, machine name, machine profile)."""
    machine = cli.get("machine") or file_cfg.get("machine") or "auto"
    if machine == "auto":
        machine = detect_machine()
    name, prof = load_profile(machine)

    cfg = Cfg(copy.deepcopy(DEFAULTS))
    cfg.update(prof["defaults"])
    for src in (file_cfg, cli):
        for k, v in src.items():
            if k in ("versions", "extra_args", "use_existing"):
                cfg[k] = dict(cfg[k], **parse_kv(v if isinstance(v, list) else [v], k))
            elif k in ("skip", "force", "only", "add_module"):
                vals = v if isinstance(v, list) else [v]
                cfg[k] = parse_list(vals) if k != "add_module" else list(vals)
            else:
                cfg[k] = v
    cfg["machine"] = machine  # name, path or inline dict, as given (saved for reruns)
    if isinstance(cfg.modules, str):
        cfg["modules"] = cfg.modules.split()

    if not cfg.prefix:
        raise InstallError("--prefix is required (or use --interactive)")
    if cfg.blas not in BLAS_CHOICES:
        raise InstallError("--blas must be one of %s" % ", ".join(BLAS_CHOICES))
    if cfg.blas == "libsci" and not prof["cray"]:
        raise InstallError("--blas libsci needs a Cray machine profile (\"cray\": true)")

    if cfg.gpu:
        if not cfg.gpu_vendor:
            raise InstallError("--gpu needs --gpu-vendor (nvidia, amd or intel)")
        if cfg.gpu_vendor not in GPU_VENDORS:
            raise InstallError("--gpu-vendor must be one of %s" % ", ".join(GPU_VENDORS))
        if not cfg.gpu_arch:
            if cfg.gpu_vendor == "intel":
                cfg["gpu_arch"] = "pvc"
            else:
                raise InstallError("--gpu needs --gpu-arch (e.g. 80 for A100, gfx90a for MI250X)")
        cfg["gpu_arch"] = normalize_arch(cfg.gpu_vendor, cfg.gpu_arch)
        if cfg.gpu_vendor == "intel":
            if cfg.blas != "mkl":
                raise InstallError("Intel GPU builds of DFT-FE require --blas mkl")
            if cfg.invdft:
                raise InstallError("invDFT supports only CUDA and HIP GPU builds, not Intel GPUs")

    notes = []
    if cfg.elpa_gpu is None:
        cfg["elpa_gpu"] = bool(cfg.gpu and cfg.gpu_vendor != "intel")
    elif cfg.elpa_gpu and not cfg.gpu:
        notes.append("--elpa-gpu ignored because GPU support is off")
        cfg["elpa_gpu"] = False
    elif cfg.elpa_gpu and cfg.gpu_vendor == "intel":
        notes.append("ELPA is built CPU-only for Intel GPUs")
        cfg["elpa_gpu"] = False
    if cfg.dccl and not cfg.gpu:
        notes.append("--dccl ignored because GPU support is off")
        cfg["dccl"] = False
    if cfg.int64 is None:
        cfg["int64"] = not cfg.invdft
    elif cfg.int64 and cfg.invdft:
        raise InstallError("invDFT does not support 64-bit integers yet: DFT-FE must be built with "
                           "32-bit integers for invDFT. Remove --int64 (or \"int64\": true in the "
                           "config file) or pass --no-int64.")
    if cfg.petsc is None:
        cfg["petsc"] = bool(cfg.invdft)
    elif not cfg.petsc and cfg.invdft:
        notes.append("invDFT requested with --no-petsc: all-electron Gram-Schmidt "
                     "orthogonalization will not be available")
    if cfg.dftfe_branch is None:
        cfg["dftfe_branch"] = "publicGithubDevelop"
    if cfg.build_type not in ("Release", "Debug"):
        raise InstallError("--build-type must be Release or Debug")
    unknown_versions = sorted(set(cfg.versions) - set(DEFAULT_VERSIONS))
    if unknown_versions:
        raise InstallError("--pkg-version: unknown package(s) %s; known: %s"
                           % (", ".join(unknown_versions), ", ".join(sorted(DEFAULT_VERSIONS))))
    return cfg, notes, name, prof


# ---------------------------------------------------------------------------
# Context: everything derived from the configuration
# ---------------------------------------------------------------------------

MODULE_INIT = r'''# make the "module" command available in non-interactive shells
if ! type module >/dev/null 2>&1; then
  for _f in /etc/profile.d/lmod.sh /etc/profile.d/z00_lmod.sh /etc/profile.d/modules.sh \
            /usr/share/lmod/lmod/init/bash "${LMOD_PKG:-/nonexistent}/init/bash"; do
    if [ -f "$_f" ]; then . "$_f"; break; fi
  done
  unset _f
fi'''

ENV_MARKER = b"\n__INSTALL_DFTFE_ENV__\n"


class Context(object):
    def __init__(self, cfg, machine_name, profile):
        self.cfg = cfg
        self.machine_name = machine_name
        self.prof = profile
        self.P = absp(cfg.prefix)
        self.D = absp(cfg.prefix_dependencies or cfg.prefix)
        self.src = os.path.join(self.D, "src")
        self.bld = os.path.join(self.D, "build")
        self.logs = os.path.join(self.D, "logs")
        self.jobs = int(cfg.jobs or default_jobs())
        self.sig_mode = False
        self.cc = dict(self.prof["compilers"])
        for k in self.cc:
            if cfg.get(k):
                self.cc[k] = cfg[k]
        self.march = cfg.cpu_arch_flags if cfg.cpu_arch_flags is not None else self.prof["march"]
        self.versions = dict(DEFAULT_VERSIONS)
        self.versions.update(self.prof.get("versions", {}))
        self.versions.update(cfg.versions)
        self.gpu = self._gpu_settings() if cfg.gpu else None
        self.use_existing = {}
        for k, v in cfg.use_existing.items():
            self.use_existing[k] = v if v.startswith("$") else absp(v)
        self.env = None
        self.module_failures = []
        self.loc = {}
        self.blas = None
        self.blas_pkgs = {"openblas": ["openblas"], "blis+flame": ["blis", "libflame"]}.get(cfg.blas, [])
        self.dealii_names = ["dealii-real", "dealii-complex"] if cfg.petsc else ["dealii"]
        self.scalapack_link = ""
        self.dccl_prefix = None
        self.xml_inc = self.xml_lib = None

    @property
    def j(self):
        # Parallelism must not change a package's signature.
        return "N" if self.sig_mode else str(self.jobs)

    def T(self, obj):
        """Fill {march}, {arch}, compiler tokens in a string (or dict/list of strings)."""
        if isinstance(obj, dict):
            return {k: self.T(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.T(v) for v in obj]
        if not isinstance(obj, str):
            return obj
        tokens = dict(self.cc, march=self.march, arch=self.cfg.gpu_arch or "")
        for k, v in tokens.items():
            obj = obj.replace("{%s}" % k, v)
        return obj

    def _gpu_settings(self):
        vendor = self.cfg.gpu_vendor
        g = deep_merge(GPU_DEFAULTS[vendor], self.prof["gpu"].get(vendor, {}))
        return self.T(g)

    def extra(self, pkg):
        out = []
        for key in dict.fromkeys([pkg.key, pkg.name]):
            out += shlex.split(self.cfg.extra_args.get(key, ""))
        return out

    def dealii_dir(self, kind):
        return self.loc["dealii-" + kind] if self.cfg.petsc else self.loc["dealii"]

    def modules(self):
        c = self.cfg
        p = self.prof
        mods = list(c.modules) if c.modules is not None else list(p["modules"])
        if c.modules is None:
            if c.gpu:
                mods += p["gpu_modules"].get(c.gpu_vendor, [])
                if c.dccl:
                    mods += p["dccl_modules"].get(c.gpu_vendor, [])
            else:
                mods += p["cpu_modules"]
            mods += p["blas_modules"].get(c.blas, [])
        mods += c.add_module
        return self.T(mods)

    # -- environment --
    def env_sh(self, pkgs):
        c = self.cfg
        lines = ["# Environment for DFT-FE%s, generated by %s on %s" % (
                     " and invDFT" if c.invdft else "", SCRIPT,
                     datetime.datetime.now().strftime("%Y-%m-%d %H:%M")),
                 "# Source this file before building or running:  source %s" %
                 os.path.join(self.P, "env.sh"),
                 MODULE_INIT]
        for m in self.modules():
            lines.append('module load %s || echo "install_dftfe: failed to load module %s" >&2' % (m, m))
        if self.prof["cray"] and c.blas != "libsci":
            lines.append("module unload cray-libsci 2>/dev/null || true")
        if c.gpu:
            for k, v in sorted(self.prof.get("gpu_exports", {}).items()):
                lines.append("export %s=%s" % (k, q(v)))
            if c.gpu_vendor == "nvidia":
                lines.append('export CUDA_HOME="${CUDA_HOME:-${CUDA_PATH:-$(dirname "$(dirname '
                             '"$(command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)")")}}"')
        lines.append("export DFTFE_DEPS_PREFIX=%s" % q(self.D))
        libdirs, pcdirs = [os.path.join(self.D, "lib"), os.path.join(self.D, "lib64")], []
        for p in pkgs:
            libdirs += p.runtime_libdirs()
            pcdirs += p.pkgconfig_dirs()
        for name in ("boost", "libxml2"):
            if name in self.loc:
                libdirs += [os.path.join(self.loc[name], "lib"), os.path.join(self.loc[name], "lib64")]
        libdirs = list(dict.fromkeys(libdirs))
        pcdirs = list(dict.fromkeys(pcdirs))
        lines.append('export LD_LIBRARY_PATH="%s${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"' % ":".join(libdirs))
        if self.prof["cray"]:
            lines.append('export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:+$CRAY_LD_LIBRARY_PATH:}$LD_LIBRARY_PATH"')
        if pcdirs:
            lines.append('export PKG_CONFIG_PATH="%s${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"' % ":".join(pcdirs))
        return "\n".join(lines) + "\n"

    def capture_env(self, env_text):
        script = env_text + "\nprintf '\\n__INSTALL_DFTFE_ENV__\\n'\nenv -0\n"
        p = subprocess.run(["bash", "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out = p.stdout
        idx = out.rfind(ENV_MARKER)
        if idx < 0:
            raise InstallError("could not set up the build environment:\n"
                               + p.stderr.decode(errors="replace"))
        env = {}
        for item in out[idx + len(ENV_MARKER):].decode(errors="replace").split("\0"):
            if "=" in item:
                k, v = item.split("=", 1)
                env[k] = v
        self.env = env
        self.module_failures = [l.split("module", 2)[-1].strip()
                                for l in p.stderr.decode(errors="replace").splitlines()
                                if "install_dftfe: failed to load module" in l]

    # -- derived settings that need the build environment --
    def set_locations(self, pkgs):
        for p in pkgs:
            self.loc[p.name] = self.use_existing.get(p.name, p.location())
        for name in EXTERNALS:
            if name in self.use_existing:
                self.loc[name] = self.use_existing[name]

    def finalize(self, pkgs):
        warnings = []
        names = {p.name for p in pkgs}
        self.blas = self._blas_info(warnings)
        if self.cfg.blas == "libsci":
            self.scalapack_link = self.blas["scalapack_link"]
        else:
            self.scalapack_link = "-L%s/lib -lscalapack" % self.loc["scalapack"]
        if self.cfg.dccl:
            self.dccl_prefix = (self.loc.get("dccl")
                                or self.prof.get("dccl_prefix", {}).get(self.cfg.gpu_vendor))
            if not self.dccl_prefix:
                raise InstallError("--dccl: no default NCCL/RCCL location for this machine; "
                                   "pass --use-existing dccl=PATH")
        if names & {"dftfe", "invdft"}:
            self.xml_inc, self.xml_lib, ok = self._find_libxml2()
            if not ok:
                warnings.append("libxml2 development files not found; using %s and %s. "
                                "Pass --use-existing libxml2=PREFIX if that is wrong."
                                % (self.xml_inc, self.xml_lib))
        if self.cfg.gpu and self.cfg.gpu_vendor == "amd" and not self.env.get("ROCM_PATH"):
            warnings.append("ROCM_PATH is not set after loading modules")
        if self.cfg.gpu and self.cfg.gpu_vendor == "nvidia" and not self._which("nvcc"):
            warnings.append("nvcc not found after loading modules")
        return warnings

    def _which(self, prog):
        return shutil.which(prog, path=(self.env or os.environ).get("PATH", ""))

    def _blas_info(self, warnings):
        kind = self.cfg.blas
        info = {"petsc": "", "scalapack_link": ""}
        if kind == "openblas":
            lib = os.path.join(self.loc["openblas"], "lib", "libopenblas.so")
            link = "-L%s/lib -lopenblas" % self.loc["openblas"]
            info.update(c_link=link, f_link=link, blas_c=[lib], lapack_c=[lib], blas_f=[lib],
                        lapack_f=[lib], petsc="--with-blaslapack-lib=[%s]" % lib)
        elif kind == "blis+flame":
            blis = os.path.join(self.loc["blis"], "lib", "libblis.so")
            flame = os.path.join(self.loc["libflame"], "lib", "libflame.so")
            link = "-L%s/lib -lflame -L%s/lib -lblis" % (self.loc["libflame"], self.loc["blis"])
            info.update(c_link=link, f_link=link, blas_c=[blis], lapack_c=[flame, blis],
                        blas_f=[blis], lapack_f=[flame, blis],
                        petsc="--with-blaslapack-lib=[%s,%s]" % (flame, blis))
        elif kind == "mkl":
            root = self.env.get("MKLROOT")
            if root:
                libdir = os.path.join(root, "lib", "intel64")
                if not os.path.isdir(libdir):
                    libdir = os.path.join(root, "lib")
            else:
                libdir = "$MKLROOT/lib/intel64"
                warnings.append("MKLROOT is not set after loading modules (add the MKL module "
                                "with --add-module)")
            tail = "-lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"
            c_link = "-L%s -Wl,--no-as-needed -lmkl_intel_lp64 %s" % (libdir, tail)
            f_link = "-L%s -Wl,--no-as-needed -lmkl_gf_lp64 %s" % (libdir, tail)
            info.update(c_link=c_link, f_link=f_link, blas_c=[c_link], lapack_c=[c_link],
                        blas_f=[f_link], lapack_f=[f_link], petsc="--with-blaslapack-dir=$MKLROOT")
        else:  # libsci
            root = (self.env.get("CRAY_PE_LIBSCI_PREFIX_DIR") or self.env.get("CRAY_LIBSCI_PREFIX_DIR"))
            serial, mpi = "sci_gnu", "sci_gnu_mpi"
            if root and os.path.isdir(os.path.join(root, "lib")):
                libdir = os.path.join(root, "lib")
                libs = sorted((f[3:-3] for f in os.listdir(libdir)
                               if f.startswith("libsci_gnu") and f.endswith(".so")), key=len)
                serial = next((l for l in libs if "_mp" not in l), serial)
                mpi = next((l for l in libs if l.endswith("_mpi")), mpi)
            else:
                libdir = "$CRAY_PE_LIBSCI_PREFIX_DIR/lib"
                warnings.append("cray-libsci location not found after loading modules")
            lib = os.path.join(libdir, "lib%s.so" % serial)
            link = "-L%s -l%s" % (libdir, serial)
            info.update(c_link=link, f_link=link, blas_c=[lib], lapack_c=[lib], blas_f=[lib],
                        lapack_f=[lib], scalapack_link="-L%s -l%s -l%s" % (libdir, mpi, serial))
        return info

    def _find_libxml2(self):
        prefix = self.loc.get("libxml2")
        if prefix:
            inc = os.path.join(prefix, "include", "libxml2")
            lib = os.path.join(prefix, "lib64")
            if not os.path.exists(os.path.join(lib, "libxml2.so")):
                lib = os.path.join(prefix, "lib")
            return inc, lib, True
        if self._which("pkg-config"):
            try:
                out = subprocess.run(["pkg-config", "--variable=includedir", "--variable=libdir",
                                      "libxml-2.0"], stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, env=self.env)
                vals = out.stdout.decode().split()
                if out.returncode == 0 and len(vals) == 2:
                    inc, lib = os.path.join(vals[0], "libxml2"), vals[1]
                    if os.path.isdir(inc) and os.path.exists(os.path.join(lib, "libxml2.so")):
                        return inc, lib, True
            except OSError:
                pass
        inc = "/usr/include/libxml2"
        for lib in ("/usr/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu", "/usr/lib"):
            if os.path.isdir(inc) and os.path.exists(os.path.join(lib, "libxml2.so")):
                return inc, lib, True
        return inc, "/usr/lib64", False


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def build_package_list(ctx):
    c = ctx.cfg
    pkgs = []
    if c.blas == "openblas":
        pkgs.append(OpenBLAS(ctx))
    elif c.blas == "blis+flame":
        pkgs += [BLIS(ctx), LibFlame(ctx)]
    pkgs += [Alglib(ctx), Libxc(ctx), Spglib(ctx), P4est(ctx), Kokkos(ctx)]
    if c.blas != "libsci":
        pkgs.append(ScaLAPACK(ctx))
    pkgs.append(ELPA(ctx))
    if c.petsc:
        pkgs += [PETSc(ctx, "real"), PETSc(ctx, "complex"), SLEPc(ctx, "real"), SLEPc(ctx, "complex"),
                 DealII(ctx, "real"), DealII(ctx, "complex")]
    else:
        pkgs.append(DealII(ctx))
    for name in ("dftd3", "dftd4"):
        if c[name]:
            pkgs.append(Dftd(ctx, name))
    pkgs.append(DFTFE(ctx))
    if c.invdft:
        pkgs.append(InvDFT(ctx))
    return pkgs


def expand_names(names, pkgs, what, allow_externals=False, allow_all=False, exact=False):
    """Map user-given names (with aliases petsc/slepc/dealii unless exact) to package names."""
    present = [p.name for p in pkgs]
    out = []
    for n in names:
        if allow_all and n == "all":
            out += present
            continue
        if allow_externals and n in EXTERNALS:
            out.append(n)
            continue
        matches = [p.name for p in pkgs if p.name == n or (not exact and p.key == n)]
        if not matches:
            known = present + (EXTERNALS if allow_externals else [])
            raise InstallError("%s: '%s' is not part of this build (packages: %s)"
                               % (what, n, ", ".join(known)))
        out += matches
    return list(dict.fromkeys(out))


def read_stamp(pkg):
    try:
        with open(pkg.stampfile()) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def signature(ctx, pkg, sigs):
    ctx.sig_mode = True
    try:
        h = hashlib.sha256()
        h.update(("%s\n%s\n" % (pkg.name, pkg.version)).encode())
        for st in pkg.build_steps(clean=False):
            h.update(st.cwd.encode() + b"\n" + st.cmd.encode() + b"\n")
        for d in pkg.deps():
            h.update(sigs.get(d, "").encode())
        return h.hexdigest()
    finally:
        ctx.sig_mode = False


def make_plan(ctx, pkgs):
    """Return [(pkg, action, reason)], action in build/done/keep/existing/skip."""
    c = ctx.cfg
    only = set(expand_names(c.only, pkgs, "--only")) if c.only is not None else None
    skip = set(expand_names(c.skip, pkgs, "--skip"))
    force = set(expand_names(c.force, pkgs, "--force", allow_all=True))
    needed = set()
    if only is not None:  # everything the --only packages depend on, transitively
        by_name = {p.name: p for p in pkgs}
        todo = list(only)
        while todo:
            for d in by_name[todo.pop()].deps():
                if d in by_name and d not in needed:
                    needed.add(d)
                    todo.append(d)
    sigs, plan, missing = {}, [], []
    for p in pkgs:
        stamp = read_stamp(p)
        if p.name in ctx.use_existing:
            sigs[p.name] = "existing:" + ctx.use_existing[p.name]
            plan.append((p, "existing", ctx.use_existing[p.name]))
            continue
        if p.name in skip:
            sigs[p.name] = stamp["signature"] if stamp else "skipped"
            plan.append((p, "skip", "--skip" if stamp else "--skip (not installed by this script)"))
            continue
        sig = signature(ctx, p, sigs)
        if only is not None and p.name not in only:
            if stamp:
                sigs[p.name] = stamp["signature"]
                plan.append((p, "keep", "not in --only"))
            else:
                sigs[p.name] = sig
                if p.name in needed:
                    missing.append(p.name)
                plan.append((p, "skip", "not in --only, not installed"))
            continue
        sigs[p.name] = sig
        if only is not None:
            plan.append((p, "build", "--only"))
        elif p.name in force:
            plan.append((p, "build", "--force"))
        elif not stamp:
            plan.append((p, "build", "not installed"))
        elif stamp.get("signature") != sig:
            plan.append((p, "build", "configuration changed"))
        else:
            plan.append((p, "done", "up to date"))
        p.signature_value = sig
    if missing:
        raise InstallError("--only: these dependencies are not installed yet: %s. Add them to "
                           "--only, or use --use-existing NAME=PATH." % ", ".join(missing))
    return plan


def stale_dependents(plan):
    """Packages kept as-is although something they depend on is rebuilt by --force/--only."""
    rebuilt = {p.name for p, a, r in plan if a == "build" and r in ("--force", "--only")}
    out = []
    for p, action, _ in plan:
        if action in ("done", "keep"):
            hit = rebuilt & set(p.deps())
            if hit:
                out.append((p.name, sorted(hit)))
                rebuilt.add(p.name)
    return out


def fresh_build(reason):
    """DFT-FE/invDFT build dirs are wiped only for new or changed configurations; --force and
    --only rebuild incrementally (e.g. after a git pull). Dependencies always build from scratch."""
    return reason in ("not installed", "configuration changed")


def cleanup_dirs(ctx):
    c = ctx.cfg
    dirs = []
    if not c.keep_src_dep:
        dirs.append(ctx.src)
    if not c.keep_build_dep:
        dirs.append(ctx.bld)
    if not c.keep_logs_dep:
        dirs.append(ctx.logs)
    return dirs


# ---------------------------------------------------------------------------
# Output: summary, script rendering, execution
# ---------------------------------------------------------------------------

def print_summary(ctx, plan, notes):
    c = ctx.cfg
    p = ctx.prof
    print("=" * 78)
    print("Machine       : %s -- %s" % (ctx.machine_name, p["description"]))
    print("Prefix        : %s   (DFT-FE%s, env.sh)" % (ctx.P, ", invDFT" if c.invdft else ""))
    print("Deps prefix   : %s" % ctx.D)
    if c.gpu:
        print("GPU           : %s, arch %s (%s); ELPA GPU: %s; NCCL/RCCL: %s" % (
            c.gpu_vendor, c.gpu_arch, ctx.gpu["lang"], "yes" if c.elpa_gpu else "no",
            "yes" if c.dccl else "no"))
    else:
        print("GPU           : off")
    built = c.blas in ("openblas", "blis+flame")
    print("BLAS/LAPACK   : %s (%s)" % (c.blas, "built from source" if built else "from the environment"))
    print("PETSc/SLEPc   : %s    dftd3: %s    dftd4: %s    64-bit int: %s" % tuple(
        "yes" if x else "no" for x in (c.petsc, c.dftd3, c.dftd4, c.int64)))
    repo = DFTFE_REPOS.get(c.dftfe_repo, c.dftfe_repo)
    print("DFT-FE        : %s @ %s (%s)" % (repo, c.dftfe_branch, c.build_type))
    if c.invdft:
        print("invDFT        : %s @ %s" % (c.invdft_repo, c.invdft_branch))
    print("Modules       : %s" % (" ".join(ctx.modules()) or "(none)"))
    print("Parallel jobs : %d" % ctx.jobs)
    print("-" * 78)
    print("%-16s %-22s %s" % ("package", "version", "action"))
    for pkg, action, reason in plan:
        print("%-16s %-22s %s (%s)" % (pkg.name, pkg.version[:22], action, reason))
    print("-" * 78)
    dirs = cleanup_dirs(ctx)
    print("After success : %s" % (("delete " + ", ".join(dirs)) if dirs else "keep dependency sources, builds and logs"))
    for n in notes:
        print("NOTE: " + n)
    print("=" * 78)


def render_script(ctx, plan, env_text, fetch_only):
    lines = ["#!/bin/bash",
             "# Generated by %s on %s" % (SCRIPT, datetime.datetime.now().strftime("%Y-%m-%d %H:%M")),
             "# Command: %s" % " ".join(shlex.quote(a) for a in sys.argv),
             "set -eo pipefail", "",
             "# ---- environment (the installer also writes this to %s) ----"
             % os.path.join(ctx.P, "env.sh"),
             env_text]
    for pkg, action, reason in plan:
        if action != "build":
            continue
        steps = pkg.fetch_steps()
        if not fetch_only:
            steps += pkg.build_steps(clean=fresh_build(reason))
        lines.append("# ---- %s %s (%s) ----" % (pkg.name, pkg.version, reason))
        for st in steps:
            lines += ["mkdir -p %s" % q(st.cwd), "(", "cd %s" % q(st.cwd), st.cmd, ")"]
        lines.append("")
    if not fetch_only:
        dirs = cleanup_dirs(ctx)
        if dirs:
            lines.append("# ---- remove dependency sources/builds/logs ----")
            lines += ["rm -rf %s" % q(d) for d in dirs]
    return "\n".join(lines) + "\n"


class StepFailed(Exception):
    pass


def tail(path, n=40):
    try:
        with open(path, "rb") as f:
            data = f.read()[-20000:]
        return "\n".join(data.decode(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def run_steps(ctx, pkg, steps):
    log_path = pkg.logfile()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "ab") as log:
        for st in steps:
            os.makedirs(st.cwd, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log.write(("\n### [%s] cd %s\n%s\n" % (stamp, st.cwd, st.cmd)).encode())
            log.flush()
            proc = subprocess.Popen(["bash", "-c", "set -eo pipefail\n" + st.cmd], cwd=st.cwd,
                                    env=ctx.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            try:
                for line in iter(proc.stdout.readline, b""):
                    log.write(line)
                    if ctx.cfg.verbose:
                        sys.stdout.buffer.write(line)
                        sys.stdout.flush()
            except KeyboardInterrupt:
                proc.kill()
                raise
            if proc.wait() != 0:
                log.flush()
                raise StepFailed(log_path)


def write_stamp(pkg):
    path = pkg.stampfile()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"name": pkg.name, "version": pkg.version, "signature": pkg.signature_value,
                   "location": pkg.ctx.loc[pkg.name],
                   "date": datetime.datetime.now().isoformat(timespec="seconds")}, f, indent=2)


def claim_dirs(ctx):
    """Create src/build/logs, marking them as ours so cleanup never deletes a user's directory."""
    foreign = []
    for d in (ctx.src, ctx.bld, ctx.logs):
        marker = os.path.join(d, OWNER_MARKER)
        if os.path.isdir(d) and os.listdir(d) and not os.path.exists(marker):
            foreign.append(d)
            continue
        os.makedirs(d, exist_ok=True)
        open(marker, "a").close()
    return foreign


def cleanup(ctx, pkgs):
    protected = [ctx.loc[p.name] for p in pkgs if not p.dependency]
    for d in cleanup_dirs(ctx):
        if not os.path.exists(os.path.join(d, OWNER_MARKER)):
            if os.path.exists(d):
                print("  keeping %s (not created by %s)" % (d, SCRIPT))
            continue
        if any(os.path.commonpath([d, x]) == d for x in protected):
            print("  keeping %s (it contains the DFT-FE/invDFT sources)" % d)
            continue
        shutil.rmtree(d)
        print("  removed %s" % d)


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return "%dh%02dm%02ds" % (h, m, s) if h else "%dm%02ds" % (m, s)


def preflight(ctx, fetch_only):
    problems = []
    tools = ["curl", "tar", "git"] + ([] if fetch_only else ["make", "cmake"])
    for t in tools:
        if not ctx._which(t):
            problems.append("'%s' not found in PATH after loading modules" % t)
    if ctx.module_failures:
        problems.append("these modules failed to load: %s (change them with --modules / "
                        "--add-module)" % ", ".join(ctx.module_failures))
    if problems:
        raise InstallError("cannot start:\n  " + "\n  ".join(problems))


def execute(ctx, pkgs, plan, env_text):
    c = ctx.cfg
    os.makedirs(ctx.P, exist_ok=True)
    with open(os.path.join(ctx.P, "env.sh"), "w") as f:
        f.write(env_text)
    saved = {k: v for k, v in c.items() if k not in TRANSIENT}
    with open(os.path.join(ctx.P, "install_dftfe_config.json"), "w") as f:
        json.dump(saved, f, indent=2, sort_keys=True)
    foreign = claim_dirs(ctx)
    for d in foreign:
        print("WARNING: %s already exists and was not created by %s; it will not be deleted." % (d, SCRIPT))

    todo = [(p, r) for p, a, r in plan if a == "build"]
    t_all = time.time()
    for i, (pkg, reason) in enumerate(todo, 1):
        what = "fetch" if c.fetch_only else "build"
        sys.stdout.write("[%d/%d] %-15s %-12s %s ... " % (i, len(todo), pkg.name, pkg.version[:12], what))
        sys.stdout.flush()
        if c.verbose:
            print()
        t0 = time.time()
        steps = pkg.fetch_steps()
        if not c.fetch_only:
            steps += pkg.build_steps(clean=fresh_build(reason))
        try:
            run_steps(ctx, pkg, steps)
        except StepFailed as e:
            log = e.args[0]
            print("FAILED")
            print("-" * 78)
            print(tail(log))
            print("-" * 78)
            sys.stdout.flush()
            raise InstallError("%s failed; full log: %s\nFix the problem and rerun the same command "
                               "to resume." % (pkg.name, log))
        if not c.fetch_only:
            write_stamp(pkg)
        print("ok (%s)" % fmt_time(time.time() - t0))

    if c.fetch_only:
        print("All sources downloaded. Rerun without --fetch-only (e.g. on a compute node) to build.")
        return
    print("All done in %s." % fmt_time(time.time() - t_all))
    dirs = cleanup_dirs(ctx)
    if dirs:
        print("Cleaning up dependency sources/builds/logs:")
        cleanup(ctx, pkgs)
    print_outputs(ctx, pkgs)


def print_outputs(ctx, pkgs):
    c = ctx.cfg
    names = {p.name for p in pkgs if not p.dependency and read_stamp(p)}
    print()
    print("Before running, load the environment with:")
    print("  source %s" % os.path.join(ctx.P, "env.sh"))
    bt = c.build_type.lower()
    if "dftfe" in names:
        base = os.path.join(ctx.loc["dftfe"], "build", bt)
        print("DFT-FE executables:")
        print("  %s" % os.path.join(base, "real", "dftfe"))
        if not c.real_only:
            print("  %s" % os.path.join(base, "complex", "dftfe"))
    if "invdft" in names:
        print("invDFT executable:")
        print("  %s" % os.path.join(ctx.loc["invdft"], "build", bt, "real", "invDFT_exe"))


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

EPILOG = """
package names (for --only/--skip/--force/--use-existing/--extra-args):
  openblas | blis libflame, alglib, libxc, spglib, p4est, kokkos, scalapack, elpa,
  petsc-real petsc-complex slepc-real slepc-complex dealii-real dealii-complex
  (with PETSc) or dealii (without), dftd3, dftd4, dftfe, invdft.
  'petsc', 'slepc' and 'dealii' select all of their variants (not for --use-existing,
  which needs the exact name). --use-existing also accepts boost, libxml2 and dccl
  (NCCL/RCCL prefix) and dftfe (an existing DFT-FE checkout, built in-tree).

run control:
  --only PKG[,PKG]      rebuild just these packages; their dependencies must already be
                        installed (by an earlier run or --use-existing)
  --skip PKG            leave this package out of the run (assumed to be provided)
  --force PKG|all       rebuild even if already installed with the same configuration
  --use-existing PKG=PATH   use an existing installation instead of building one
  --config FILE         read settings from a JSON file (keys = option names with '_';
                        "machine" may be a name, a profile path, or an inline profile);
                        command-line options override it

See README.md for the full description of every option and of machine profiles.
  --dry-run             print the plan and every command, change nothing
  --emit-script FILE    write the commands as a standalone bash script, change nothing
  --interactive         answer questions instead of passing options
"""


def add_bool(group, name, help_on, help_off):
    dest = name.replace("-", "_")
    group.add_argument("--" + name, dest=dest, action="store_const", const=True, default=None, help=help_on)
    group.add_argument("--no-" + name, dest=dest, action="store_const", const=False, help=help_off)


def build_parser():
    p = argparse.ArgumentParser(
        prog=SCRIPT, formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EPILOG,
        description="Install DFT-FE (and optionally invDFT) with all dependencies.")
    g = p.add_argument_group("location and machine")
    g.add_argument("--machine", metavar="NAME|FILE.json",
                   help="machine profile: a name from machines/ (%s), a path to a profile "
                        "file, or 'auto' (default: auto-detect, else generic)"
                        % ", ".join(available_machines()))
    g.add_argument("--prefix", help="where DFT-FE/invDFT are checked out and built, and env.sh is written")
    g.add_argument("--prefix-dependencies", "--prefix_dependencies", dest="prefix_dependencies",
                   help="separate install prefix for all dependencies (default: --prefix)")
    g.add_argument("--jobs", "-j", type=int, help="parallel build jobs (default: min(#cpus, 16))")
    g.add_argument("--config", help="JSON file with settings")
    g.add_argument("--interactive", action="store_true", help="prompt for the main settings")
    g.add_argument("--list-machines", action="store_true", help="show machine profiles and exit")
    g.add_argument("--yes", "-y", dest="yes", action="store_const", const=True, default=None,
                   help="do not ask for confirmation")

    g = p.add_argument_group("features")
    add_bool(g, "gpu", "enable GPU support", "CPU-only build")
    g.add_argument("--gpu-vendor", choices=GPU_VENDORS)
    g.add_argument("--gpu-arch", help="e.g. 80 / sm_80 (A100), 90 (H100), gfx90a (MI250X), pvc (Intel)")
    add_bool(g, "elpa-gpu", "build ELPA with GPU kernels (default with --gpu)", "build ELPA CPU-only")
    g.add_argument("--blas", choices=BLAS_CHOICES,
                   help="BLAS/LAPACK: openblas and blis+flame are built from source; "
                        "mkl and libsci come from modules")
    add_bool(g, "petsc", "build PETSc/SLEPc and real+complex deal.II (default on with --invdft)",
             "no PETSc/SLEPc")
    add_bool(g, "dftd3", "build simple-dftd3", "no simple-dftd3")
    add_bool(g, "dftd4", "build dftd4", "no dftd4")
    add_bool(g, "dccl", "link NCCL (CUDA) / RCCL (HIP)", "no NCCL/RCCL")
    add_bool(g, "gpu-aware-mpi", "WITH_GPU_AWARE_MPI=ON (only if the machine supports it)", "off")
    add_bool(g, "int64", "USE_64BIT_INT (default, except with --invdft, which does not support it)",
             "32-bit integers (default with --invdft)")
    add_bool(g, "higher-quad-psp", "HIGHERQUAD_PSP=ON", "HIGHERQUAD_PSP=OFF")
    g.add_argument("--real-only", action="store_const", const=True, default=None,
                   help="build only the real DFT-FE executable")
    g.add_argument("--build-type", choices=["Release", "Debug"], help="DFT-FE/invDFT build type")

    g = p.add_argument_group("DFT-FE and invDFT sources")
    g.add_argument("--dftfe-repo", help="github (default), bitbucket, or a git URL")
    g.add_argument("--dftfe-branch", help="default: publicGithubDevelop")
    g.add_argument("--dftfe-src", help="DFT-FE checkout location (default: <prefix>/dftfe)")
    g.add_argument("--invdft", action="store_const", const=True, default=None, help="also build invDFT")
    g.add_argument("--invdft-repo", help="default: %s" % INVDFT_REPO)
    g.add_argument("--invdft-branch", help="default: invGKS")
    g.add_argument("--invdft-src", help="invDFT checkout location (default: <prefix>/invDFT)")
    g.add_argument("--git-pull", action="store_const", const=True, default=None,
                   help="update existing DFT-FE/invDFT checkouts (fetch, checkout branch, pull)")

    g = p.add_argument_group("toolchain overrides")
    g.add_argument("--modules", help="replace the profile's module list (space separated)")
    g.add_argument("--add-module", action="append", help="load an extra module (repeatable)")
    for k in ("cc", "cxx", "fc", "mpicc", "mpicxx", "mpifc"):
        g.add_argument("--" + k, help="%s compiler" % k)
    g.add_argument("--cpu-arch-flags", help="CPU flags, e.g. '-march=znver3' (default from profile)")
    g.add_argument("--device-flags", help="CUDA/HIP/SYCL flags for DFT-FE/invDFT")
    g.add_argument("--cxx-flags", help="CMAKE_CXX_FLAGS for DFT-FE/invDFT")
    g.add_argument("--pkg-version", dest="versions", action="append", metavar="PKG=VER",
                   help="override a dependency version (repeatable)")
    g.add_argument("--extra-args", action="append", metavar="PKG=ARGS",
                   help="extra configure/cmake/make arguments for a package (repeatable)")

    g = p.add_argument_group("run control (see below)")
    g.add_argument("--only", action="append", metavar="PKG[,PKG]")
    g.add_argument("--skip", action="append", metavar="PKG")
    g.add_argument("--force", action="append", metavar="PKG")
    g.add_argument("--use-existing", action="append", metavar="PKG=PATH")
    g.add_argument("--dry-run", action="store_const", const=True, default=None)
    g.add_argument("--emit-script", metavar="FILE")
    g.add_argument("--fetch-only", action="store_const", const=True, default=None,
                   help="only download sources (e.g. on a login node with internet access)")
    g.add_argument("--verbose", "-v", action="store_const", const=True, default=None,
                   help="show build output on the terminal (it always goes to the logs)")

    g = p.add_argument_group("cleanup (dependency src/build/logs are deleted after success by default)")
    g.add_argument("--keep-src-dep", action="store_const", const=True, default=None)
    g.add_argument("--keep-build-dep", action="store_const", const=True, default=None)
    g.add_argument("--keep-logs-dep", action="store_const", const=True, default=None)
    return p


def ask(prompt, default=None, choices=None, required=False):
    while True:
        hint = " (%s)" % "/".join(choices) if choices else ""
        dflt = " [%s]" % default if default not in (None, "") else ""
        try:
            ans = input("%s%s%s: " % (prompt, hint, dflt)).strip()
        except EOFError:
            raise InstallError("input ended during interactive setup")
        if not ans:
            ans = default
        if required and ans in (None, ""):
            print("  a value is required")
            continue
        if choices and ans not in choices:
            print("  choose one of: %s" % ", ".join(choices))
            continue
        return ans


def ask_bool(prompt, default):
    ans = ask(prompt, "y" if default else "n", choices=["y", "n"])
    return ans == "y"


def interactive(cli, file_cfg):
    def get(key, fallback):
        v = cli.get(key, file_cfg.get(key))
        return fallback if v is None else v

    print("Interactive setup -- press Enter to accept the [default].")
    machine = get("machine", "auto")
    if isinstance(machine, dict):
        print("Machine: inline profile from the config file")
    else:
        if machine == "auto":
            machine = detect_machine()
        while True:
            machine = ask("Machine (%s, or a profile .json path)" % "/".join(available_machines()),
                          machine)
            try:
                load_profile(machine)
                break
            except InstallError as e:
                print("  %s" % e)
    pd = load_profile(machine)[1]["defaults"]
    ans = {"machine": machine}
    ans["prefix"] = ask("Install prefix for DFT-FE/invDFT", get("prefix", None), required=True)
    ans["prefix_dependencies"] = ask("Prefix for dependencies (Enter = same as above)",
                                     get("prefix_dependencies", "")) or None
    ans["gpu"] = ask_bool("Enable GPU support", get("gpu", pd.get("gpu", False)))
    if ans["gpu"]:
        ans["gpu_vendor"] = ask("GPU vendor", get("gpu_vendor", pd.get("gpu_vendor", "nvidia")),
                                choices=GPU_VENDORS)
        arch_default = pd.get("gpu_arch") if ans["gpu_vendor"] == pd.get("gpu_vendor") else None
        if ans["gpu_vendor"] == "intel":
            arch_default = "pvc"
        ans["gpu_arch"] = ask("GPU architecture (80/90 for A100/H100, gfx90a for MI250X, pvc)",
                              get("gpu_arch", arch_default), required=True)
        if ans["gpu_vendor"] != "intel":
            ans["elpa_gpu"] = ask_bool("Build ELPA with GPU support", get("elpa_gpu", True))
        ans["dccl"] = ask_bool("Link NCCL/RCCL", get("dccl", False))
    ans["blas"] = ask("BLAS/LAPACK", get("blas", pd.get("blas", "openblas")), choices=BLAS_CHOICES)
    ans["invdft"] = ask_bool("Also build invDFT", get("invdft", False))
    ans["petsc"] = ask_bool("Build PETSc/SLEPc (all-electron Gram-Schmidt)", get("petsc", ans["invdft"]))
    ans["dftd3"] = ask_bool("Build simple-dftd3", get("dftd3", False))
    ans["dftd4"] = ask_bool("Build dftd4", get("dftd4", False))
    ans["dftfe_repo"] = ask("DFT-FE repository (github, bitbucket or URL)", get("dftfe_repo", "github"))
    ans["dftfe_branch"] = ask("DFT-FE branch", get("dftfe_branch", "publicGithubDevelop"))
    ans["jobs"] = int(ask("Parallel build jobs", str(get("jobs", default_jobs()))))
    cli.update({k: v for k, v in ans.items() if v is not None})

    parts = ["python3", SCRIPT]
    for k, v in ans.items():
        if v is None or isinstance(v, dict):  # an inline machine profile only fits in --config
            continue
        flag = k.replace("_", "-")
        if v is True:
            parts.append("--" + flag)
        elif v is False:
            if k != "invdft":  # --invdft has no --no- form; leaving it out means no
                parts.append("--no-" + flag)
        else:
            parts += ["--" + flag, str(v)]
    print("\nEquivalent command:\n  " + " ".join(shlex.quote(x) for x in parts) + "\n")
    if isinstance(ans["machine"], dict):
        print("(the inline machine profile is only kept if you save a config file below)\n")
    path = ask("Save these settings to a JSON config file (Enter to skip)", "")
    if path:
        with open(absp(path), "w") as f:
            json.dump(ans, f, indent=2)
        print("  saved %s (use it with --config)" % absp(path))
    return cli


def list_machines():
    print("Machine profiles in %s:" % MACHINES_DIR)
    for name in available_machines():
        try:
            p = load_profile(name)[1]
        except InstallError as e:
            print("%-11s (invalid: %s)" % (name, e))
            continue
        d = p["defaults"]
        gpu = "%s %s" % (d.get("gpu_vendor"), d.get("gpu_arch")) if d.get("gpu") else "off"
        print("%-11s %s" % (name, p["description"]))
        print("%-11s modules: %s" % ("", " ".join(p["modules"]) or "(none)"))
        print("%-11s default GPU: %s, default BLAS: %s" % ("", gpu, d.get("blas", "openblas")))
    print("This machine is detected as: %s" % detect_machine())


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv)
    if args.list_machines:
        list_machines()
        return 0
    file_cfg = load_config_file(args.config) if args.config else {}
    cli = {k: v for k, v in vars(args).items()
           if v is not None and k not in ("config", "interactive", "list_machines")}
    if args.interactive or not argv:
        cli = interactive(cli, file_cfg)

    cfg, notes, machine_name, profile = resolve_config(cli, file_cfg)
    ctx = Context(cfg, machine_name, profile)
    pkgs = build_package_list(ctx)
    expand_names(list(cfg.use_existing), pkgs, "--use-existing", allow_externals=True, exact=True)
    expand_names(list(cfg.extra_args), pkgs, "--extra-args")
    ctx.set_locations(pkgs)  # needed for env.sh, before the environment is captured
    env_text = ctx.env_sh(pkgs)
    ctx.capture_env(env_text)
    notes += ctx.finalize(pkgs)
    if ctx.module_failures:
        notes.append("modules that failed to load: %s" % ", ".join(ctx.module_failures))
    plan = make_plan(ctx, pkgs)
    for name, deps in stale_dependents(plan):
        notes.append("%s is kept as-is although %s will be rebuilt; add --force %s if needed"
                     % (name, ", ".join(deps), name))
    if not os.environ.get("SLURM_JOB_ID") and not cfg.fetch_only:
        notes.append("not inside a Slurm job: builds on login nodes may run out of memory, and "
                     "GPU builds should run where the GPU toolchain is available")

    print_summary(ctx, plan, notes)

    if cfg.dry_run or cfg.emit_script:
        script = render_script(ctx, plan, env_text, cfg.fetch_only)
        if cfg.emit_script:
            path = absp(cfg.emit_script)
            with open(path, "w") as f:
                f.write(script)
            os.chmod(path, 0o755)
            print("Wrote %s" % path)
        else:
            print(script)
        return 0

    if not any(a == "build" for _, a, _ in plan):
        print("Nothing to do: everything is up to date.")
        print_outputs(ctx, pkgs)
        return 0
    preflight(ctx, cfg.fetch_only)
    if not cfg.yes and sys.stdin.isatty():
        if not ask_bool("Proceed", True):
            print("Aborted.")
            return 1
    execute(ctx, pkgs, plan, env_text)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except InstallError as e:
        sys.stderr.write("\nERROR: %s\n" % e)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted. Rerun the same command to resume.\n")
        sys.exit(130)
