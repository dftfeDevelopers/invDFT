#!/bin/bash
# Installation script for DFT-FE (forInvGKSCalc/invGKS) and invDFT (invGKS),
# together with all of their dependencies.
#
# Usage (either way works); --prefix is the install location (WD) and is required
# unless you accept the default $PSCRATCH/install_invDFT:
#   source perlmutter.sh --prefix=/path/to/install && install_all
#   ./perlmutter.sh --prefix=/path/to/install install_all
#   ./perlmutter.sh --prefix /path/to/install install_blis install_libflame
#
# The environment section below runs every time this file is sourced or executed:
# it loads the Perlmutter modules and sets WD (from --prefix) and INST.

# ---------------------------------------------------------------------------
# Command-line options
# ---------------------------------------------------------------------------
#   --prefix=DIR | --prefix DIR   install location (becomes WD)
#   anything else                 name of a function to run (when executed directly)
_usage() {
    echo "Usage: $0 [--prefix=<install_dir>] <function> [<function> ...]" >&2
    echo "       e.g. $0 --prefix=\$PSCRATCH/install_invDFT install_all" >&2
}

_parse_args() {
    PREFIX=""
    FUNCS=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --prefix=*) PREFIX="${1#--prefix=}"; shift ;;
            --prefix)   if [ -z "${2:-}" ]; then echo "ERROR: --prefix needs a value." >&2; return 1; fi
                        PREFIX="$2"; shift 2 ;;
            -h|--help)  _usage; return 1 ;;
            -*)         echo "ERROR: unknown option: $1" >&2; _usage; return 1 ;;
            *)          FUNCS+=("$1"); shift ;;
        esac
    done
}

if ! _parse_args "$@"; then
    if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then exit 1; else return 1; fi
fi

# ---------------------------------------------------------------------------
# Environment (NERSC Perlmutter)
# ---------------------------------------------------------------------------
module load gcc-native
module load PrgEnv-gnu
module load cray-mpich
module load cudatoolkit
module load cray-libsci
module load cmake
module load python/3.11-24.1.0

# invDFT source = the git repo this script lives in (<repo>/installationScripts/perlmutter.sh).
# Override by setting INVDFT_SRC before running if the script is kept elsewhere.
_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INVDFT_SRC="${INVDFT_SRC:-$(cd "$_script_dir/.." && pwd)}"


# Build locations
# Install location: --prefix if given, otherwise the old default.
mkdir -p "${PREFIX:-$PSCRATCH/install_invDFT}" || return 1 2>/dev/null || exit 1
export WD="$(cd "${PREFIX:-$PSCRATCH/install_invDFT}" && pwd)"   # absolute path
export INST="$WD/env2"
mkdir -p "$WD/src" "$INST"
echo "Install location (WD): $WD"

# MPICH_DIR (cray-mpich) and CUDA_HOME (cudatoolkit) are set by the modules above.
# NCCL_DIR is optional: it is only added to CMAKE_PREFIX_PATH, and DCCL is turned OFF in the build.

# ---------------------------------------------------------------------------
# Installation functions
# ---------------------------------------------------------------------------
# Branches checked out by compile_dftfe and compile_invDFT_invGKS
DFTFE_BRANCH=invGKS
INVDFT_BRANCH=invGKS


install_blis() {
    echo "===== [install_blis] Installing BLIS 3.0.1 (BLAS library) ====="
    cd "$WD/src"
    if [ ! -d blis-3.0.1 ]; then
        wget 'https://github.com/amd/blis/archive/refs/tags/3.0.1.tar.gz'
        tar xzf '3.0.1.tar.gz'
        rm -f '3.0.1.tar.gz'
    fi
    cd blis-3.0.1
    ./configure "--prefix=$INST" \
                'CC=cc' 'CXX=CC' 'FC=ftn' \
                'CFLAGS=-O2' \
                'CPPFLAGS=-O2' \
                'FFLAGS=-O2' \
                'zen'

    make -j16
    make install
    cd "$WD"
    echo "===== [install_blis] Completed: BLIS 3.0.1 (BLAS library) installed ====="
}

install_libflame() {
    echo "===== [install_libflame] Installing libflame 5.2.0 (LAPACK library) ====="
    cd "$WD/src"
    if [ ! -d libflame-5.2.0 ]; then
        wget 'https://github.com/flame/libflame/archive/refs/tags/5.2.0.tar.gz'
        tar xzf '5.2.0.tar.gz'
        rm -f '5.2.0.tar.gz'
    fi
    cd libflame-5.2.0

    ./configure "--prefix=$INST" \
                'CC=cc' 'CXX=CC' 'FC=ftn' \
                'CFLAGS=-O2' \
                'CPPFLAGS=-O2' \
                'FFLAGS=-O2' \
                '--enable-dynamic-build' \
                '--enable-lapack2flame' \
                '--enable-max-arg-list-hack'

    make -j16
    make install
    cd "$WD"
    echo "===== [install_libflame] Completed: libflame 5.2.0 (LAPACK library) installed ====="
}


# Install alglib, libxc, spglib and p4est using the typical route (cf. manual)
install_alglib() {
    echo "===== [install_alglib] Installing ALGLIB 3.20.0 ====="
    cd "$WD/src"
    if [ ! -d alglib-cpp ]; then
        wget 'https://www.alglib.net/translator/re/alglib-3.20.0.cpp.gpl.tgz'
        tar xzf alglib-3.20.0.cpp.gpl.tgz
        rm -f alglib-3.20.0.cpp.gpl.tgz
    fi
    cd alglib-cpp/src
    # TODO: change to CC (in case of linking error)
    g++ -o libAlglib.so -shared -fPIC -O2 *.cpp

    mkdir -p "$INST/lib/alglib"
    mv libAlglib.so "$INST/lib/alglib/"
    cp *.h "$INST/lib/alglib/"

    cd "$WD"
    echo "===== [install_alglib] Completed: ALGLIB 3.20.0 installed ====="
}

install_libxc() {
    echo "===== [install_libxc] Installing Libxc 6.2.2 ====="
    cd "$WD/src"
    if [ ! -d libxc-6.2.2 ]; then
        wget 'https://gitlab.com/libxc/libxc/-/archive/6.2.2/libxc-6.2.2.tar.gz'
        tar xzf 'libxc-6.2.2.tar.gz'
        rm 'libxc-6.2.2.tar.gz'
    fi
    cd libxc-6.2.2
    rm -fr build
    mkdir build && cd build
    cmake '-DCMAKE_C_COMPILER=cc' \
          '-DCMAKE_C_FLAGS=-O2 -fPIC' \
          "-DCMAKE_INSTALL_PREFIX=$INST" \
          '-DBUILD_SHARED_LIBS=ON' \
          ..
    make -j16
    make install
    cd "$WD"
    echo "===== [install_libxc] Completed: Libxc 6.2.2 installed ====="
}


install_dftd4() {
    echo "===== [install_dftd4] Installing DFT-D4 3.7.0 ====="
    cd "$WD/src"
    if [ ! -d dftd4-3.7.0 ]; then
        wget 'https://github.com/dftd4/dftd4/archive/refs/tags/v3.7.0.tar.gz'
        tar xzf v3.7.0.tar.gz
        rm v3.7.0.tar.gz
    fi
    cd dftd4-3.7.0
    rm -fr build
    mkdir build && cd build
    cmake '-DCMAKE_Fortran_COMPILER=ftn' \
          '-DCMAKE_C_COMPILER=cc' \
          "-DBLAS_LIBRARIES=$INST/lib/libblis.so" \
          "-DLAPACK_LIBRARIES=$INST/lib/libflame.so" \
          '-DBUILD_SHARED_LIBS=ON' \
          "-DCMAKE_INSTALL_PREFIX=$INST" \
          '-DWITH_OpenMP=OFF' \
          ..
    make -j16
    make install
    cd "$WD"
    echo "===== [install_dftd4] Completed: DFT-D4 3.7.0 installed ====="
}

install_spglib() {
    echo "===== [install_spglib] Installing spglib ====="
    cd "$WD/src"
    if [ ! -d spglib ]; then
        git clone 'https://github.com/atztogo/spglib.git'
        (cd spglib && git checkout 02159eef6e7349535049a43fe2272bb634c77945)
    fi
    cd spglib
    rm -fr build
    mkdir -p build && cd build
    cmake '-DCMAKE_CXX_COMPILER=CC' \
          '-DCMAKE_C_COMPILER=cc' \
          "-DCMAKE_INSTALL_PREFIX=$INST" \
          ..
    make -j16
    make install
    cd "$WD"
    echo "===== [install_spglib] Completed: spglib installed ====="
}

install_p4est() {
    echo "===== [install_p4est] Installing p4est 2.8.7 ====="
    cd "$WD/src"
    rm -rf p4est
    mkdir p4est
    cd p4est
    wget 'https://p4est.github.io/release/p4est-2.8.7.tar.gz'
    wget 'https://raw.githubusercontent.com/dftfeDevelopers/dftfe/manual/p4est-setup-craycompiler.sh'
    chmod u+x p4est-setup-craycompiler.sh
    ./p4est-setup-craycompiler.sh p4est-2.8.7.tar.gz "$INST"
    cd "$WD"
    echo "===== [install_p4est] Completed: p4est 2.8.7 installed ====="
}

# Install netlib-scalapack 2.2.0 version linking to openblas
# note that the openblas (sourced via module) provides lapack
install_scalapack() {
    echo "===== [install_scalapack] Installing ScaLAPACK 2.2.0 ====="
    cd "$WD/src"
    if [ ! -d scalapack-2.2.0 ]; then
        wget 'https://github.com/Reference-ScaLAPACK/scalapack/archive/refs/tags/v2.2.0.tar.gz'
        tar xzf v2.2.0.tar.gz
        rm -f v2.2.0.tar.gz
    fi
    cd scalapack-2.2.0

    mkdir -p build
    cd build
    cmake \
        '-DBUILD_SHARED_LIBS=ON' \
        '-DBUILD_STATIC_LIBS=OFF' \
        '-DBUILD_TESTING=OFF' \
        '-DCMAKE_C_COMPILER=cc' \
        '-DCMAKE_Fortran_COMPILER=ftn' \
        '-DCMAKE_C_FLAGS=-fPIC -march=znver3 -Wno-implicit-function-declaration' \
        '-DCMAKE_Fortran_FLAGS=-fPIC -march=znver3 -fallow-argument-mismatch' \
        '-DUSE_OPTIMIZED_LAPACK_BLAS=ON' \
        "-DBLAS_LIBRARIES=$INST/lib/libblis.so" \
        "-DLAPACK_LIBRARIES=$INST/lib/libflame.so" \
        "-DCMAKE_INSTALL_PREFIX=$INST" \
        ..
    make -j16
    make install
    cd "$WD"
    echo "===== [install_scalapack] Completed: ScaLAPACK 2.2.0 installed ====="
}

# Install ELPA latest version (elpa-2025.01.001) with NVIDIA GPU support
install_elpa() {
    echo "===== [install_elpa] Installing ELPA 2025.01.001 (with NVIDIA GPU support) ====="
    cd "$WD/src"
    if [ ! -d elpa ]; then
        local ver=2025.01.001
        wget "https://elpa.mpcdf.mpg.de/software/tarball-archive/Releases/$ver/elpa-$ver.tar.gz"
        tar xzf "elpa-$ver.tar.gz"
        mv "elpa-$ver" elpa
        rm -f "elpa-$ver.tar.gz"
    fi
    cd elpa

    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$INST/lib"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$INST/lib64"
    local cflags="-march=znver3 -fPIC -O2 -target-accel=nvidia80 -I${MPICH_DIR}/include"
    local libs="-L$INST/lib -lscalapack -L$INST/lib -lblis -L$INST/lib -lflame"
    rm -fr build
    mkdir build && cd build
    ../configure 'CXX=CC' 'CC=cc' 'FC=ftn' \
                 "FCFLAGS=-ffree-line-length-none $cflags" \
                 "CFLAGS=$cflags" \
                 "CXXFLAGS=-std=c++17 $cflags" \
                 "--prefix=$INST" \
                 "LIBS=$libs" \
                 --disable-sse --disable-sse-assembly --disable-avx \
                 --disable-avx2 --disable-avx512 '--enable-c-tests=no' \
                 '--enable-option-checking=fatal' --enable-shared \
                 '--enable-cpp-tests=no' '--enable-nvidia-gpu' \
                 '--enable-gpu-streams=nvidia' \
                 '--with-NVIDIA-GPU-compute-capability=sm_80' \
                 "--with-cuda-path=$CUDA_HOME" \
                 "--with-cuda-sdk-path=$CUDA_HOME" \
                 '--without-threading-support-check-during-build'
    make -j16
    make install
    cd "$WD"
    echo "===== [install_elpa] Completed: ELPA 2025.01.001 (with NVIDIA GPU support) installed ====="
}

install_kokkos() {
    echo "===== [install_kokkos] Installing Kokkos 4.3.00 ====="
    cd "$WD/src"
    if [ ! -d kokkos-4.3.00 ]; then
        wget 'https://github.com/kokkos/kokkos/archive/refs/tags/4.3.00.tar.gz'
        tar xzvf '4.3.00.tar.gz'
        rm '4.3.00.tar.gz'
    fi
    cd kokkos-4.3.00
    rm -fr build
    mkdir build && cd build
    cmake '-DCMAKE_C_COMPILER=cc' \
          '-DCMAKE_C_FLAGS=-O2 -fPIC' \
          '-DCMAKE_CXX_COMPILER=CC' \
          '-DCMAKE_CXX_FLAGS=-O2 -fPIC' \
          "-DCMAKE_INSTALL_PREFIX=$INST" \
          ..
    make -j16
    make install
    cd "$WD"
    echo "===== [install_kokkos] Completed: Kokkos 4.3.00 installed ====="
}


install_petsc() {
    echo "===== [install_petsc] Installing PETSc 3.21.1 (real and complex builds) ====="
    cd "$WD/src"
    if [ ! -d petsc-3.21.1 ]; then
        wget https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-3.21.1.tar.gz
        tar xf petsc-3.21.1.tar.gz
        rm -f petsc-3.21.1.tar.gz
    fi
    cd petsc-3.21.1

    # ---- real build ----
    rm -fr build_real
    mkdir build_real
    export PETSC_DIR="$WD/src/petsc-3.21.1"
    unset PETSC_ARCH
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$INST/lib"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$INST/lib64"
    ./configure "--prefix=$WD/src/petsc-3.21.1/build_real/" \
                '--with-debugging=no' \
                '--with-64-bit-indices=true' \
                '--with-cc=cc' '--with-cxx=CC' '--with-fc=ftn' \
                '--with-fortran-kernels=true' '--with-scalar-type=real' \
                "--with-blas-lapack-lib=$INST/lib/libblis.so" \
                'CFLAGS=-O2' 'CXXFLAGS=-O2' 'FFLAGS=-O2' \
                'CXXOPTFLAGS=-O2' 'COPTFLAGS=-O2' 'FOPTFLAGS=-O2' || return 1

    # (libflame can be appended to the blas-lapack list as ';'$INST/lib/libflame.so)
    make PETSC_DIR="$WD/src/petsc-3.21.1" PETSC_ARCH=arch-linux-c-opt PYTHON="$HOME/bin/python3" all
    make PETSC_DIR="$WD/src/petsc-3.21.1" PETSC_ARCH=arch-linux-c-opt PYTHON="$HOME/bin/python3" install

    # ---- complex build ----
    cd "$WD/src/petsc-3.21.1"
    rm -fr build_complex
    mkdir build_complex
    export PETSC_DIR="$WD/src/petsc-3.21.1"
    unset PETSC_ARCH
    ./configure "--prefix=$WD/src/petsc-3.21.1/build_complex/" \
                '--with-debugging=no' \
                '--with-64-bit-indices=true' \
                '--with-cc=cc' '--with-cxx=CC' '--with-fc=ftn' \
                '--with-fortran-kernels=true' '--with-scalar-type=complex' \
                "--with-blas-lapack-lib=$INST/lib/libblis.so" \
                'CFLAGS=-O2' 'CXXFLAGS=-O2' 'FFLAGS=-O2' \
                'CXXOPTFLAGS=-O2' 'COPTFLAGS=-O2' 'FOPTFLAGS=-O2' || return 1

    make PETSC_DIR="$WD/src/petsc-3.21.1" PETSC_ARCH=arch-linux-c-opt PYTHON="$HOME/bin/python3" all
    make PETSC_DIR="$WD/src/petsc-3.21.1" PETSC_ARCH=arch-linux-c-opt PYTHON="$HOME/bin/python3" install
    cd "$WD"
    echo "===== [install_petsc] Completed: PETSc 3.21.1 (real and complex builds) installed ====="
}

install_slepc() {
    echo "===== [install_slepc] Installing SLEPc 3.21.1 (real and complex builds) ====="
    cd "$WD/src"
    if [ ! -d slepc-3.21.1 ]; then
        wget https://slepc.upv.es/download/distrib/slepc-3.21.1.tar.gz
        tar xf slepc-3.21.1.tar.gz
        rm -f slepc-3.21.1.tar.gz
    fi
    cd slepc-3.21.1

    # ---- real build ----
    rm -rf build_real
    mkdir build_real
    export PETSC_DIR="$WD/src/petsc-3.21.1/build_real"
    unset PETSC_ARCH
    ./configure "--prefix=$WD/src/slepc-3.21.1/build_real" || return 1
    make SLEPC_DIR="$WD/src/slepc-3.21.1/" PETSC_DIR="$WD/src/petsc-3.21.1/build_real" PYTHON="$HOME/bin/python3"
    make SLEPC_DIR="$WD/src/slepc-3.21.1/" PETSC_DIR="$WD/src/petsc-3.21.1/build_real" PYTHON="$HOME/bin/python3" install

    # ---- complex build ----
    cd "$WD/src/slepc-3.21.1"
    rm -rf build_complex
    mkdir build_complex
    export PETSC_DIR="$WD/src/petsc-3.21.1/build_complex"
    unset PETSC_ARCH
    ./configure "--prefix=$WD/src/slepc-3.21.1/build_complex" || return 1
    make SLEPC_DIR="$WD/src/slepc-3.21.1" PETSC_DIR="$WD/src/petsc-3.21.1/build_complex" PYTHON="$HOME/bin/python3"
    make SLEPC_DIR="$WD/src/slepc-3.21.1" PETSC_DIR="$WD/src/petsc-3.21.1/build_complex" PYTHON="$HOME/bin/python3" install
    cd "$WD"
    echo "===== [install_slepc] Completed: SLEPc 3.21.1 (real and complex builds) installed ====="
}

# Install latest release dealii from https://github.com/dealii/dealii
install_dealii() {
    echo "===== [install_dealii] Installing deal.II 9.6.2 ====="
    cd "$WD/src"
    local ver=9.6.2
    if [ ! -d "dealii-$ver" ]; then
        wget "https://github.com/dealii/dealii/releases/download/v$ver/dealii-$ver.tar.gz"
        tar xzf "dealii-$ver.tar.gz"
    fi
    cd "dealii-$ver"
    rm -fr build
    mkdir build
    cd build
    cmake '-DCMAKE_CXX_STANDARD=17' \
          '-DCMAKE_CXX_FLAGS=-std=c++17' \
          '-DCMAKE_C_FLAGS=-std=c++17' \
          '-DDEAL_II_FORCE_BUNDLED_BOOST=OFF' \
          '-DDEAL_II_ALLOW_PLATFORM_INTROSPECTION=OFF' \
          "-DKOKKOS_DIR=$INST" \
          '-DDEAL_II_WITH_TASKFLOW=OFF' \
          '-DCMAKE_BUILD_TYPE=Release' \
          '-DDEAL_II_CXX_FLAGS_RELEASE=-O2' \
          '-DCMAKE_C_COMPILER=cc' \
          '-DCMAKE_CXX_COMPILER=CC' \
          '-DCMAKE_Fortran_COMPILER=ftn' \
          '-DDEAL_II_WITH_TBB=OFF' \
          '-DDEAL_II_COMPONENT_EXAMPLES=OFF' \
          '-DDEAL_II_WITH_MPI=ON' \
          '-DDEAL_II_WITH_64BIT_INDICES=ON' \
          "-DP4EST_DIR=$INST" \
          '-DDEAL_II_WITH_LAPACK=ON' \
          "-DLAPACK_DIR=${OLCF_OPENBLAS_ROOT:+$OLCF_OPENBLAS_ROOT;}$INST" \
          '-DLAPACK_FOUND=true' \
          "-DLAPACK_LIBRARIES=$INST/lib/libblis.so;$INST/lib/libflame.so" \
          "-DCMAKE_INSTALL_PREFIX=$INST/dealii" \
          ..
    # may add to lapack libraries ';'$INST/lib64/liblapack.so
    make -j16 || echo 1
    make install
    mv "$INST"/dealii/{detailed.log,summary.log,README.md,LICENSE.md} "$INST/dealii/share/deal.II/"
    cd "$WD"
    echo "===== [install_dealii] Completed: deal.II 9.6.2 installed ====="
}


install_dealii_real() {
    echo "===== [install_dealii_real] Installing deal.II 9.6.2 (real build with PETSc/SLEPc) ====="
    cd "$WD/src"
    local ver=9.6.2
    if [ ! -d "dealii-$ver" ]; then
        wget "https://github.com/dealii/dealii/releases/download/v$ver/dealii-$ver.tar.gz"
        tar xzf "dealii-$ver.tar.gz"
    fi
    cd "dealii-$ver"
    rm -fr build_real
    mkdir build_real
    cd build_real
    cmake '-DCMAKE_CXX_STANDARD=17' \
          '-DCMAKE_CXX_FLAGS=-std=c++17' \
          '-DCMAKE_C_FLAGS=-std=c++17' \
          '-DDEAL_II_FORCE_BUNDLED_BOOST=OFF' \
          '-DDEAL_II_ALLOW_PLATFORM_INTROSPECTION=OFF' \
          "-DKOKKOS_DIR=$INST" \
          '-DDEAL_II_WITH_TASKFLOW=OFF' \
          '-DCMAKE_BUILD_TYPE=Release' \
          '-DDEAL_II_CXX_FLAGS_RELEASE=-O2' \
          '-DCMAKE_C_COMPILER=cc' \
          '-DCMAKE_CXX_COMPILER=CC' \
          '-DCMAKE_Fortran_COMPILER=ftn' \
          '-DDEAL_II_WITH_TBB=OFF' \
          '-DDEAL_II_COMPONENT_EXAMPLES=OFF' \
          '-DDEAL_II_WITH_MPI=ON' \
          '-DDEAL_II_WITH_64BIT_INDICES=ON' \
          '-DDEAL_II_WITH_PETSC=ON' \
          "-DPETSC_DIR=$WD/src/petsc-3.21.1/build_real" \
          '-DDEAL_II_WITH_SLEPC=ON' \
          "-DSLEPC_DIR=$WD/src/slepc-3.21.1/build_real" \
          "-DP4EST_DIR=$INST" \
          '-DDEAL_II_WITH_LAPACK=ON' \
          "-DLAPACK_DIR=${OLCF_OPENBLAS_ROOT:+$OLCF_OPENBLAS_ROOT;}$INST" \
          '-DLAPACK_FOUND=true' \
          "-DLAPACK_LIBRARIES=$INST/lib/libblis.so;$INST/lib/libflame.so" \
          "-DCMAKE_INSTALL_PREFIX=$INST/dealii_real" \
          ..
    # may add to lapack libraries ';'$INST/lib64/liblapack.so
    make -j16 || echo 1
    make install
    mv "$INST"/dealii_real/{detailed.log,summary.log,README.md,LICENSE.md} "$INST/dealii_real/share/deal.II/"
    cd "$WD"
    echo "===== [install_dealii_real] Completed: deal.II 9.6.2 (real build with PETSc/SLEPc) installed ====="
}

install_dealii_complex() {
    echo "===== [install_dealii_complex] Installing deal.II 9.6.2 (complex build with PETSc/SLEPc) ====="
    cd "$WD/src"
    local ver=9.6.2
    if [ ! -d "dealii-$ver" ]; then
        wget "https://github.com/dealii/dealii/releases/download/v$ver/dealii-$ver.tar.gz"
        tar xzf "dealii-$ver.tar.gz"
    fi
    cd "dealii-$ver"
    rm -fr build_complex
    mkdir build_complex
    cd build_complex
    cmake '-DCMAKE_CXX_STANDARD=17' \
          '-DCMAKE_CXX_FLAGS=-std=c++17' \
          '-DCMAKE_C_FLAGS=-std=c++17' \
          '-DDEAL_II_FORCE_BUNDLED_BOOST=OFF' \
          '-DDEAL_II_ALLOW_PLATFORM_INTROSPECTION=OFF' \
          "-DKOKKOS_DIR=$INST" \
          '-DDEAL_II_WITH_TASKFLOW=OFF' \
          '-DCMAKE_BUILD_TYPE=Release' \
          '-DDEAL_II_CXX_FLAGS_RELEASE=-O2' \
          '-DCMAKE_C_COMPILER=cc' \
          '-DCMAKE_CXX_COMPILER=CC' \
          '-DCMAKE_Fortran_COMPILER=ftn' \
          '-DDEAL_II_WITH_TBB=OFF' \
          '-DDEAL_II_COMPONENT_EXAMPLES=OFF' \
          '-DDEAL_II_WITH_MPI=ON' \
          '-DDEAL_II_WITH_64BIT_INDICES=ON' \
          '-DDEAL_II_WITH_PETSC=ON' \
          "-DPETSC_DIR=$WD/src/petsc-3.21.1/build_complex" \
          '-DDEAL_II_WITH_SLEPC=ON' \
          "-DSLEPC_DIR=$WD/src/slepc-3.21.1/build_complex" \
          "-DP4EST_DIR=$INST" \
          '-DDEAL_II_WITH_LAPACK=ON' \
          "-DLAPACK_DIR=${OLCF_OPENBLAS_ROOT:+$OLCF_OPENBLAS_ROOT;}$INST" \
          '-DLAPACK_FOUND=true' \
          "-DLAPACK_LIBRARIES=$INST/lib/libblis.so;$INST/lib/libflame.so" \
          "-DCMAKE_INSTALL_PREFIX=$INST/dealii_complex" \
          ..
    # may add to lapack libraries ';'$INST/lib64/liblapack.so
    make -j16 || echo 1
    make install
    mv "$INST"/dealii_complex/{detailed.log,summary.log,README.md,LICENSE.md} "$INST/dealii_complex/share/deal.II/"
    cd "$WD"
    echo "===== [install_dealii_complex] Completed: deal.II 9.6.2 (complex build with PETSc/SLEPc) installed ====="
}



install_numdiff() {
    echo "===== [install_numdiff] Installing numdiff 5.9.0 ====="
    cd "$WD/src"
    if [ ! -d numdiff-5.9.0 ]; then
        wget http://nongnu.askapache.com/numdiff/numdiff-5.9.0.tar.gz
        tar xvzf numdiff-5.9.0.tar.gz
        rm numdiff-5.9.0.tar.gz
    fi
    cd numdiff-5.9.0

    ./configure "--prefix=$INST/numdiff_install"
    make
    make install
    cd "$WD"
    echo "===== [install_numdiff] Completed: numdiff 5.9.0 installed ====="
}


compile_dftfe() {
    echo "===== [compile_dftfe] Compiling DFT-FE (branch $DFTFE_BRANCH) ====="
    cd "$WD/src"
    local branch=$DFTFE_BRANCH
    if [ ! -d dftfe ]; then
        git clone -b "$branch" https://github.com/dftfeDevelopers/dftfe.git dftfe
    else
        (cd dftfe && git fetch && git checkout "$branch" && git pull)
    fi
    cd dftfe
    local SRC
    SRC=$(pwd)
    mkdir -p build
    cd build

    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$INST/lib"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$INST/lib64"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$INST/numdiff_install"
    local alglibDir=$INST/lib/alglib
    local libxcDir=$INST
    local spglibDir=$INST
    local xmlIncludeDir=/usr/include/libxml2
    local xmlLibDir=/usr/lib64

    local ELPA_PATH=$INST
    local DCCL_PATH=$NCCL_DIR
    local TORCH_PATH=''

    # Compiler options and flags
    local cxx_compiler=CC
    local cxx_flags='-march=znver3 -fPIC'
    local cxx_flagsRelease="-fPIC -target-accel=nvidia80 -I${MPICH_DIR}/include"
    local device_flags="-I${MPICH_DIR}/include -arch=sm_80"
    local device_architectures=80

    # HIGHERQUAD_PSP option compiles with default or higher order
    # quadrature for storing pseudopotential data
    # ON is recommended for MD simulations with hard pseudopotentials

    # build type: "Release" or "Debug"
    local build_type=Release
    local out
    out=$(echo "$build_type" | tr '[:upper:]' '[:lower:]')

    # Note: MDI_PATH is not used by project.
    local cmake_flags=(
        -DCMAKE_CXX_STANDARD=17
        "-DCMAKE_CXX_COMPILER=$cxx_compiler"
        "-DCMAKE_CXX_FLAGS=$cxx_flags"
        "-DCMAKE_CXX_FLAGS_RELEASE=$cxx_flagsRelease"
        "-DCMAKE_BUILD_TYPE=$build_type"
        "-DALGLIB_DIR=$alglibDir"
        "-DLIBXC_DIR=$libxcDir"
        "-DSPGLIB_DIR=$spglibDir"
        "-DXML_LIB_DIR=$xmlLibDir"
        "-DXML_INCLUDE_DIR=$xmlIncludeDir"
        -DWITH_MDI=OFF
        -DMDI_PATH=
        -DWITH_DCCL=OFF
        -DWITH_TORCH=OFF
        "-DCMAKE_PREFIX_PATH=$ELPA_PATH;$DCCL_PATH;$TORCH_PATH"
        -DWITH_GPU=ON
        -DGPU_LANG=cuda
        -DGPU_VENDOR=nvidia
        -DWITH_GPU_AWARE_MPI=OFF
        "-DCMAKE_CUDA_FLAGS=$device_flags"
        "-DCMAKE_CUDA_ARCHITECTURES=$device_architectures"
        -DWITH_TESTING=ON
        "-DNUMDIFF_EXECUTABLE=$INST/numdiff_install/bin/numdiff"
        -DMINIMAL_COMPILE=OFF
        -DHIGHERQUAD_PSP=OFF
        -DUSE_64BIT_INT=OFF
    )

    cmake_real() {
        mkdir -p real && cd real
        cmake "${cmake_flags[@]}" \
            -DWITH_COMPLEX=OFF "-DDEAL_II_DIR=$INST/dealii" \
            "$1"
        make -j8
        cd ..
    }

    cmake_cplx() {
        mkdir -p complex && cd complex
        cmake "${cmake_flags[@]}" \
            -DWITH_COMPLEX=ON "-DDEAL_II_DIR=$INST/dealii" \
            "$1"
        make -j8
        cd ..
    }

    mkdir -p "$out"
    cd "$out"

    echo "Building Real executable in $build_type mode..."
    cmake_real "$SRC"

    #echo "Building Complex executable in $build_type mode..."
    #cmake_cplx "$SRC"

    echo 'Build complete.'
    cd "$WD"
    echo "===== [compile_dftfe] Completed: DFT-FE (branch $DFTFE_BRANCH) compiled ====="
}



compile_invDFT_invGKS() {
    echo "===== [compile_invDFT_invGKS] Compiling invDFT from $INVDFT_SRC ====="
    # Use the repo this script belongs to (no download); build in $WD/src/invDFT/build
    local SRC="$INVDFT_SRC"
    if [ ! -f "$SRC/CMakeLists.txt" ]; then
        echo "ERROR: no CMakeLists.txt in $SRC; set INVDFT_SRC to the invDFT source directory." >&2
        return 1
    fi
    mkdir -p "$WD/src/invDFT"
    mkdir -p "$WD/src/invDFT/build"
    cd "$WD/src/invDFT/build"
 
    local dealiiDir=$INST/dealii
    local dftfeRealDir=$WD/src/dftfe
    local dftfeIncludeDir=$WD/src/dftfe/include
    local alglibDir=$INST/lib/alglib
    local libxcDir=$INST
    local spglibDir=$INST
    local xmlIncludeDir=/usr/include/libxml2
    local xmlLibDir=/usr/lib64
 
    local ELPA_PATH=$INST
    local DCCL_PATH=$NCCL_DIR
    local TORCH_PATH=''
 
    # Compiler options and flags
    local cxx_compiler=CC
    local cxx_flags='-march=znver3 -fPIC -Wdeprecated-declarations -Wmissing-template-keyword -O2'
    local cxx_flagsRelease="-Wno-deprecated-declarations -Wmissing-template-keyword -fPIC -target-accel=nvidia80 -I${MPICH_DIR}/include -O2"
    local device_flags="-I${MPICH_DIR}/include -arch=sm_80"
    local device_architectures=80
 
    # HIGHERQUAD_PSP option compiles with default or higher order
    # quadrature for storing pseudopotential data
    # ON is recommended for MD simulations with hard pseudopotentials
 
    # build type: "Release" or "Debug"
    local build_type=Release
    local out
    out=$(echo "$build_type" | tr '[:upper:]' '[:lower:]')
 
    # Note: MDI_PATH is not used by project.
    local cmake_flags=(
        -DCMAKE_CXX_STANDARD=17
        "-DCMAKE_CXX_COMPILER=$cxx_compiler"
        "-DCMAKE_CXX_FLAGS=$cxx_flags"
        "-DCMAKE_CXX_FLAGS_RELEASE=$cxx_flagsRelease"
        "-DCMAKE_BUILD_TYPE=$build_type"
        "-DDEAL_II_DIR=$dealiiDir"
        "-DALGLIB_DIR=$alglibDir"
        "-DLIBXC_DIR=$libxcDir"
        "-DSPGLIB_DIR=$spglibDir"
        "-DXML_LIB_DIR=$xmlLibDir"
        "-DXML_INCLUDE_DIR=$xmlIncludeDir"
        "-DDFTFE_INSTALL_PATH=$dftfeRealDir"
        "-DDFTFE_INCLUDE_PATH=$dftfeIncludeDir"
        -DWITH_MDI=OFF
        -DMDI_PATH=
        -DWITH_DCCL=OFF
        -DWITH_TORCH=OFF
        "-DCMAKE_PREFIX_PATH=$ELPA_PATH;$DCCL_PATH;$TORCH_PATH"
        -DWITH_GPU=ON
        -DGPU_LANG=cuda
        -DGPU_VENDOR=nvidia
        -DWITH_GPU_AWARE_MPI=OFF
        "-DCMAKE_CUDA_FLAGS=$device_flags"
        "-DCMAKE_CUDA_ARCHITECTURES=$device_architectures"
    )
 
    cmake_real() {
        mkdir -p real && cd real
        cmake "${cmake_flags[@]}" \
            -DWITH_COMPLEX=OFF \
            "$1"
        make -j8
        cd ..
    }
 
    mkdir -p "$out"
    cd "$out"
 
    echo "Building Real executable in $build_type mode..."
    cmake_real "$SRC"
 
    echo 'Build complete.'
    cd "$WD"
    echo "===== [compile_invDFT_invGKS] Completed: invDFT compiled in $WD/src/invDFT/build ====="
}

install_all() {
    echo "===== [install_all] Starting full installation of invDFT and all dependencies ====="
    # Stops at the first step that fails (also when this file is sourced).
    install_blis &&
    install_libflame &&
    install_alglib &&
    install_libxc &&
    install_spglib &&
    install_p4est &&
    install_scalapack &&
    install_elpa &&
    install_kokkos &&
    install_petsc &&
    install_slepc &&
    install_numdiff &&
    install_dealii &&
    compile_dftfe &&
    compile_invDFT_invGKS &&
    echo "===== [install_all] Completed: invDFT and all dependencies installed ====="
}


# When run directly (not sourced), execute the functions named on the command line.
# Fail-fast is enabled only here, so that sourcing this file never makes
# your interactive shell exit on an error.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    if [[ ${#FUNCS[@]} -eq 0 ]]; then
        _usage
        exit 1
    fi
    for fn in "${FUNCS[@]}"; do
        if ! declare -F "$fn" > /dev/null; then
            echo "ERROR: unknown function: $fn" >&2
            exit 1
        fi
    done
    set -e
    for fn in "${FUNCS[@]}"; do
        "$fn"
    done
fi
