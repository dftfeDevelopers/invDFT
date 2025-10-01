function compile_invDFT_1_0_pre_pgd {
  cd /path_to_invDFT/
  SRC=$PWD
  mkdir build_pgd
  cd build_pgd

  dealiiDir=$INST
  dftfeRealDir=/path_to_dftfe/dftfe/build/release/real
  dftfeIncludeDir=/path_ti_dftfe/dftfe/include
  alglibDir=$INST/lib/alglib
  libxcDir=$INST

  ELPA_PATH=$INST
  DCCL_PATH=$ROCM_PATH/include/rccl

  #Compiler options and flags
  cxx_compiler=CC
  cxx_flags="-march=znver3 -fPIC -I$MPICH_DIR/include -I$ROCM_PATH/include -I$ROCM_PATH/include/hip -I$ROCM_PATH/include/hipblas -I$ROCM_PATH/include/rocblas"
  cxx_flagsRelease=-O2 #sets DCMAKE_CXX_FLAGS_RELEASE
  device_flags="-march=znver3 -O2 -munsafe-fp-atomics -I$MPICH_DIR/include -I$ROCM_PATH/include -I$ROCM_PATH/include/hip -I$ROCM_PATH/include/hipblas -I$ROCM_PATH/include/rocblas"
  device_architectures=gfx90a

  # HIGHERQUAD_PSP option compiles with default or higher order
  # quadrature for storing pseudopotential data
  # ON is recommended for MD simulations with hard pseudopotentials

  # build type: "Release" or "Debug"
  build_type=Release
  out=`echo "$build_type" | tr '[:upper:]' '[:lower:]'`

  function cmake_real {
    mkdir -p real && cd real
    cmake -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_COMPILER=$cxx_compiler -DCMAKE_CXX_FLAGS="$cxx_flags" -DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" -DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiDir -DDFTFE_INSTALL_PATH=$dftfeRealDir -DDFTFE_INCLUDE_PATH=$dftfeIncludeDir -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir  -DWITH_DCCL=OFF -DCMAKE_PREFIX_PATH="$ELPA_PATH;$DCCL_PATH;$TORCH_PATH" -DWITH_GPU=ON -DGPU_LANG=hip -DGPU_VENDOR=amd -DWITH_GPU_AWARE_MPI=OFF -DCMAKE_HIP_FLAGS="$device_flags" -DCMAKE_HIP_ARCHITECTURES=$device_architectures -DCMAKE_SHARED_LINKER_FLAGS="-L$ROCM_PATH/lib -lamdhip64 -L$MPICH_DIR/lib -lmpi $CRAY_XPMEM_POST_LINK_OPTS -lxpmem $PE_MPICH_GTL_DIR_amd_gfx90a $PE_MPICH_GTL_LIBS_amd_gfx90a"  $1
    make -j16
    cd ..
  }

  mkdir -p $out
  cd $out

  echo Building Real executable in $build_type mode...
  cmake_real $SRC

  echo Build complete.
  cd /path_to_invDFT/
}
