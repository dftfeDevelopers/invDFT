invDFT : A finite-element based C++ code to perform inverse DFT calculations
=======================================================

About
-----
invDFT is a massively parallel C++ code that can perform inverse DFT calculations. In the invGKS branch, we have extended the capability of the code to perform inverse calculations under the Generalised Kohn Sham (GKS) framework with a fraction of exact exchange ($\alpha$). The code uses a complete finite-element basis providing a robust and efficient formulation to compute the exact ${v^{\alpha}_{\text{xc}}}$ potential for a given input ground state electron density.
The code can run on CPU and GPU (NVidia and AMD) architectures and its efficiency and accuracy has been demonstrated for different molecules.

Directory structure of invDFT
-----------------------------

 - src/ (Folder containing all the source files of invDFT)
  - gaussian/ ( Folder containg the files for reading the density matrix using a Gaussian atomic orbitals)
  - slater/   ( Folder containg the files for reading the density matrix using a Slater atomic orbitals)
  - InverseDFTEngine.cpp ( This class initialises the required FE infrasture and passes the relavent variables to InverseDFTSolverFunction for the iverse calculation)
  - InverseDFTSolverFunction.cpp ( This class performs the inverse calculations. This class computes the force for a given input of v_{xc} and passes it to the BFGS solver.)
  - BFGSInverseDFTSolver.cpp ( This class uses the BFGS algorithm to update the guess for v_{xc} based on the force vectors obtained from InverseDFTSolverFunction.) 
  - MultiVectorAdjointLinearSolverProblem.cpp ( This class provides the infrastructure for Adjoint problem that arises due to imposing the constraints in the Lagrangian.)
  - TriangulationManagerVxc.cpp ( This class creates the linear finite-element mesh on which v_{xc} is computed.) 
  - inverseDFTParameters.cpp ( Infrastructure to parse input parameters from the input parameter file.)
  - TestMultiVectorAdjointProblem.cpp ( A class that provides a functionality test for MultiVectorAdjointLinearSolverProblem .)
 - include/ (contains all the include files containing class and namespace declarations.)
 - installationScripts/ ( Provides installation scripts.) 
 - manual/ (Contains the manual for the invDFT.)
 - demo/ (Contains examples for running inverse DFT calculation.) 
 - indentationStandard / (contains scripts for automatic code indendation based on clang format)

Installation instructions
-------------------------

invDFT is built on top of [DFT-FE](https://github.com/dftfeDevelopers/dftfe), from which it borrows its finite-element infrastructure and solvers. Installing the `invGKS` branch of invDFT therefore takes two steps:

1. Install the dependencies and build the **`invGKS`** branch of DFT-FE.
2. Build the **`invGKS`** branch of invDFT against that DFT-FE installation.

> **Note:** The `invGKS` branch of invDFT only works with the `invGKS` branch of DFT-FE. Other DFT-FE branches (for example, `publicGithubDevelop`) are not compatible with it.

### Step 1: Install DFT-FE (`invGKS` branch)

DFT-FE depends on a number of external libraries, including deal.II, p4est, PETSc, SLEPc, Kokkos, ScaLAPACK, ELPA, Libxc, spglib, ALGLIB, and BLAS/LAPACK libraries (such as BLIS and libflame). It also needs an MPI-enabled C++ compiler and, for GPU runs, the CUDA (NVIDIA) or ROCm (AMD) toolchain. The steps to install DFT-FE and its dependencies are described in the *Installation* section of the DFT-FE manual. Download the development version of the manual [here](https://github.com/dftfeDevelopers/dftfe/blob/manual/manual-develop.pdf).

Once the dependencies are in place, check out the `invGKS` branch of DFT-FE before compiling:

```
git clone https://github.com/dftfeDevelopers/dftfe.git
cd dftfe
git checkout invGKS
```

**Installation scripts for common machines.** To simplify the process, shell-based installation scripts for the development version of DFT-FE are available for several supercomputers:

- [OLCF Frontier](https://github.com/dftfeDevelopers/install_DFTFE/tree/frontierDevelop)
- [NERSC Perlmutter](https://github.com/dftfeDevelopers/install_DFTFE/tree/perlmutterDevelop)
- [UMich Great Lakes](https://github.com/dftfeDevelopers/install_DFTFE/tree/greatlakesDevelop)

These scripts are written for the `publicGithubDevelop` branch of DFT-FE. Before running them, change the DFT-FE branch that they check out to `invGKS`, so that the compiled DFT-FE is the one invDFT expects.

### Step 2: Install invDFT (`invGKS` branch)

With DFT-FE built, follow the *Installation* section of the invDFT manual, available [here](https://github.com/dftfeDevelopers/invDFT/blob/main/manual/invDFTFEmanual_develop.pdf). Fetch the `invGKS` branch of invDFT with:

```
git clone https://github.com/dftfeDevelopers/invDFT.git
cd invDFT
git checkout invGKS
```

Sample installation scripts for invDFT are provided in the [`installationScripts`](https://github.com/dftfeDevelopers/invDFT/tree/main/installationScripts) folder of the repository and can be adapted to your machine.

### One-step installation on NERSC Perlmutter

If you are working on NERSC Perlmutter, you can use [this script](https://github.com/dftfeDevelopers/invDFT/blob/invGKS/install_invDFT_withDependencies_inPerlmutter/installInvDFT.sh), which installs the `invGKS` branch of invDFT together with all of its dependencies. By default, everything is installed in `$PSCRATCH/install_invDFT`. To use a different location, change the value of `WD` in the script. To install, run:

```
source installInvDFT.sh
install_all
```

If the installation is successful, the `invDFT_exe` executable is created in `$WD/src/invDFT/build/release/real/`.

The script can also serve as a template for other machines. The same sequence of builds is needed, and only the module names, compilers, and paths change.



Running invDFT
-------------------------

Instructions for running invDFT, including demo examples, can also be found in the *Running invDFT* section of the manual. For a more detailed explanation of all the parameters involved in the calculation, we refer readers to the [invDFT paper](https://www.sciencedirect.com/science/article/pii/S0010465526002006).

The `demo` folder contains two examples (`invGKS_LiH_eq` and `invGKS_LiH_2eq`) of inversion under the generalized Kohn-Sham (GKS) framework. For `invGKS_LiH_eq`, we provide the explanation for the input files used to compute the exchange-correlation (XC) potential of the LiH molecule at equilibrium bond distance with 25% (0.25) exact exchange.

### Input files

Two main input files drive the calculation:

- `allElectronParameterFile.prm` contains the parameters used to construct the finite-element (FE) mesh for the wavefunctions, the boundary conditions, and so on. Broadly, it holds all the parameters needed for the forward SCF calculation.
- `inverseDFTParams.prm` contains the parameters used to generate the FE mesh for the XC potential $v_{\text{xc}}^{\alpha}$, the solver tolerances, the initial guess, and the input densities.

### `allElectronParameterFile.prm`

#### General settings

```
set VERBOSITY = 4
set SOLVER MODE = GS
set USE GPU        = true
```

`VERBOSITY` controls how much information is written to the output file. It is set to 4 so that the eigenvalues are printed at the end of each eigensolve. `SOLVER MODE` is set to `GS` to perform a ground-state calculation. `USE GPU` is set to `true` to run the computation on GPUs; set it to `false` if the calculation is run on CPUs.

#### Geometry

```
subsection Geometry
  set NATOMS=2
  set NATOM TYPES=2
  set ATOMIC COORDINATES FILE      = coordinates.inp 
  set DOMAIN VECTORS FILE = domainVectors.inp
end
```

This subsection defines the geometry of the molecule. `NATOMS` is the number of atoms, which is 2 for LiH. `NATOM TYPES` is the number of distinct species; LiH contains `Li` and `H`, so it is also 2. The file `coordinates.inp` lists the atomic coordinates, one atom per line, in the format:

```
atomic_number  valence_number  x_coord  y_coord  z_coord
```

For all-electron calculations, the valence number equals the atomic number. The file `domainVectors.inp` specifies the size of the simulation domain.

#### Boundary conditions

```
subsection Boundary conditions
  set PERIODIC1                       = false
  set PERIODIC2                       = false
  set PERIODIC3                       = false
end
```

This subsection sets the boundary conditions. Since LiH is a molecule, non-periodic boundary conditions are used in all three directions.

#### Finite-element mesh

```
subsection Finite element mesh parameters
  set POLYNOMIAL ORDER=5
  set POLYNOMIAL ORDER ELECTROSTATICS = 7
  subsection Auto mesh generation parameters
    set AUTO ADAPT BASE MESH SIZE = true
    set MESH SIZE AT ATOM = 0.06
    set MESH SIZE AROUND ATOM = 0.45
    set ATOM BALL RADIUS=8.0
    set INNER ATOM BALL RADIUS = 0.5
  end
end
```

This subsection sets the FE mesh on which the wavefunctions are computed.

- `POLYNOMIAL ORDER` is the order of the FE basis polynomials. For inverse calculations, we find an order between 4 and 6 to be optimal.
- `POLYNOMIAL ORDER ELECTROSTATICS` is the polynomial order used to solve the electrostatics problem. We find `POLYNOMIAL ORDER + 2` to be optimal.
- `AUTO ADAPT BASE MESH SIZE` automatically chooses the base mesh size so that the most refined elements have a size close to `MESH SIZE AT ATOM`.
- `MESH SIZE AT ATOM` is the mesh size at the atom, and therefore the smallest mesh size in the calculation.
- `INNER ATOM BALL RADIUS` is the radius around each atom within which the mesh size stays at `MESH SIZE AT ATOM`. Beyond this radius, the mesh coarsens until it reaches `MESH SIZE AROUND ATOM`, and it keeps that size out to `ATOM BALL RADIUS`.

#### Exchange-correlation and exact exchange

```
subsection DFT functional parameters
  set EXCHANGE CORRELATION TYPE   = LDA-PW
  set PSEUDOPOTENTIAL CALCULATION = false
  set TOL EXACT EXCHANGE ENERGY = 1e-9
  set FRACTION OF EXACT EXCHANGE IN FORWARD CALC = 1.0 
  set FRACTION OF EXACT EXCHANGE IN INV CALC = 0.25
  set Hartree Fock Calculation IN INV CALC = true
  set Hartree Fock Calculation in Forward Solve = true
  set Add Correlation Potential to HF Calculation = false
end
```

Hybrid calculations use a nested SCF procedure based on the Adaptively Compressed Exchange (ACE) method. In the first outer SCF iteration, the ground state is computed using the functional given by `EXCHANGE CORRELATION TYPE`, with no exact exchange. The resulting wavefunctions are used to construct the ACE operator, and the outer SCF iterations continue until the change in exact exchange energy between successive outer iterations falls below `TOL EXACT EXCHANGE ENERGY`.

The remaining parameters are as follows:

- `PSEUDOPOTENTIAL CALCULATION` is set to `false` because this is an all-electron calculation.
- `FRACTION OF EXACT EXCHANGE IN FORWARD CALC` is set to 1.0 and `Add Correlation Potential to HF Calculation` to `false`. Together these make the forward calculation a Hartree-Fock calculation, which provides the ground-state density and wavefunctions used as inputs to the inverse calculation.
- `FRACTION OF EXACT EXCHANGE IN INV CALC` is the fraction of exact exchange used in the inverse calculation (0.25 here).
- `Hartree Fock Calculation in Forward Solve` and `Hartree Fock Calculation IN INV CALC` are set to `true` to enable exact exchange in the forward and inverse calculations, respectively.

#### SCF and eigensolver

```
subsection SCF parameters
  set MIXING PARAMETER =0.2
  set COMPUTE ENERGY EACH ITER = false
  set MIXING METHOD= ANDERSON
  set STARTING WFC             = RANDOM
  set TEMPERATURE              = 10
  set TOLERANCE                = 1e-8
  subsection Eigen-solver parameters
    set NUMBER OF KOHN-SHAM WAVEFUNCTIONS                    = 10
    set CHEBYSHEV FILTER TOLERANCE = 1e-4
    set ORTHOGONALIZATION TYPE=CGS
  end
end
```

This subsection specifies the parameters of the forward calculation.

- `MIXING METHOD` is set to `ANDERSON`, with a `MIXING PARAMETER` of 0.2.
- `STARTING WFC` is set to `RANDOM`, so the starting wavefunctions are initialized randomly.
- `TEMPERATURE` is the temperature (in Kelvin) used in the Fermi-Dirac smearing of the electron occupations.
- `TOLERANCE` is the convergence tolerance of the SCF iterations.
- `COMPUTE ENERGY EACH ITER` is set to `false` to skip computing the energy at every SCF iteration.
- `NUMBER OF KOHN-SHAM WAVEFUNCTIONS` is the number of wavefunctions computed by the eigensolver. It is larger than the number of occupied states to allow for smearing and unoccupied states.
- `CHEBYSHEV FILTER TOLERANCE` is the tolerance used by the Chebyshev filtered subspace iteration eigensolver.
- `ORTHOGONALIZATION TYPE` is set to `CGS` (classical Gram-Schmidt).

### `inverseDFTParams.prm`

This file contains the parameters used in the inverse calculation.

#### BFGS solver

```
   set TOL FOR BFGS = 1e-12
   set BFGS LINE SEARCH = 1
   set TOL FOR BFGS LINE SEARCH = 1e-6
   set BFGS HISTORY = 10
```

These parameters configure the BFGS optimizer: its convergence tolerance, the line-search option and its tolerance, and the number of previous steps (history) stored.

#### Output

```
   set WRITE VXC DATA = true
   set FOLDER FOR VXC DATA = vxcDataOut
   set POSTFIX TO THE FILENAME FOR WRITING VXC DATA = vxcData_mesh0p06_ball6p0_temp10_1
   set FREQUENCY FOR WRITING VXC = 5
```

These parameters control the output of the XC potential. `WRITE VXC DATA` turns the output on, `FOLDER FOR VXC DATA` is the folder where the files are written, and `POSTFIX TO THE FILENAME FOR WRITING VXC DATA` is a label appended to the file names. `FREQUENCY FOR WRITING VXC` sets how often (in BFGS iterations) the potential is written.

#### Mesh for the XC potential

```
   set RHO TOL FOR CONSTRAINTS = 1e-6
   set VXC MESH DOMAIN SIZE = 8.0
   set VXC MESH SIZE NEAR ATOM = 0.06
```

These parameters are used to construct the linear FE mesh on which $v_{\text{xc}}^{\alpha}$ is solved. `VXC MESH SIZE NEAR ATOM` is the mesh size near the atoms. It is kept fixed out to a distance of `VXC MESH DOMAIN SIZE` along all directions, beyond which the mesh coarsens. `RHO TOL FOR CONSTRAINTS` is the density tolerance used when imposing the constraints in the inverse problem.

#### Loss function and initial guess

```
   set INITIAL TOL FOR CHEBYSHEV FILTERING = 1e-7
   set ALPHA1 FOR WEIGHTS FOR LOSS FUNCTION = 0.0
   set ALPHA2 FOR WEIGHTS FOR LOSS FUNCTION = 0.0
   set TAU FOR WEIGHTS FOR LOSS FUNCTION = 1e-5
   set TAU FOR WEIGHTS FOR SETTING VX BC = 1e-8
   set TAU FOR WEIGHTS FOR SETTING FABC = 1e-3
   set SET FERMIAMALDI IN THE FAR FIELD AS INPUT = true
```

These parameters define the loss function and the initial guess for the potential. With `SET FERMIAMALDI IN THE FAR FIELD AS INPUT` set to `true`, the Fermi-Amaldi potential is used in the far field. See the [invDFT paper](https://www.sciencedirect.com/science/article/pii/S0010465526002006) for the meaning of the weight parameters.

#### Input density

```
   set READ GAUSSIAN DATA AS INPUT = true
   set READ SLATER DATA AS INPUT = false
   set USE DELTA RHO CORRECTION                      = true
   set GAUSSIAN DENSITY FOR PRIMARY RHO SPIN UP = DensityMatrix
   set GAUSSIAN DENSITY FOR DFT RHO SPIN UP = DensityMat_LiH_HF
   set ATOMIC ORBITAL ATOMIC COORD FILE = AtomicCoords
   set GAUSSIAN S MATRIX FILE = SMatrix
```

These parameters provide the Gaussian-basis data for the density that is inverted. `READ GAUSSIAN DATA AS INPUT` is `true` and `READ SLATER DATA AS INPUT` is `false`, so the target density is read from Gaussian-basis data rather than Slater-basis data. The remaining entries name the files containing the density matrices, the atomic orbital coordinates, and the overlap (S) matrix.


More information
----------------

For more information please contact the following, 

	- Vishal Subramanian (vishalsu@umich.edu)
	- Bikash Kanungo (bikash@umich.edu)
	- Vikram Gavini (vikramg@umich.edu) [Mentor]

License
-------

invDFT is published under [LGPL v2.1 or newer](https://github.com/dftfeDevelopers/invDFT/blob/main/LICENSE).

