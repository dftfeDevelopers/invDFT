# Demo examples

There are a few demo examples provided in this folder. Each example is a separate sub-folder with all the necessary input files. Below we describe how to run invDFT for the demo examples and what each input file means. Each example is provided with an `out_reference` for reproducibility.

## How to run

```bash
srun -n <np> ./invDFT_exe dftfeParams.prm inverseDFTParams.prm &> out
```

We recommend the following computing resources for the demo examples. These are ballpark suggestions to ensure sufficient memory and reasonable walltimes.

| Example   | GPUs | Walltime |
|-----------|-----:|---------:|
| `LiH_eq`  | 96   | 12 hours |
| `LiH_2eq` | 120  | 20 hours |

## Input files description

This folder contains the inputs provided to the invDFT code and the corresponding outputs obtained from invDFT.

### `dftfeParams.prm`

This is the parameter file that provides the input parameters required by the DFT-FE library. This file sets the finite-element mesh and the Kohn-Sham eigenvalue solver parameters that are eventually used in inverse DFT. For the list of all available parameters please refer to the [DFT-FE manual](https://github.com/dftfeDevelopers/dftfe/blob/manual/manual-develop.pdf).

> [!NOTE]
> To use a fraction ($\alpha$) of exact exchange in inverse DFT, the following has to be set in the `dftfeParams.prm` under the "DFT functional parameters" subsection:
>
> ```text
> set FRACTION OF EXACT EXCHANGE IN INV CALC = 0.25
> set Hartree Fock Calculation IN INV CALC = true
> ```

### `inverseDFTParams.prm`

The parameter file that provides the input parameters required by invDFT.

### `coordinates.inp`

File specifying the coordinates of the molecule provided in the DFT-FE format. The units are in bohr. This file needs to be specified inside the `dftfeParams.prm` under the "Geometry" subsection using:

```text
set ATOMIC COORDINATES FILE = coordinates.inp
```

See the [DFT-FE manual](https://github.com/dftfeDevelopers/dftfe/blob/manual/manual-develop.pdf) for details.

### `domainVectors.inp`

The file that specifies the extent of the simulation domain. The units are in bohr. This file needs to be specified inside the `dftfeParams.prm` under the "Geometry" subsection using:

```text
set DOMAIN VECTORS FILE = domainVectors.inp
```

### `DensityMatrix`

This file provides the target one-electron reduced density matrix (1RDM) in the atomic orbital (AO) basis. This is specified inside the `inverseDFTParams.prm` file. Only Gaussian and Slater AOs are supported.

> [!NOTE]
> The 1RDM should be normalized to $N_e/2$ (where $N_e$ is the number of electrons and is even for closed-shell). This is done to uniformly handle spin-restricted and spin-unrestricted calculations (although the spin-unrestricted case is not yet supported).

### `DensityMatrixSecondary`

The 1RDM corresponding to a self-consistent DFT or Hartree-Fock calculation using the same AO basis used to generate the target 1RDM (`DensityMatrix` above). Support is provided for only LDA (Perdew-Wang 1992, Perdew-Zunger 1981 functionals), GGA (PBE 1996), and HF (with full exact exchange) in the same atomic orbital basis. This is required for the $\Delta\rho$ cusp correction.

> [!NOTE]
> To use HF, the following has to be set in the `dftfeParams.prm` under the "DFT functional parameters" subsection:
>
> ```text
> set FRACTION OF EXACT EXCHANGE IN FORWARD CALC = 1.0
> set Hartree Fock Calculation in Forward Solve = true
> ```

### `AtomicCoords`

The coordinates of the molecule provided in the standard quantum chemistry format. Each line contains the symbol of the element followed by the x, y, z coordinates in Angstroms. The last entry specifies the file name of the AO basis for that element. The order of the atoms should be the same as that of `coordinates.inp`.

> [!NOTE]
> This file assumes the unit to be Angstrom whereas `coordinates.inp` assumes it to be bohr.

### `H_gaussian`, `Li_gaussian`, etc

The atomic basis files for the different elements. These are the file names provided in the `AtomicCoords` file above. The Gaussian AOs should be in QChem format and the Slater AOs should be in the ADF format. The file name `H_gaussian`, `Li_gaussian` are just for illustration, the actual name may change.

### `SMatrix`

The overlap of the AO basis.
