This folder contains the inputs provided to invDFT code and the corresponding outputs obtained from invDFT. 

- allElectronParameterFile.prm: This is the parameter file that provides the input parameters required by dftfe library. For the list of all available parameters please refer to the dftfe manual. (https://github.com/dftfeDevelopers/dftfe/blob/manual/manual-develop.pdf).

- coordinates.inp: coordinate file specifying the coordinates of the molecule ( The units are in Bohrs).

- domainVectors.inp: The file that specifies the extent of the domain.

- inverseDFTParams.prm: The parameter file that provides the input parameters required by invDFT. 

- DensityMatrix:  The file that provides the input density that needs to be inverted. For a density from an atomic orbital basis (Gaussian/Slater) the Denisty matrix is provided along with the basis file. 

- H_gaussian_paulzim, Li_gaussian_paulzim: The atomic basis files for the different elements present. 

- AtomicCoords: The coordinates of the molecule provided to the atomic orbital code. The units are in Angstroms. The order of the atoms should be same as that of coordinates.inp.

- SMatrix: The file that provides the overlap of the atomic orbitals.

- DensityMatrixSecondary: The density correspnding to a LDA calculation in the same atomic orbital basis. This is required for the delta rho correction. 

- vxcData_alpha1_680_output_vtuOutput_00*: The final output obtained after running the invDFT code and post processing it.


