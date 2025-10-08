invDFT : A finite-element based C++ code to perform inverse DFT calculations
=======================================================

About
-----
invDFT is a massively parallel C++ code that can perform inverse DFT calculations. The code uses a complete finite-element basis providing a robust and efficient formulation to compute the exact v_{xc} potential for a given input ground state electron density.
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

invDFT is built on DFT-FE from which it borrows efficient finite-element infrastructure and solvers. 

The steps to install the necessary dependencies and DFT-FE itself are described in the *Installation* section of the DFT-FE manual (download the development version manual [here](https://github.com/dftfeDevelopers/dftfe/blob/manual/manual-develop.pdf)). 

Several shell based installation scripts have been created for the development version of DFT-FE (`publicGithubDevelop` branch) on various machines:
  - [OLCF Frontier](https://github.com/dftfeDevelopers/install_DFTFE/tree/frontierDevelop)
  - [NERSC Perlmutter](https://github.com/dftfeDevelopers/install_DFTFE/tree/perlmutterDevelop)
  - [UMICH Greatlakes](https://github.com/dftfeDevelopers/install_DFTFE/tree/greatlakesDevelop) 

For the installation of invDFT please refer to *Installation* section of the *invDFT* manual (available [here](https://github.com/dftfeDevelopers/invDFT/manual/invDFTFEmanual_develop.pdf))

For convenience, sample installation scripts for invDFT are provided in the installationScripts folder.


Running invDFT
--------------

Instructions on how to run invDFT including demo examples can also be found in the *Running invDFT* section of the manual.

More information
----------------

For more information please contact the following, 

	- Vishal Subramanian (vishalsu@umich.edu)
	- Bikash Kanungo (bikash@umich.edu)
	- Vikram Gavini (vikramg@umich.edu) [Mentor]

License
-------

invDFT is published under [LGPL v2.1 or newer](https://github.com/dftfeDevelopers/invDFT/blob/main/LICENSE).

