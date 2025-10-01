invDFT : A finite-element based C++ code to perform inverse DFT calculations
=======================================================

About
-----
invDFT is a massively parallel C++ code that can perform inverse DFT calculations. The code uses a complete finite-element basis providing a robust and efficient formulation to compute the exact v_{xc} potential for the input ground state electron density.
The code can run on CPU and GPU (NVidia and AMD) architectures and its efficiency and accuracy has been demonstrated for different molecules.

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

