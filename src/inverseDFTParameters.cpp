// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------
//
// @author Vishal Subramanian, Bikash Kanungo
//

#include "inverseDFTParameters.h"

namespace invDFT {

namespace internalInvDFTParams {
void declare_parameters(dealii::ParameterHandler &prm) {
    prm.declare_entry(
            "SOLVER MODE",
            "INVERSE",
            dealii::Patterns::Selection(
                    "INVERSE|POST_PROCESS|FUNCTIONAL_TEST"),
            "[Standard] invDFT SOLVER MODE: If GS: performs inverse DFT calculation. If POST_PROCESS: interpolates the Vxc from an input file to a set of points. If FUNCTIONAL_TEST: run a functional test on the adjoint solve");

    prm.enter_subsection("POST PROCESS");
    {
        prm.declare_entry("READS POINTS FROM FILE", "false", dealii::Patterns::Bool(),
                          "[Standard] if the the points to which the Vxc has to be interpolated should be read from file");

        prm.declare_entry("FILENAME FOR POINTS", ".",
                          dealii::Patterns::Anything(),
                          "[Standard] Name of the file from which the points are to be read");

        prm.declare_entry("FILENAME FOR OUTPUT", ".",
                          dealii::Patterns::Anything(),
                          "[Standard] Name of the file to which the interpolated values are written");

        prm.declare_entry("STARTING X", "-2.0",
                          dealii::Patterns::Double(),
                          "[Standard] Starting point for X axis");

        prm.declare_entry("STARTING Y", "-2.0",
                          dealii::Patterns::Double(),
                          "[Standard] Starting point for Y axis");

        prm.declare_entry("STARTING Z", "-2.0",
                          dealii::Patterns::Double(),
                          "[Standard] Starting point for Z axis");

        prm.declare_entry("ENDING X", "2.0",
                          dealii::Patterns::Double(),
                          "[Standard] Ending point for X axis");

        prm.declare_entry("ENDING Y", "2.0",
                          dealii::Patterns::Double(),
                          "[Standard] Ending point for Y axis");

        prm.declare_entry("ENDING Z", "2.0",
                          dealii::Patterns::Double(),
                          "[Standard] Ending point for Z axis");

        prm.declare_entry("NUMBER OF POINTS ALONG X DIRECTION", "1000",
                          dealii::Patterns::Integer(1, 100000),
                          "[Standard] Number of points along x direction.");

        prm.declare_entry("NUMBER OF POINTS ALONG Y DIRECTION", "1000",
                          dealii::Patterns::Integer(1, 100000),
                          "[Standard] Number of points along y direction.");

        prm.declare_entry("NUMBER OF POINTS ALONG Z DIRECTION", "1000",
                          dealii::Patterns::Integer(1, 100000),
                          "[Standard] Number of points along z direction.");

    }
    prm.leave_subsection();
    prm.enter_subsection("Inverse DFT parameters");
  {
    prm.declare_entry("TOL FOR BFGS", "1e-12", dealii::Patterns::Double(0.0),
                      "[Standard] tol for the BFGS solver convergence");

    prm.declare_entry("BFGS LINE SEARCH", "1", dealii::Patterns::Integer(0, 20),
                      "[Standard] Number of times line search is performed "
                      "before finding the optimal lambda.");

    prm.declare_entry("TOL FOR BFGS LINE SEARCH", "1e-6",
                      dealii::Patterns::Double(0.0),
                      "[Standard] tol for the BFGS solver line search");

    prm.declare_entry("BFGS HISTORY", "100", dealii::Patterns::Integer(1, 1000),
                      "[Standard] Number of times line search is performed "
                      "before finding the optimal lambda.");

    prm.declare_entry("BFGS MAX ITERATIONS", "10000",
                      dealii::Patterns::Integer(1, 100000),
                      "[Standard] Max number of iterations in BFGS.");

    prm.declare_entry("READ VXC DATA", "true", dealii::Patterns::Bool(),
                      "[Standard] Flag to determine if the initial Vxc is read "
                      "from a file or not");

    prm.declare_entry("POSTFIX TO THE FILENAME FOR READING VXC DATA", ".",
                      dealii::Patterns::Anything(),
                      "[Standard] Post fix added to the filenames from which "
                      "the vxc data is read");
    prm.declare_entry("WRITE VXC DATA", "true", dealii::Patterns::Bool(),
                      "[Standard] Write Vxc data so that it can be read later");

    prm.declare_entry(
        "FOLDER FOR VXC DATA", ".", dealii::Patterns::Anything(),
        "[Standard] Folder into which the Vxc data is written or read");

    prm.declare_entry("POSTFIX TO THE FILENAME FOR WRITING VXC DATA", ".",
                      dealii::Patterns::Anything(),
                      "[Standard] Post fix added to the filenames in which the "
                      "vxc data is written");

    prm.declare_entry(
        "FREQUENCY FOR WRITING VXC", "20", dealii::Patterns::Integer(1, 2000),
        "[Standard] Frequency with which the Vxc data is written to the disk");

    prm.declare_entry("INITIAL TOL FOR CHEBYSHEV FILTERING", "1e-6",
		    dealii::Patterns::Double(0.0),
                    "[Standard] The tolerance to which the chebyshev filtering is solved to initially. The tolerance is progressively made tighteer as the loss decreases.");
    prm.declare_entry("RHO TOL FOR CONSTRAINTS", "1e-6",
                      dealii::Patterns::Double(0.0),
                      "[Standard] The tol for rho less than which the initial "
                      "guess of Vxc is not updated");
    prm.declare_entry("VXC MESH DOMAIN SIZE", "6.0",
                      dealii::Patterns::Double(0.0),
                      "[Standard] The distance of the bounding box from the "
                      "atoms in which the Vxc mesh is refined");

    prm.declare_entry("VXC MESH SIZE NEAR ATOM", "0.3",
                      dealii::Patterns::Double(0.0),
                      "[Standard] The mesh size near atom for the Vxc mesh");

    prm.declare_entry("INITIAL TOL FOR ADJOINT PROBLEM", "1e-11",
                      dealii::Patterns::Double(0.0),
                      "[Standard] The initial tol to which the adjoint problem "
                      "is solved. This tol is adaptively reduced as the "
                      "iterations proceed based on the loss.");

    prm.declare_entry("MAX ITERATIONS FOR ADJOINT PROBLEM", "5000",
                      dealii::Patterns::Integer(10, 10000),
                      "[Standard] The maximum number of iterations allowed in "
                      "MinRes while solving the adjoint problem.");

    prm.declare_entry("ALPHA1 FOR WEIGHTS FOR LOSS FUNCTION", "0.0",
                      dealii::Patterns::Double(0.0),
                      "[Standard] The parameter used for weight assigned to "
                      "loss. The weight at a point is assigned based on  "
                      "\frac{1}{\rho^{\alpha1} + \tau} + \rho^{\alpha2}");

    prm.declare_entry("ALPHA2 FOR WEIGHTS FOR LOSS FUNCTION", "0.0",
                      dealii::Patterns::Double(0.0),
                      "[Standard] The parameter used for weight assigned to "
                      "loss. The weight at a point is assigned based on  "
                      "\frac{1}{\rho^{\alpha} + \tau} + \rho^{\alpha2}");

    prm.declare_entry(
        "TAU FOR WEIGHTS FOR LOSS FUNCTION", "1e-2",
        dealii::Patterns::Double(0.0),
        "[Standard] The parameter used for weight assigned to loss. The weight "
        "at a point is assigned based on  \frac{1}{\rho^{\alpha} + \tau}.");
    prm.declare_entry(
        "TAU FOR WEIGHTS FOR SETTING VX BC", "1e-2",
        dealii::Patterns::Double(0.0),
        "[Standard] The parameter used for weight assigned for Vx. The weight "
        "at a point is assigned based on  \frac{\rho}{\rho + \tau}.");
    prm.declare_entry(
        "TAU FOR WEIGHTS FOR SETTING FABC", "1e-2",
        dealii::Patterns::Double(0.0),
        "[Standard] The parameter used for weight assigned to set FA BC. The "
        "weight at a point is assigned based on  \frac{\rho}{\rho+ \tau}. This "
        "parameter is used to transition from Vxc to Vfa in the far field ");

    prm.declare_entry("TOL FOR FRACTIONAL OCCUPANCY", "1e-8",
                      dealii::Patterns::Double(0.0),
                      "[STANDARD] tol for checking fractional occupancy");

    prm.declare_entry("TOL FOR DEGENERACY", "0.002",
                      dealii::Patterns::Double(0.0),
                      "[STANDARD] tol for checking fractional occupancy");

    prm.declare_entry("READ GAUSSIAN DATA AS INPUT", "true",
                      dealii::Patterns::Bool(),
                      "[Standard] Flag to determine if the initial Vxc is read "
                      "from a file which is written in gaussian format");
    prm.declare_entry("SET FERMIAMALDI IN THE FAR FIELD AS INPUT", "true",
                      dealii::Patterns::Bool(),
                      "[Standard] Flag to determine if the initial Vxc has "
                      "fermi-amaldi as the far field in the input");

    prm.declare_entry(
        "GAUSSIAN DENSITY FOR PRIMARY RHO SPIN UP", ".",
        dealii::Patterns::Anything(),
        "[Standard] File name containing the density matrix obtained from the "
        "gaussian code. This is the density for which the Vxc is computed. In "
        "case of spin un polarised, provide half the total density ");

    prm.declare_entry(
        "GAUSSIAN DENSITY FOR PRIMARY RHO SPIN DOWN", ".",
        dealii::Patterns::Anything(),
        "[Standard] File name containing the density matrix obtained from the "
        "gaussian code. This is the density for which the Vxc is computed. In "
        "case of spin un polarised, this is not used");

    prm.declare_entry("GAUSSIAN DENSITY FOR DFT RHO SPIN UP", ".",
                      dealii::Patterns::Anything(),
                      "[Standard] File name containing the density matrix "
                      "obtained from the gaussian code. This density is used "
                      "for computing the delta rho correction. In case of spin "
                      "un polarised, provide half the total density ");

    prm.declare_entry(
        "GAUSSIAN DENSITY FOR DFT RHO SPIN DOWN", ".",
        dealii::Patterns::Anything(),
        "[Standard] File name containing the density matrix obtained from the "
        "gaussian code. This density is used for computing the delta rho "
        "correction. In case of spin un polarised, pthis file is not used ");

    prm.declare_entry(
        "GAUSSIAN ATOMIC COORD FILE", ".", dealii::Patterns::Anything(),
        "[Standard] File name containing the coordinates of the atoms. These "
        "coordinates will be used by the Gaussian code. This has to compatible "
        "with the input coordinates file");

    prm.declare_entry("GAUSSIAN S MATRIX FILE", ".",
                      dealii::Patterns::Anything(),
                      "[Standard] File containing the S matrix");
  }
  prm.leave_subsection();
}

} // namespace internalInvDFTParams

inverseDFTParameters::inverseDFTParameters() {

    // parameters for post process
    readPointsFromFile = false;
     fileNameReadPoints = "";
    fileNameWriteVxcPostProcess = "output_file";
    startX = -2.0;
    startY = -2.0;
    startZ = -2.0;
    endX = 2.0;
    endY = 2.0;
    endZ = 2.0;
    numPointsX = 100;
    numPointsY = 100;
    numPointsZ = 100;
  // Parameters for inverse problem
  inverseBFGSTol = 1e-12;
  inverseBFGSLineSearch = 1;
  inverseBFGSLineSearchTol = 1e-6;
  inverseBFGSHistory = 100;
  inverseMaxBFGSIter = 10000;
  writeVxcData = true;
  readVxcData = true;
  fileNameReadVxcPostFix = ".";
  vxcDataFolder = ".";
  fileNameWriteVxcPostFix = ".";
  writeVxcFrequency = 20;

  initialTolForChebFiltering = 1e-6;
  rhoTolForConstraints = 1e-6;
  VxcInnerDomain = 6.0;
  VxcInnerMeshSize = 0.0;
  inverseAdjointInitialTol = 1e-11;
  inverseAdjointMaxIterations = 5000;
  inverseAlpha1ForWeights = 0.0;
  inverseAlpha2ForWeights = 0.0;
  inverseTauForSmoothening = 1e-2;
  inverseTauForVxBc = 1e-2;
  inverseTauForFABC = 1e-2;
  inverseFractionOccTol = 1e-8;
  inverseDegeneracyTol = 0.002;

  readGaussian = false;
  fermiAmaldiBC = false;
  densityMatPrimaryFileNameSpinUp = ".";
  densityMatPrimaryFileNameSpinDown = ".";
  gaussianAtomicCoord = ".";
  sMatrixName = ".";
  densityMatDFTFileNameSpinUp = ".";
  densityMatDFTFileNameSpinDown = ".";
}

void inverseDFTParameters::parse_parameters(const std::string &parameter_file,
                                            const MPI_Comm &mpi_comm_parent,
                                            const bool printParams) {
  dealii::ParameterHandler prm;
  internalInvDFTParams::declare_parameters(prm);
  prm.parse_input(parameter_file);

    solvermode       = prm.get("SOLVER MODE");
    prm.enter_subsection("POST PROCESS");
    {
        readPointsFromFile = prm.get_bool("READS POINTS FROM FILE");
        fileNameReadPoints = prm.get("FILENAME FOR POINTS");
        fileNameWriteVxcPostProcess = prm.get("FILENAME FOR OUTPUT");
        startX = prm.get_double("STARTING X");
        startY = prm.get_double("STARTING Y");
        startZ = prm.get_double("STARTING Z");
        endX = prm.get_double("ENDING X");
        endY = prm.get_double("ENDING Y");
        endZ = prm.get_double("ENDING Z");
        numPointsX = prm.get_integer("NUMBER OF POINTS ALONG X DIRECTION");
        numPointsY = prm.get_integer("NUMBER OF POINTS ALONG Y DIRECTION");
        numPointsZ = prm.get_integer("NUMBER OF POINTS ALONG Z DIRECTION");
    }
    prm.leave_subsection();
  prm.enter_subsection("Inverse DFT parameters");
  {
    inverseBFGSTol = prm.get_double("TOL FOR BFGS");
    inverseBFGSLineSearch = prm.get_integer("BFGS LINE SEARCH");
    inverseBFGSLineSearchTol = prm.get_double("TOL FOR BFGS LINE SEARCH");
    inverseBFGSHistory = prm.get_integer("BFGS HISTORY");
    inverseMaxBFGSIter = prm.get_integer("BFGS MAX ITERATIONS");
    readVxcData = prm.get_bool("READ VXC DATA");
    fileNameReadVxcPostFix =
        prm.get("POSTFIX TO THE FILENAME FOR READING VXC DATA");
    writeVxcData = prm.get_bool("WRITE VXC DATA");
    vxcDataFolder = prm.get("FOLDER FOR VXC DATA");
    fileNameWriteVxcPostFix =
        prm.get("POSTFIX TO THE FILENAME FOR WRITING VXC DATA");
    writeVxcFrequency = prm.get_integer("FREQUENCY FOR WRITING VXC");

    rhoTolForConstraints = prm.get_double("RHO TOL FOR CONSTRAINTS");
    VxcInnerDomain = prm.get_double("VXC MESH DOMAIN SIZE");
    VxcInnerMeshSize = prm.get_double("VXC MESH SIZE NEAR ATOM");
    initialTolForChebFiltering = prm.get_double("INITIAL TOL FOR CHEBYSHEV FILTERING");
    inverseAdjointInitialTol =
        prm.get_double("INITIAL TOL FOR ADJOINT PROBLEM");
    inverseAdjointMaxIterations =
        prm.get_integer("MAX ITERATIONS FOR ADJOINT PROBLEM");
    inverseAdjointInitialTol =
        prm.get_double("INITIAL TOL FOR ADJOINT PROBLEM");
    inverseAdjointInitialTol =
        prm.get_double("INITIAL TOL FOR ADJOINT PROBLEM");
    inverseAlpha1ForWeights =
        prm.get_double("ALPHA1 FOR WEIGHTS FOR LOSS FUNCTION");
    inverseAlpha2ForWeights =
        prm.get_double("ALPHA2 FOR WEIGHTS FOR LOSS FUNCTION");
    inverseTauForSmoothening =
        prm.get_double("TAU FOR WEIGHTS FOR LOSS FUNCTION");
    inverseTauForVxBc = prm.get_double("TAU FOR WEIGHTS FOR SETTING VX BC");
    inverseTauForFABC = prm.get_double("TAU FOR WEIGHTS FOR SETTING FABC");
    inverseFractionOccTol = prm.get_double("TOL FOR FRACTIONAL OCCUPANCY");
    inverseDegeneracyTol = prm.get_double("TOL FOR DEGENERACY");
    readGaussian = prm.get_bool("READ GAUSSIAN DATA AS INPUT");
    fermiAmaldiBC = prm.get_bool("SET FERMIAMALDI IN THE FAR FIELD AS INPUT");
    densityMatPrimaryFileNameSpinUp =
        prm.get("GAUSSIAN DENSITY FOR PRIMARY RHO SPIN UP");
    densityMatPrimaryFileNameSpinDown =
        prm.get("GAUSSIAN DENSITY FOR PRIMARY RHO SPIN DOWN");
    gaussianAtomicCoord = prm.get("GAUSSIAN ATOMIC COORD FILE");
    sMatrixName = prm.get("GAUSSIAN S MATRIX FILE");
    densityMatDFTFileNameSpinUp =
        prm.get("GAUSSIAN DENSITY FOR DFT RHO SPIN UP");
    densityMatDFTFileNameSpinDown =
        prm.get("GAUSSIAN DENSITY FOR DFT RHO SPIN DOWN");
  }
  prm.leave_subsection();

  const bool printParametersToFile = false;
  if (printParametersToFile &&
      dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0) {
    prm.print_parameters(std::cout,
                         dealii::ParameterHandler::OutputStyle::LaTeX);
    exit(0);
  }

  if (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0 &&
      printParams) {
    prm.print_parameters(std::cout, dealii::ParameterHandler::ShortText);
  }
}

} // end of namespace invDFT
