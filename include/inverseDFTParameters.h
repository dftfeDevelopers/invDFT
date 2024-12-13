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

#ifndef DFTFE_EXE_INVERSEDFTPARAMETERS_H
#define DFTFE_EXE_INVERSEDFTPARAMETERS_H

#include <deal.II/base/data_out_base.h>
#include <deal.II/base/parameter_handler.h>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <string>

namespace invDFT {
class inverseDFTParameters {
public:
  std::string solvermode;

  // post process
  bool readPointsFromFile;
  bool writeVtuFile;
  bool writeToPoints;
  std::string fileNameReadPoints;
  std::string fileNameWriteVxcPostProcess;
  std::string BFGSSolverType;
  double startX, startY, startZ;
  double endX, endY, endZ;
  unsigned int numPointsX, numPointsY, numPointsZ;
  // Parameters for inverse problem
  double inverseBFGSTol;
  unsigned int inverseBFGSLineSearch;
  double inverseBFGSLineSearchTol;
  unsigned int inverseBFGSHistory;
  unsigned int inverseMaxBFGSIter;
  bool writeVxcData;
  bool readVxcData;
  std::string fileNameReadVxcPostFix;
  std::string vxcDataFolder;
  std::string fileNameWriteVxcPostFix;
  unsigned int writeVxcFrequency;

  unsigned int additionalEigenStatesSolved;
  unsigned int netCharge;
  bool useDeltaRhoCorrection;
  double adaptiveFactorForAdjoint;
  double adaptiveFactorForChebFiltering;
  double initialTolForChebFiltering;
  double rhoTolForConstraints;
  double VxcInnerDomain;
  double VxcInnerMeshSize;
  double inverseAdjointInitialTol;
  unsigned int inverseAdjointMaxIterations;
  double inverseAlpha1ForWeights;
  double inverseAlpha2ForWeights;
  double inverseTauForSmoothening;
  double inverseTauForVxBc;
  double inverseTauForFABC;
  double inverseFractionOccTol;
  double inverseDegeneracyTol;

  bool readFEDensity;
  std::string fileNameReadDensity;
  unsigned int interBlockSize;
  bool useLb94InInitialguess;
  bool useMemOptForTransfer;
  bool readGaussian;
  bool fermiAmaldiBC;
  double factorFermiAmaldi;
  std::string densityMatPrimaryFileNameSpinUp;
  std::string densityMatPrimaryFileNameSpinDown;
  std::string gaussianAtomicCoord;
  std::string sMatrixName;
  std::string densityMatDFTFileNameSpinUp;
  std::string densityMatDFTFileNameSpinDown;

  double factorForLDAVxc;
  inverseDFTParameters();

  /**
   * Parse parameters.
   */
  void parse_parameters(const std::string &parameter_file,
                        const MPI_Comm &mpi_comm_parent,
                        const bool printParams = false);

}; // class dftParameters
} // end of namespace invDFT

#endif // DFTFE_EXE_INVERSEDFTPARAMETERS_H
