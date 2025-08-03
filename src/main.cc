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
// @author Phani Motamarri, Denis Davydov, Sambit Das
//

//
// dft header
//
#include "InverseDFTBase.h"
#include "InverseDFTEngine.h"
#include "inverseDFTParameters.h"

#include "dftfeWrapper.h"
#include "runParameters.h"

//
// C++ headers
//
#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <sys/stat.h>

int main(int argc, char *argv[]) {
  //
  MPI_Init(&argc, &argv);

  dftfe::dftfeWrapper::globalHandlesInitialize(MPI_COMM_WORLD);
  const double start = MPI_Wtime();
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // deal.II tests expect parameter file as a first (!) argument
  AssertThrow(argc > 1, dealii::ExcMessage(
                            "Usage:\n"
                            "mpirun -np nProcs executable parameterfile.prm\n"
                            "\n"));
  const std::string parameter_file = argv[1];

  const std::string inverse_parameter_file = argv[2];

  dftfe::runParameters runParams;
  runParams.parse_parameters(parameter_file);

  invDFT::inverseDFTParameters invParams;
  invParams.parse_parameters(inverse_parameter_file, MPI_COMM_WORLD, true);

  dftfe::dftfeWrapper dftfeWrapped(parameter_file, MPI_COMM_WORLD, true, true,
                                   "GS", runParams.restartFilesPath,
                                   runParams.verbosity, runParams.useDevice);

  auto dftBasePtr = dftfeWrapped.getDftfeBasePtr();

  dftfe::dftParameters dftParams;
  dftParams.parse_parameters(parameter_file, MPI_COMM_WORLD, true, "GS",
                             runParams.restartFilesPath, 4, false);

  invDFT::InverseDFTBase *invBasePtr;

  AssertThrow(dftParams.finiteElementPolynomialOrder <= 8,
              dealii::ExcMessage(" Fe order is too high\n"));
  AssertThrow(dftParams.finiteElementPolynomialOrderElectrostatics <= 10,
              dealii::ExcMessage(" Fe order is too high\n"));

  int order = dftParams.finiteElementPolynomialOrder * 1000 +
              dftParams.finiteElementPolynomialOrderElectrostatics;
  if (runParams.useDevice == false) {
    switch (order) {
    case 2002:
      invBasePtr =
          new invDFT::InverseDFTEngine<2, 2, dftfe::utils::MemorySpace::HOST>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());

      break;

    case 2003:
      invBasePtr =
          new invDFT::InverseDFTEngine<2, 3, dftfe::utils::MemorySpace::HOST>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());

      break;
    case 2004:
      invBasePtr =
          new invDFT::InverseDFTEngine<2, 4, dftfe::utils::MemorySpace::HOST>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());

      break;
    case 3003:
      invBasePtr =
          new invDFT::InverseDFTEngine<3, 3, dftfe::utils::MemorySpace::HOST>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());

      break;
    case 3004:
      invBasePtr =
          new invDFT::InverseDFTEngine<3, 4, dftfe::utils::MemorySpace::HOST>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());

      break;
    case 3005:
      invBasePtr =
          new invDFT::InverseDFTEngine<3, 5, dftfe::utils::MemorySpace::HOST>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());

      break;
    case 3006:
      invBasePtr =
          new invDFT::InverseDFTEngine<3, 6, dftfe::utils::MemorySpace::HOST>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());

      break;
    case 4004:
      invBasePtr =
          new invDFT::InverseDFTEngine<4, 4, dftfe::utils::MemorySpace::HOST>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 4005:
      invBasePtr =
          new invDFT::InverseDFTEngine<4, 5, dftfe::utils::MemorySpace::HOST>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 4006:
      invBasePtr =
          new invDFT::InverseDFTEngine<4, 6, dftfe::utils::MemorySpace::HOST>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 4007:
      invBasePtr =
          new invDFT::InverseDFTEngine<4, 7, dftfe::utils::MemorySpace::HOST>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 5005:
      invBasePtr =
          new invDFT::InverseDFTEngine<5, 5, dftfe::utils::MemorySpace::HOST>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 5006:
      invBasePtr =
          new invDFT::InverseDFTEngine<5, 6, dftfe::utils::MemorySpace::HOST>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 5007:
      invBasePtr =
          new invDFT::InverseDFTEngine<5, 7, dftfe::utils::MemorySpace::HOST>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 5008:
      invBasePtr =
          new invDFT::InverseDFTEngine<5, 8, dftfe::utils::MemorySpace::HOST>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 6006:
      invBasePtr =
          new invDFT::InverseDFTEngine<6, 6, dftfe::utils::MemorySpace::HOST>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 6007:
      invBasePtr =
          new invDFT::InverseDFTEngine<6, 7, dftfe::utils::MemorySpace::HOST>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 6008:
      invBasePtr =
          new invDFT::InverseDFTEngine<6, 8, dftfe::utils::MemorySpace::HOST>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 6009:
      invBasePtr =
          new invDFT::InverseDFTEngine<6, 9, dftfe::utils::MemorySpace::HOST>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 7007:
      invBasePtr =
          new invDFT::InverseDFTEngine<7, 7, dftfe::utils::MemorySpace::HOST>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 7008:
      invBasePtr =
          new invDFT::InverseDFTEngine<7, 8, dftfe::utils::MemorySpace::HOST>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 7009:
      invBasePtr =
          new invDFT::InverseDFTEngine<7, 9, dftfe::utils::MemorySpace::HOST>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    default:
      AssertThrow(false,
                  dealii::ExcMessage(
                      "InvDFT Error: FEOrder is not an appropriate value"));
      break;
    }
  }

#ifdef DFTFE_WITH_DEVICE
  if (runParams.useDevice == true) {
    switch (order) {
    case 2002:
      invBasePtr =
          new invDFT::InverseDFTEngine<2, 2, dftfe::utils::MemorySpace::DEVICE>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 2003:
      invBasePtr =
          new invDFT::InverseDFTEngine<2, 3, dftfe::utils::MemorySpace::DEVICE>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 2004:
      invBasePtr =
          new invDFT::InverseDFTEngine<2, 4, dftfe::utils::MemorySpace::DEVICE>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 3003:
      invBasePtr =
          new invDFT::InverseDFTEngine<3, 3, dftfe::utils::MemorySpace::DEVICE>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 3004:
      invBasePtr =
          new invDFT::InverseDFTEngine<3, 4, dftfe::utils::MemorySpace::DEVICE>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 3005:
      invBasePtr =
          new invDFT::InverseDFTEngine<3, 5, dftfe::utils::MemorySpace::DEVICE>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 3006:
      invBasePtr =
          new invDFT::InverseDFTEngine<3, 6, dftfe::utils::MemorySpace::DEVICE>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 4004:
      invBasePtr =
          new invDFT::InverseDFTEngine<4, 4, dftfe::utils::MemorySpace::DEVICE>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 4005:
      invBasePtr =
          new invDFT::InverseDFTEngine<4, 5, dftfe::utils::MemorySpace::DEVICE>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 4006:
      invBasePtr =
          new invDFT::InverseDFTEngine<4, 6, dftfe::utils::MemorySpace::DEVICE>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 4007:
      invBasePtr =
          new invDFT::InverseDFTEngine<4, 7, dftfe::utils::MemorySpace::DEVICE>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 5005:
      invBasePtr =
          new invDFT::InverseDFTEngine<5, 5, dftfe::utils::MemorySpace::DEVICE>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 5006:
      invBasePtr =
          new invDFT::InverseDFTEngine<5, 6, dftfe::utils::MemorySpace::DEVICE>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 5007:
      invBasePtr =
          new invDFT::InverseDFTEngine<5, 7, dftfe::utils::MemorySpace::DEVICE>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 5008:
      invBasePtr =
          new invDFT::InverseDFTEngine<5, 8, dftfe::utils::MemorySpace::DEVICE>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 6006:
      invBasePtr =
          new invDFT::InverseDFTEngine<6, 6, dftfe::utils::MemorySpace::DEVICE>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 6007:
      invBasePtr =
          new invDFT::InverseDFTEngine<6, 7, dftfe::utils::MemorySpace::DEVICE>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 6008:
      invBasePtr =
          new invDFT::InverseDFTEngine<6, 8, dftfe::utils::MemorySpace::DEVICE>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 6009:
      invBasePtr =
          new invDFT::InverseDFTEngine<6, 9, dftfe::utils::MemorySpace::DEVICE>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 7007:
      invBasePtr =
          new invDFT::InverseDFTEngine<7, 7, dftfe::utils::MemorySpace::DEVICE>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 7008:
      invBasePtr =
          new invDFT::InverseDFTEngine<7, 8, dftfe::utils::MemorySpace::DEVICE>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    case 7009:
      invBasePtr =
          new invDFT::InverseDFTEngine<7, 9, dftfe::utils::MemorySpace::DEVICE>(
              *dftBasePtr, dftParams, invParams, dftBasePtr->getMPIParent(),
              dftBasePtr->getMPIDomain(), dftBasePtr->getMPIInterBand(),
              dftBasePtr->getMPIInterPool());
      break;
    default:
      AssertThrow(false,
                  dealii::ExcMessage(
                      "InvDFT Error: FEOrder is not an appropriate value"));
      break;
    }
  }
#endif

  dftfeWrapped.run();
  invBasePtr->run();

  delete invBasePtr;

  const double end = MPI_Wtime();
  if (runParams.verbosity >= 1 && world_rank == 0) {
    std::cout << "============================================================="
                 "================================"
              << std::endl;
    std::cout
        << "invDFT Program ends. Elapsed wall time since start of the program: "
        << end - start << " seconds." << std::endl;
    std::cout << "============================================================="
                 "================================"
              << std::endl;
  }

  dftfe::dftfeWrapper::globalHandlesFinalize();
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
