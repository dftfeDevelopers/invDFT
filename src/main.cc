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
#include "InverseDFTEngine.h"
#include "InverseDFTBase.h"
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


int
main(int argc, char *argv[])
{
  //
  MPI_Init(&argc, &argv);

  dftfe::dftfeWrapper::globalHandlesInitialize(MPI_COMM_WORLD);
  const double start = MPI_Wtime();
  int          world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // deal.II tests expect parameter file as a first (!) argument
  AssertThrow(argc > 1,
              dealii::ExcMessage(
                "Usage:\n"
                "mpirun -np nProcs executable parameterfile.prm\n"
                "\n"));
  const std::string parameter_file = argv[1];

 const std::string inverse_parameter_file = argv[2];

  dftfe::runParameters runParams;
  runParams.parse_parameters(parameter_file);



  if (runParams.solvermode == "UNIT_TEST")
    {
      dftfe::dftfeWrapper dftfeWrapped(parameter_file,
                                       MPI_COMM_WORLD,
                                       true,
                                       true,
                                       "UNIT_TEST",
                                       runParams.restartFilesPath,
                                       runParams.verbosity,
                                       runParams.useDevice);
      dftfeWrapped.run();
    }
  else
    {
      dftfe::dftfeWrapper dftfeWrapped(parameter_file,
                                       MPI_COMM_WORLD,
                                       true,
                                       true,
                                       "GS",
                                       runParams.restartFilesPath,
                                       runParams.verbosity,
                                       runParams.useDevice);

      auto dftBasePtr = dftfeWrapped.getDftfeBasePtr();

      dftfe::dftParameters dftParams;
      dftParams.parse_parameters(parameter_file,
                                 MPI_COMM_WORLD,
                                 true,
                                 "GS",
                                 runParams.restartFilesPath,
                                 4,
                                 false);

      invDFT::inverseDFTParameters invParams;
      invParams.parse_parameters(inverse_parameter_file,
                                 MPI_COMM_WORLD,
                                 true);

        invDFT::InverseDFTBase* invBasePtr;

      if ( runParams.useDevice == false)
      {
          int order = dftParams.finiteElementPolynomialOrder;
          switch (order)
          {
              case 2:
                  invBasePtr = new invDFT::InverseDFTEngine<2,2,dftfe::utils::MemorySpace::HOST>(*dftBasePtr,
                                                                                     dftParams,
                                                                                     invParams,
                                                                                     dftBasePtr->getMPIParent(),
                                                                                     dftBasePtr->getMPIDomain(),
                                                                                     dftBasePtr->getMPIInterBand(),
                                                                                     dftBasePtr->getMPIInterPool());

                  break;

              case 3:
                  invBasePtr = new invDFT::InverseDFTEngine<3,3,dftfe::utils::MemorySpace::HOST>(*dftBasePtr,
                                                                                     dftParams,
                                                                                     invParams,
                                                                                     dftBasePtr->getMPIParent(),
                                                                                     dftBasePtr->getMPIDomain(),
                                                                                     dftBasePtr->getMPIInterBand(),
                                                                                     dftBasePtr->getMPIInterPool());


                  break;
              case 4:
                  invBasePtr = new invDFT::InverseDFTEngine<4,4,dftfe::utils::MemorySpace::HOST>(*dftBasePtr,
                                                                                     dftParams,
                                                                                     invParams,
                                                                                     dftBasePtr->getMPIParent(),
                                                                                     dftBasePtr->getMPIDomain(),
                                                                                     dftBasePtr->getMPIInterBand(),
                                                                                     dftBasePtr->getMPIInterPool());
                  break;
              case 5:
                  invBasePtr = new invDFT::InverseDFTEngine<5,5,dftfe::utils::MemorySpace::HOST>(*dftBasePtr,
                                                                                     dftParams,
                                                                                     invParams,
                                                                                     dftBasePtr->getMPIParent(),
                                                                                     dftBasePtr->getMPIDomain(),
                                                                                     dftBasePtr->getMPIInterBand(),
                                                                                     dftBasePtr->getMPIInterPool());
                  break;
              case 6:
                  invBasePtr = new invDFT::InverseDFTEngine<6,6,dftfe::utils::MemorySpace::HOST>(*dftBasePtr,
                                                                                     dftParams,
                                                                                     invParams,
                                                                                     dftBasePtr->getMPIParent(),
                                                                                     dftBasePtr->getMPIDomain(),
                                                                                     dftBasePtr->getMPIInterBand(),
                                                                                     dftBasePtr->getMPIInterPool());
                  break;
              default :
                  AssertThrow(
                          false,
                          dealii::ExcMessage(
                                  "InvDFT Error: FEOrder is not an appropriate value"));
                  break;
          }
      }

#ifdef DFTFE_WITH_DEVICE
      if ( runParams.useDevice == true)
      {
           int order = dftParams.finiteElementPolynomialOrder;
           switch (order)
           {
               case 2:
                    invBasePtr = new invDFT::InverseDFTEngine<2,2,dftfe::utils::MemorySpace::DEVICE>(*dftBasePtr,
                                  dftParams,
                                  invParams,
                                  dftBasePtr->getMPIParent(),
                                  dftBasePtr->getMPIDomain(),
                                  dftBasePtr->getMPIInterBand(),
                                  dftBasePtr->getMPIInterPool());
                   break;

                   case 3:
                    invBasePtr = new invDFT::InverseDFTEngine<3,3,dftfe::utils::MemorySpace::DEVICE>(*dftBasePtr,
                                  dftParams,
                                  invParams,
                                  dftBasePtr->getMPIParent(),
                                  dftBasePtr->getMPIDomain(),
                                  dftBasePtr->getMPIInterBand(),
                                  dftBasePtr->getMPIInterPool());
                   break;
                   case 4:
                    invBasePtr = new invDFT::InverseDFTEngine<4,4,dftfe::utils::MemorySpace::DEVICE>(*dftBasePtr,
                                  dftParams,
                                  invParams,
                                  dftBasePtr->getMPIParent(),
                                  dftBasePtr->getMPIDomain(),
                                  dftBasePtr->getMPIInterBand(),
                                  dftBasePtr->getMPIInterPool());
                   break;
                   case 5:
                    invBasePtr = new invDFT::InverseDFTEngine<5,5,dftfe::utils::MemorySpace::DEVICE>(*dftBasePtr,
                                  dftParams,
                                  invParams,
                                  dftBasePtr->getMPIParent(),
                                  dftBasePtr->getMPIDomain(),
                                  dftBasePtr->getMPIInterBand(),
                                  dftBasePtr->getMPIInterPool());
                   break;
                   case 6:
                    invBasePtr = new invDFT::InverseDFTEngine<6,6,dftfe::utils::MemorySpace::DEVICE>(*dftBasePtr,
                                  dftParams,
                                  invParams,
                                  dftBasePtr->getMPIParent(),
                                  dftBasePtr->getMPIDomain(),
                                  dftBasePtr->getMPIInterBand(),
                                  dftBasePtr->getMPIInterPool());
                   break;
               default :
                    AssertThrow(
                      false,
                      dealii::ExcMessage(
                        "InvDFT Error: FEOrder is not an appropriate value"));
                                   break;
           }


      }
#endif

      dftfeWrapped.run();
      invBasePtr->run();

      delete invBasePtr;

    }


  const double end = MPI_Wtime();
  if (runParams.verbosity >= 1 && world_rank == 0)
    {
      std::cout
        << "============================================================================================="
        << std::endl;
      std::cout
        << "invDFT Program ends. Elapsed wall time since start of the program: "
        << end - start << " seconds." << std::endl;
      std::cout
        << "============================================================================================="
        << std::endl;
    }

  dftfe::dftfeWrapper::globalHandlesFinalize();
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}

