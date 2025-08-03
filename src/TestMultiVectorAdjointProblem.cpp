// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE
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

#include "MultiVectorAdjointLinearSolverProblem.h"
#include "MultiVectorMinResSolver.h"

namespace dftfe {
template <unsigned int FeOrder, unsigned int FeOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void testMultiVectorAdjointProblem(
    const std::shared_ptr<dftfe::basis::FEBasisOperations<
        double, double, dftfe::utils::MemorySpace::HOST>> &basisOperationsPtr,
    dealii::MatrixFree<3, double> &matrixFreeData,
    std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
        BLASWrapperPtr,
    KohnShamDFTBaseOperator<memorySpace> &kohmShamObj,
    const dealii::AffineConstraints<double> &constraintMatrixPsi,
    std::vector<const dealii::AffineConstraints<double> *> &constraintMatrixVec,
    const dealii::AffineConstraints<double> &constraintMatrixAdjoint,
    std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &densityFromSCF,
    const dftfe::utils::MemoryStorage<double, memorySpace> &psiStdVecInput,
    const std::vector<std::vector<double>> &eigenValuesInput,
    const std::vector<std::vector<double>> &partialOccupancies,
    unsigned int noOfSpin, unsigned int noKPoints,
    unsigned int numberOfWaveFunctions,
    const unsigned int matrixFreePsiVectorComponent,
    const unsigned int matrixFreeAdjointVectorComponent,
    const unsigned int matrixFreeQuadratureComponentRhs,
    const MPI_Comm &mpi_comm_parent, const MPI_Comm &mpi_comm_domain,
    const MPI_Comm &interpoolcomm) {
  unsigned int domainMPIRank =
      dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain);

  unsigned int kPoolMPIRank =
      dealii::Utilities::MPI::this_mpi_process(interpoolcomm);

  std::shared_ptr<dftfe::basis::FEBasisOperations<
      dftfe::dataTypes::number, double, dftfe::utils::MemorySpace::HOST>>
      basisOpHost = std::make_shared<dftfe::basis::FEBasisOperations<
          dftfe::dataTypes::number, double, dftfe::utils::MemorySpace::HOST>>(
          BLASWrapperPtr);

  std::vector<dftfe::basis::UpdateFlags> updateFlags;
  updateFlags.resize(1);
  updateFlags[0] = dftfe::basis::update_jxw | dftfe::basis::update_values |
                   dftfe::basis::update_gradients |
                   dftfe::basis::update_quadpoints |
                   dftfe::basis::update_transpose;

  std::vector<unsigned int> quadVec;
  quadVec.resize(1);
  quadVec[0] = matrixFreeQuadratureComponentRhs;

  std::cout << " matrixFreeQuadratureComponentRhsDensity = "
            << matrixFreeQuadratureComponentRhs << "\n";

  basisOpHost->init(matrixFreeData, constraintMatrixVec,
                    matrixFreeVectorComponent, quadVec, updateFlags);

  unsigned int locallyOwnedSize = basisOpHost->nOwnedDofs();
  MultiVectorAdjointLinearSolverProblem<memorySpace> adjointProblemObj(
      mpi_comm_parent, mpi_comm_domain);

  std::cout << " Entering reinit\n";

  adjointProblemObj.reinit(
      BLASWrapperPtr, basisOpHost, kohmShamObj, constraintMatrixAdjoint,
      matrixFreeAdjointVectorComponent, matrixFreeQuadratureComponentRhs,
      true); // isComputeDiagonalA

  dftfe::MultiVectorMinResSolver linearSolver(mpi_comm_parent, mpi_comm_domain);

  dftUtils::constraintMatrixInfo<memorySpace> constraintsMatrixPsiDataInfo;

  const dealii::DoFHandler<3> *dofHandlerAdjoint =
      &matrixFreeData.get_dof_handler(matrixFreeAdjointVectorComponent);

  unsigned int locallyOwnedDofs = dofHandlerAdjoint->n_locally_owned_dofs();
  // TODO read the target density from file

  unsigned int numTotallyOwnedCells = matrixFreeData.n_physical_cells();

  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      targetDensity;

  const dealii::Quadrature<3> &quadratureRhs =
      matrixFreeData.get_quadrature(matrixFreeQuadratureComponentRhs);

  const unsigned int numberQuadraturePointsRhs = quadratureRhs.size();

  targetDensity.resize(numTotallyOwnedCells * numberQuadraturePointsRhs);

  dealii::FEValues<3> fe_valuesRhs(dofHandlerAdjoint->get_fe(), quadratureRhs,
                                   dealii::update_JxW_values);

  unsigned int defaultBlockSize = 100;
  distributedCPUMultiVec<double> psiBlockVec, boundaryValues, multiVectorOutput;

  for (unsigned int spinIndex = 0; spinIndex < noOfSpin; ++spinIndex) {
    for (unsigned int kPoint = 0; kPoint < noKPoints; ++kPoint) {
      const std::vector<double> &waveCurrentKPoint =
          psiStdVecInput[noOfSpin * kPoint + spinIndex];

      kohmShamObj.reinitkPointSpinIndex(kPoint, spinIndex);
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          differenceInDensities;

      differenceInDensities.resize(
          numTotallyOwnedCells * numberQuadraturePointsRhs, 0.0);

      typename dealii::DoFHandler<3>::active_cell_iterator
          cell = dofHandlerAdjoint->begin_active(),
          endc = dofHandlerAdjoint->end();
      unsigned int iElem = 0;
      std::cout << "Reading density values \n";

      double densityDiffNorm = 0.0;
      for (; cell != endc; ++cell)
        if (cell->is_locally_owned()) {
          fe_valuesRhs.reinit(cell);

          for (unsigned int iQuad = 0; iQuad < numberQuadraturePointsRhs;
               iQuad++) {
            // Assumes the ordering of writing is same

            targetDensity[iElem * numberQuadraturePointsRhs + iQuad] =
                densityFromSCF[0][iElem * numberQuadraturePointsRhs + iQuad];

            differenceInDensities[iElem * numberQuadraturePointsRhs + iQuad] =
                0.1 * targetDensity[iElem * numberQuadraturePointsRhs + iQuad];

            densityDiffNorm +=
                differenceInDensities[iElem * numberQuadraturePointsRhs +
                                      iQuad] *
                differenceInDensities[iElem * numberQuadraturePointsRhs +
                                      iQuad] *
                fe_valuesRhs.JxW(iQuad);
          }
          iElem++;
        }

      MPI_Allreduce(MPI_IN_PLACE, &densityDiffNorm, 1, MPI_DOUBLE, MPI_SUM,
                    mpi_comm_domain);

      densityDiffNorm = std::sqrt(densityDiffNorm);

      std::cout << " norm of density diff = " << densityDiffNorm << "\n";

      for (unsigned int jvec = 0; jvec < numberOfWaveFunctions;
           jvec += defaultBlockSize) {
        const unsigned int currentBlockSize =
            std::min(defaultBlockSize, numberOfWaveFunctions - jvec);

        if (currentBlockSize != defaultBlockSize || jvec == 0) {
          dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
              matrixFreeData.get_vector_partitioner(
                  matrixFreePsiVectorComponent),
              currentBlockSize, psiBlockVec);

          dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
              matrixFreeData.get_vector_partitioner(
                  matrixFreeAdjointVectorComponent),
              currentBlockSize, multiVectorOutput);

          boundaryValues.reinit(multiVectorOutput);

          constraintsMatrixPsiDataInfo.initialize(
              matrixFreeData.get_vector_partitioner(
                  matrixFreePsiVectorComponent),
              constraintMatrixPsi);
        }

        // We assume that there is only homogenous Dirichlet BC
        boundaryValues.setValue(0.0);

        multiVectorOutput.setValue(0.0);

        for (unsigned int iNode = 0; iNode < locallyOwnedSize; ++iNode)
          for (unsigned int iWave = 0; iWave < currentBlockSize; ++iWave)
            psiBlockVec.data()[iNode * currentBlockSize + iWave] =
                waveCurrentKPoint[iNode * numberOfWaveFunctions + jvec + iWave];

        constraintsMatrixPsiDataInfo.distribute(psiBlockVec);

        std::vector<double> effectiveOrbitalOccupancy;
        std::vector<std::vector<unsigned int>> degeneracy;
        effectiveOrbitalOccupancy.resize(currentBlockSize);
        degeneracy.resize(currentBlockSize);
        std::vector<double> shiftValues;
        shiftValues.resize(currentBlockSize);
        for (unsigned int iBlock = 0; iBlock < currentBlockSize; iBlock++) {
          shiftValues[iBlock] =
              eigenValuesInput[kPoint][numberOfWaveFunctions * spinIndex +
                                       iBlock + jvec];

          // TODO setting the eigenvalues zero for testing
          //                    shiftValues[iBlock] =  -10.0;
          if (noOfSpin == 1) {
            effectiveOrbitalOccupancy[iBlock] =
                2.0 *
                partialOccupancies[kPoint][numberOfWaveFunctions * spinIndex +
                                           iBlock + jvec];
          } else {
            effectiveOrbitalOccupancy[iBlock] =
                partialOccupancies[kPoint][numberOfWaveFunctions * spinIndex +
                                           iBlock + jvec];
          }

          // TODO assumes there is no degeneracy
          // TODO have to extend this further
          degeneracy[iBlock].push_back(iBlock);
        }
        std::cout << " updating psi\n";
        adjointProblemObj.updateInputPsi(psiBlockVec, effectiveOrbitalOccupancy,
                                         differenceInDensities, degeneracy,
                                         shiftValues, currentBlockSize);

        std::cout << " Entering minres solve\n";

        linearSolver.solve(adjointProblemObj, BLASWrapperPtr, multiVectorOutput,
                           boundaryValues, locallyOwnedSize, currentBlockSize,
                           5e-9, 5000, 4, true);

        dealii::DataOut<3, dealii::DoFHandler<3>> data_out;

        data_out.attach_dof_handler(*dofHandlerAdjoint);

        std::vector<distributedCPUVec<double>> singleVectorOutput;
        singleVectorOutput.resize(currentBlockSize);
        for (unsigned int iBlock = 0; iBlock < currentBlockSize; iBlock++) {
          vectorTools::createDealiiVector<double>(
              matrixFreeData.get_vector_partitioner(
                  matrixFreePsiVectorComponent),
              1, singleVectorOutput[iBlock]);

          for (unsigned int iNode = 0; iNode < locallyOwnedDofs; iNode++) {
            singleVectorOutput[iBlock].local_element(iNode) =
                multiVectorOutput.data()[iNode * currentBlockSize + iBlock];
          }
          std::cout << " writing output\n";
          singleVectorOutput[iBlock].update_ghost_values();
          std::string outputVecName = "solution[";
          outputVecName = outputVecName + std::to_string(iBlock);
          outputVecName = outputVecName + "]";
          data_out.add_data_vector(singleVectorOutput[iBlock], outputVecName);
        }

        data_out.build_patches();

        // The final step in generating output is to determine a file
        // name, open the file, and write the data into it (here, we use
        // VTK format):
        //   const std::string filename =
        //     "solution-" + Utilities::int_to_string(cycle, 2) +
        //     ".vtk";
        //   std::ofstream output(filename);
        //   data_out.write_vtk(output);

        unsigned int kPointSpinIndex =
            noOfSpin * kPoint * numberOfWaveFunctions +
            spinIndex * numberOfWaveFunctions + jvec;
        data_out.write_vtu_with_pvtu_record("./", "adjoint", kPointSpinIndex,
                                            mpi_comm_domain, 2, 4);
      }
    }
  }
}
} // namespace dftfe
