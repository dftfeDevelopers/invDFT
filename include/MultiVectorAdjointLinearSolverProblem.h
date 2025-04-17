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
// @author Vishal Subramanian, Bikask Kanungo
//

#ifndef DFTFE_MULTIVECTORADJOINTLINEARSOLVERPROBLEM_H
#define DFTFE_MULTIVECTORADJOINTLINEARSOLVERPROBLEM_H

#include "BLASWrapper.h"
#include "FEBasisOperations.h"
#include "KohnShamDFTBaseOperator.h"
#include "MultiVectorLinearSolverProblem.h"
#include "headers.h"
namespace invDFT {
template <dftfe::utils::MemorySpace memorySpace>
class MultiVectorAdjointLinearSolverProblem
    : public dftfe::MultiVectorLinearSolverProblem<memorySpace> {
public:
  MultiVectorAdjointLinearSolverProblem(const MPI_Comm &mpi_comm_parent,
                                        const MPI_Comm &mpi_comm_domain);

  // Destructor
  ~MultiVectorAdjointLinearSolverProblem();

  /**
   * @brief Compute right hand side vector for the problem Ax = rhs.
   *
   * @param rhs vector for the right hand side values
   */
  dftfe::linearAlgebra::MultiVector<dftfe::dataTypes::number, memorySpace> &
  computeRhs(dftfe::linearAlgebra::MultiVector<dftfe::dataTypes::number,
                                               memorySpace> &NDBCVec,
             dftfe::linearAlgebra::MultiVector<dftfe::dataTypes::number,
                                               memorySpace> &outputVec,
             unsigned int blockSizeInput) override;

  /**
   * @brief Compute A matrix multipled by x.
   *
   */
  void
  vmult(dftfe::linearAlgebra::MultiVector<dftfe::dataTypes::number, memorySpace>
            &Ax,
        dftfe::linearAlgebra::MultiVector<dftfe::dataTypes::number, memorySpace>
            &x,
        unsigned int blockSize) override;

  /**
   * @brief Apply the constraints to the solution vector.
   *
   */
  void distributeX() override;

  /**
   * @brief Jacobi preconditioning function.
   *
   */
  void precondition_Jacobi(
      dftfe::linearAlgebra::MultiVector<dftfe::dataTypes::number, memorySpace>
          &dst,
      const dftfe::linearAlgebra::MultiVector<dftfe::dataTypes::number,
                                              memorySpace> &src,
      const double omega) const override;

  /**
   * @brief Apply square-root of the Jacobi preconditioner function.
   *
   */
  void precondition_JacobiSqrt(
      dftfe::linearAlgebra::MultiVector<dftfe::dataTypes::number, memorySpace>
          &dst,
      const dftfe::linearAlgebra::MultiVector<dftfe::dataTypes::number,
                                              memorySpace> &src,
      const double omega) const override;

  void reinit(std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
                  BLASWrapperPtr,
              std::shared_ptr<dftfe::basis::FEBasisOperations<
                  dftfe::dataTypes::number, double, memorySpace>>
                  basisOperationsPtr,
              dftfe::KohnShamDFTBaseOperator<memorySpace> &ksHamiltonianObj,
              const dealii::AffineConstraints<double> &constraintMatrix,
              const double TVal, const unsigned int matrixFreeVectorComponent,
              const unsigned int matrixFreeQuadratureComponentRhs,
              const bool isComputeDiagonalA);

  void multiVectorDotProdQuadWise(
      dftfe::linearAlgebra::MultiVector<dftfe::dataTypes::number, memorySpace>
          &vec1,
      dftfe::linearAlgebra::MultiVector<dftfe::dataTypes::number, memorySpace>
          &vec2,
      dftfe::utils::MemoryStorage<dftfe::dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST>
          &dotProductOutputHost);

  void updateInputPsi(
      dftfe::linearAlgebra::MultiVector<dftfe::dataTypes::number,
                                        memorySpace>
          &psiInputVecMemSpace, // need to call distribute
      std::vector<double>
          &effectiveOrbitalOccupancy, // incorporates spin information
      dftfe::utils::MemoryStorage<double, memorySpace> &differenceInDensity,
      std::vector<std::vector<unsigned int>> &degeneracy, double fermiEnergy,
      std::vector<double> &eigenValues, unsigned int blockSize);

private:
  void computeMuMatrix(
      dftfe::utils::MemoryStorage<double, memorySpace> &inputJxwMemSpace,
      dftfe::utils::MemoryStorage<double, memorySpace> &effectiveOrbitalOcc,
      dftfe::linearAlgebra::MultiVector<dftfe::dataTypes::number, memorySpace>
          &psiVecMemSpace);

  void computeRMatrix(
      dftfe::utils::MemoryStorage<double, memorySpace> &inputJxwMemSpace);

  void computeDiagonalA();

  /// data members for the mpi implementation
  const MPI_Comm mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  dealii::ConditionalOStream pcout;

  bool d_isComputeDiagonalA;

  /// the vector that stores the output obtained by solving the poisson
  /// problem
  dftfe::linearAlgebra::MultiVector<dftfe::dataTypes::number, memorySpace>
      *d_blockedXPtr, *d_psiMemSpace;

  dftfe::linearAlgebra::MultiVector<double, memorySpace> d_diagonalSqrtA,
      d_diagonalA;

  std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
      d_BLASWrapperPtr;

  std::shared_ptr<dftfe::basis::FEBasisOperations<dftfe::dataTypes::number,
                                                  double, memorySpace>>
      d_basisOperationsPtr;

  /// pointer to dealii MatrixFree object
  const dealii::MatrixFree<3, double> *d_matrixFreeDataPtr;

  /// pointer to dealii dealii::AffineConstraints<double> object
  const dealii::AffineConstraints<double> *d_constraintMatrixPtr;

  dftfe::dftUtils::constraintMatrixInfo<memorySpace> d_constraintsInfo;

  unsigned int d_matrixFreeQuadratureComponentRhs;
  unsigned int d_matrixFreeVectorComponent;
  unsigned int d_blockSize;
  unsigned int d_locallyOwnedSize, d_numberDofsPerElement, d_numCells,
      d_numQuadsPerCell;

  dftfe::KohnShamDFTBaseOperator<memorySpace> *d_ksOperatorPtr;
  dftfe::utils::MemoryStorage<dftfe::global_size_type, memorySpace>
      d_mapNodeIdToProcId;
  dftfe::utils::MemoryStorage<dftfe::global_size_type, memorySpace>
      d_mapQuadIdToProcId;

  dftfe::utils::MemoryStorage<double, memorySpace> tempOutputDotProdMemSpace,
      oneBlockSizeMemSpace;

  dftfe::utils::MemoryStorage<double, memorySpace> d_negEigenValuesMemSpace;
  dftfe::linearAlgebra::MultiVector<double, memorySpace> d_rhsMemSpace;
  dftfe::utils::MemoryStorage<dftfe::dataTypes::number, memorySpace>
      vec1QuadValues, vec2QuadValues, vecOutputQuadValues;

  dftfe::utils::MemoryStorage<double, memorySpace> d_RMatrixMemSpace,
      d_MuMatrixMemSpace;

  dftfe::utils::MemoryStorage<double, memorySpace>
      d_effectiveOrbitalOccupancyMemSpace;

  std::vector<std::vector<unsigned int>> d_degenerateState;
  std::vector<double> d_eigenValues;

  std::vector<unsigned int> d_vectorList;

  dftfe::utils::MemoryStorage<unsigned int, memorySpace> d_vectorListMemSpace;

  dftfe::utils::MemoryStorage<double, memorySpace>
      d_4xeffectiveOrbitalOccupancyMemSpace;

  dftfe::utils::MemoryStorage<double, memorySpace> d_inputJxWMemSpace;

  dftfe::utils::MemoryStorage<double, memorySpace>
      d_cellWaveFunctionQuadMatrixMemSpace, d_cellWaveFunctionMatrixMemSpace;
  dftfe::utils::MemoryStorage<double, memorySpace> d_MuMatrixMemSpaceCellWise;
  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_MuMatrixHost, d_MuMatrixHostCellWise;

  dftfe::utils::MemoryStorage<double, memorySpace> d_onesMemSpace,
      d_onesQuadMemSpace;

  dftfe::utils::MemoryStorage<double, memorySpace>
      d_cellRMatrixTimesWaveMatrixMemSpace;

  double d_fermiEnergy, d_TVal;
  unsigned int d_cellBlockSize;
};
} // end of namespace invDFT
#endif // DFTFE_MULTIVECTORADJOINTLINEARSOLVERPROBLEM_H
