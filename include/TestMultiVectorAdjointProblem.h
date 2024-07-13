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

#ifndef DFTFE_EXE_TESTMULTIVECTORADJOINTPROBLEM_H
#define DFTFE_EXE_TESTMULTIVECTORADJOINTPROBLEM_H

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
    KohnShamHamiltonianOperator<memorySpace> &kohmShamObj,
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
    const MPI_Comm &interpoolcomm);
}

#endif // DFTFE_EXE_TESTMULTIVECTORADJOINTPROBLEM_H
