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

#ifndef DFTFE_INVERSEDFT_H
#define DFTFE_INVERSEDFT_H
#include "InverseDFTBase.h"
#include "headers.h"
#include <InverseDFTSolverFunction.h>
#include <energyCalculator.h>

#include <TriangulationManagerVxc.h>
#include <constraintMatrixInfo.h>
#include <dftUtils.h>
#include <map>
#include <vector>
namespace invDFT {
template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
class InverseDFTEngine : public InverseDFTBase {
public:
  InverseDFTEngine(dftfe::dftBase &dft, dftfe::dftParameters &dftParams,
                   inverseDFTParameters &inverseDFTParams,
                   const MPI_Comm &mpi_comm_parent,
                   const MPI_Comm &mpi_comm_domain,
                   const MPI_Comm &mpi_comm_bandgroup,
                   const MPI_Comm &mpi_comm_interpool);
  ~InverseDFTEngine();

  void run() override;

  void testAdjoint() override;

  void interpolateVxc();

  void projectVxcToParentMesh(
      std::vector<dftfe::distributedCPUVec<double>> &vxcInitialGuessParentMesh,
      std::vector<dftfe::distributedCPUVec<double>> &vxcChildNodes);

  void readDensityDataFromFile(
      std::vector<
          dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
          &densityParentQuad,
      std::vector<double> &quadCoord);

    void readDensityDataFromFileWithSearch(
            std::vector<dftfe::utils::MemoryStorage<
                    double, dftfe::utils::MemorySpace::HOST>> &densityParentQuad,
            std::vector<double> &quadCoord);

private:
  //
  // TODO
  // 1. create inverseDFTDoFManager object
  // 2. create MatrixFree object for child: contains only the pot and its
  // constraint (hanging node + periodic BC)
  //
  void createParentChildDofManager();

  //
  // @note For the spin unpolarized case, rhoTarget is just the rho_up
  // (=rho_down) and not the total rho
  //
  void setTargetDensity(
      const std::vector<
          dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
          &rhoOutValues,
      const std::vector<
          dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
          &rhoOutValuesSpinPolarized);

  //      void computeMomentOfInertia(const std::vector<std::vector<double>>
  //      &density,
  //                                               const std::vector<double>
  //                                               &coordinates, const
  //                                               std::vector<double>
  //                                               &JxWValues,
  //                                               std::vector<double>
  //                                               &I_density);
  //
  // TODO
  // Make it generic for spin
  // 1. Fetch eigenvectors from dftClass and evaluate rho on nodes of parent
  // mesh
  // 2. Evaluate v_slater from the above rho
  // 3. interpolate v_slater to child quad points
  // 4. perform L2 projection to get v_slater on child mesh nodes
  //

  // void
  // setInitialPot();

  void setInitialPotL2Proj();

  void setInitialDensityFromGaussian(
      std::vector<
          dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
          &rhoValuesFeSpin);

  void readVxcDataFromFile(
      std::vector<dftfe::distributedCPUVec<double>> &vxcChildNodes);

    void readVxcDataFromFileWithSearch(
            std::vector<dftfe::distributedCPUVec<double>> &vxcChildNodes);

  void readVxcInput();

  void computeHartreePotOnParentQuad(
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          &hartreeQuadData);

  //
  // TODO
  // 1. Solve the electrostatic potential on parent mesh using rhoTarget_total
  // (add both the spins for spin polarized case or take twice the spin_up for
  // the unpolarized case)
  // 2. Interpolate to quad on parent mesh
  //
  void setPotBase();

  void setPotBaseExactNuclear();

  void setPotBasePoissonNuclear();

  void solvePhiTotalAllElectronNonPeriodic(
      dftfe::distributedCPUVec<double> &x,
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          &rhoValues,
      const MPI_Comm &mpiComm_parent, const MPI_Comm &mpiComm_domain);

  //
  // TODO
  // 1. Set the adjoint to be homogenous for density below a tolerance (use
  // the total density from dftClass on the nodes of parent mesh)
  // 2. Create the MatrixFree object parent
  //
  void setAdjointBoundaryCondition(dftfe::distributedCPUVec<double> &rhoTarget);

  std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      d_rhoTarget;
  std::vector<double> d_sigmaGradRhoTarget;

  // triangulationManagerVxc *d_triaManagerVxcPtr;
  TriangulationManagerVxc d_triaManagerVxc;
  /// data members for the mpi implementation
  const MPI_Comm d_mpiComm_domain;
  const MPI_Comm d_mpiComm_parent;
  const MPI_Comm d_mpiComm_bandgroup;
  const MPI_Comm d_mpiComm_interpool;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  dealii::ConditionalOStream pcout;

  dftfe::triangulationManager *d_dftTriaManager;
  dftfe::dftParameters d_dftParams;
  inverseDFTParameters d_inverseDFTParams;

  dealii::MatrixFree<3, double> d_matrixFreeDataVxc;
  unsigned int d_dofHandlerVxcIndex;
  dealii::DoFHandler<3> d_dofHandlerTriaVxc;
  dealii::AffineConstraints<double> d_constraintMatrixVxc;

  dealii::QGauss<3> d_gaussQuadVxc;
  unsigned int d_quadVxcIndex;
  std::vector<std::map<unsigned int,
                       typename dealii::DoFHandler<3>::active_cell_iterator>>
      d_mapParentCellToChildCellsIter;
  std::vector<std::vector<unsigned int>> d_mapParentCellsToChild;
  std::vector<unsigned int> d_mapChildCellsToParent;

  const dealii::Quadrature<3> *d_gaussQuadAdjoint;
  unsigned int d_quadAdjointIndex;
  dealii::MatrixFree<3, double> d_matrixFreeDataAdjoint;
  unsigned int d_adjointMFAdjointConstraints, d_adjointMFPsiConstraints;
  dealii::AffineConstraints<double> d_constraintMatrixAdjoint;

  dftfe::dftClass<FEOrder, FEOrderElectro, memorySpace> *d_dftBaseClass;
  const dealii::MatrixFree<3, double> *d_dftMatrixFreeData;
  unsigned int d_dftDensityDoFHandlerIndex, d_dftQuadIndex;

  const dealii::DoFHandler<3> *d_dofHandlerDFTClass;
  dealii::AffineConstraints<double> *d_constraintDFTClass;

  const dealii::MatrixFree<3, double> *d_dftMatrixFreeDataElectro;
  unsigned int d_dftElectroDoFHandlerIndex, d_dftElectroRhsQuadIndex,
      d_dftElectroAxQuadIndex;
  const dealii::DoFHandler<3> *d_dofHandlerElectroDFTClass;

  std::shared_ptr<
      dftfe::TransferDataBetweenMeshesIncompatiblePartitioning<memorySpace>>
      d_inverseDftDoFManagerObjPtr;
  // std::shared_ptr<TransferDataBetweenMeshesBase>
  // d_inverseDftDoFManagerObjPtr;

  double d_rhoTargetTolForConstraints;

  unsigned int d_numSpins, d_numKPoints, d_numEigenValues;
  std::vector<double> d_kpointWeights;

  std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      d_potBaseQuadData;
  std::vector<dftfe::distributedCPUVec<double>> d_vxcInitialChildNodes;

  // TODO only for debugging purpose
  std::vector<std::vector<std::vector<double>>> d_targetPotValuesParentQuadData;

  std::shared_ptr<dftfe::basis::FEBasisOperations<dftfe::dataTypes::number,
                                                  double, memorySpace>>
      d_basisOperationsChildPtr;

  std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      d_blasWrapperHost;

  std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
      d_blasWrapperMemSpace;

  std::shared_ptr<dftfe::basis::FEBasisOperations<dftfe::dataTypes::number,
                                                  double, memorySpace>>
      d_basisOperationsMemSpace, d_basisOperationsElectroMemSpace;

  std::shared_ptr<dftfe::basis::FEBasisOperations<
      dftfe::dataTypes::number, double, dftfe::utils::MemorySpace::HOST>>
      d_basisOperationsHost, d_basisOperationsElectroHost,
      d_basisOperationsChildHostPtr;

  std::vector<std::shared_ptr<dftfe::basis::FEBasisOperations<
      dftfe::dataTypes::number, double, memorySpace>>>
      d_basisOperationsAdjointMemSpacePtr;
  std::vector<std::shared_ptr<dftfe::basis::FEBasisOperations<
      dftfe::dataTypes::number, double, dftfe::utils::MemorySpace::HOST>>>
      d_basisOperationsAdjointHostPtr;

  std::vector<const dealii::AffineConstraints<double> *>
      d_constraintsVectorAdjoint;

  std::vector<double> d_quadCoordinatesParent;
  std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      d_vxcLDAQuadData;
}; // end of inverseDFT class

} // end of namespace invDFT
#endif // DFTFE_INVERSEDFT_H
