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

#include "InverseDFTSolverFunction.h"
#include "CompositeData.h"
#include "MPIWriteOnFile.h"
#include "NodalData.h"
#include "dftUtils.h"
#include <densityCalculator.h>
#include <map>
#include <vector>

namespace invDFT {
namespace {

void evaluateDegeneracyMap(
    const std::vector<double> &eigenValues,
    std::vector<std::vector<unsigned int>> &degeneracyMap,
    const double degeneracyTol) {
  const unsigned int N = eigenValues.size();
  degeneracyMap.resize(N, std::vector<unsigned int>(0));
  std::map<unsigned int, std::set<unsigned int>> groupIdToEigenId;
  std::map<unsigned int, unsigned int> eigenIdToGroupId;
  unsigned int groupIdCount = 0;
  for (unsigned int i = 0; i < N; ++i) {
    auto it = eigenIdToGroupId.find(i);
    if (it != eigenIdToGroupId.end()) {
      const unsigned int groupId = it->second;
      for (unsigned int j = 0; j < N; ++j) {
        if (std::abs(eigenValues[i] - eigenValues[j]) < degeneracyTol) {
          groupIdToEigenId[groupId].insert(j);
          eigenIdToGroupId[j] = groupId;
        }
      }
    } else {
      groupIdToEigenId[groupIdCount].insert(i);
      eigenIdToGroupId[i] = groupIdCount;
      for (unsigned int j = 0; j < N; ++j) {
        if (std::abs(eigenValues[i] - eigenValues[j]) < degeneracyTol) {
          groupIdToEigenId[groupIdCount].insert(j);
          eigenIdToGroupId[j] = groupIdCount;
        }
      }

      groupIdCount++;
    }
  }

  for (unsigned int i = 0; i < N; ++i) {
    const unsigned int groupId = eigenIdToGroupId[i];
    std::set<unsigned int> degenerateIds = groupIdToEigenId[groupId];
    degeneracyMap[i].resize(degenerateIds.size());
    std::copy(degenerateIds.begin(), degenerateIds.end(),
              degeneracyMap[i].begin());
  }
}

} // namespace

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
InverseDFTSolverFunction<FEOrder, FEOrderElectro, memorySpace>::
    InverseDFTSolverFunction(const MPI_Comm &mpi_comm_parent,
                             const MPI_Comm &mpi_comm_domain,
                             const MPI_Comm &mpi_comm_interpool,
                             const MPI_Comm &mpi_comm_interband)
    : d_mpi_comm_parent(mpi_comm_parent), d_mpi_comm_domain(mpi_comm_domain),
      d_mpi_comm_interpool(mpi_comm_interpool),
      d_mpi_comm_interband(mpi_comm_interband),
      d_multiVectorAdjointProblem(mpi_comm_parent, mpi_comm_domain),
      d_multiVectorLinearMINRESSolver(mpi_comm_parent, mpi_comm_domain),
      pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_parent) == 0)),
      d_computingTimerStandard(mpi_comm_domain, pcout,
                               dealii::TimerOutput::summary,
                               dealii::TimerOutput::wall_times) {
  d_resizeMemSpaceVecDuringInterpolation = true;
  d_resizeMemSpaceBlockSizeChildQuad = true;
  d_lossPreviousIteration = 0.0;

  d_previousBlockSize = 0;
  d_numCellBlockSizeParent = 100;
  d_numCellBlockSizeChild = 100;
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTSolverFunction<FEOrder, FEOrderElectro, memorySpace>::reinit(
    const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoTargetQuadDataHost,
    const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &weightQuadDataHost,
    const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &potBaseQuadDataHost,
    dftfe::dftClass<FEOrder, FEOrder, memorySpace> &dftClass,
    const dealii::AffineConstraints<double>
        &constraintMatrixHomogeneousPsi, // assumes that the constraint matrix
                                         // has homogenous BC
    const dealii::AffineConstraints<double>
        &constraintMatrixHomogeneousAdjoint, // assumes that the constraint
                                             // matrix has homogenous BC
    const dealii::AffineConstraints<double> &constraintMatrixPot,
    std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
        &BLASWrapperHostPtr,
    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
        &BLASWrapperPtr,
    std::vector<std::shared_ptr<dftfe::basis::FEBasisOperations<
        dftfe::dataTypes::number, double, memorySpace>>>
        &basisOperationsParentPtr,
    std::vector<std::shared_ptr<dftfe::basis::FEBasisOperations<
        dftfe::dataTypes::number, double, dftfe::utils::MemorySpace::HOST>>>
        &basisOperationsParentHostPtr,
    std::shared_ptr<dftfe::basis::FEBasisOperations<
        dftfe::dataTypes::number, double, dftfe::utils::MemorySpace::HOST>>
        &basisOperationsChildPtr,
    dftfe::KohnShamHamiltonianOperator<memorySpace> &kohnShamClass,
    const std::shared_ptr<
        dftfe::TransferDataBetweenMeshesIncompatiblePartitioning<memorySpace>>
        &inverseDFTDoFManagerObjPtr,
    const std::vector<double> &kpointWeights, const unsigned int numSpins,
    const unsigned int numEigenValues,
    const unsigned int matrixFreePsiVectorComponent,
    const unsigned int matrixFreeAdjointVectorComponent,
    const unsigned int matrixFreePotVectorComponent,
    const unsigned int matrixFreeQuadratureComponentAdjointRhs,
    const unsigned int matrixFreeQuadratureComponentPot,
    const bool isComputeDiagonalA, const bool isComputeShapeFunction,
    const dftfe::dftParameters &dftParams,
    const inverseDFTParameters &inverseDFTParams) {
  d_rhoTargetQuadDataHost = rhoTargetQuadDataHost;
  d_weightQuadDataHost = weightQuadDataHost;
  d_potBaseQuadDataHost = potBaseQuadDataHost;

  d_BLASWrapperHostPtr = BLASWrapperHostPtr;
  d_BLASWrapperPtr = BLASWrapperPtr;
  d_basisOperationsParentPtr = basisOperationsParentPtr;
  d_basisOperationsParentHostPtr = basisOperationsParentHostPtr;
  d_basisOperationsChildPtr = basisOperationsChildPtr;

  d_dftClassPtr = &dftClass;
  d_constraintMatrixHomogeneousPsi = &constraintMatrixHomogeneousPsi;
  d_constraintMatrixHomogeneousAdjoint = &constraintMatrixHomogeneousAdjoint;
  d_constraintMatrixPot = &constraintMatrixPot;
  d_kohnShamClass = &kohnShamClass;

  d_transferDataPtr = inverseDFTDoFManagerObjPtr;
  d_kpointWeights = kpointWeights;
  d_numSpins = numSpins;
  d_numKPoints = d_kpointWeights.size();
  d_numEigenValues = numEigenValues;
  d_matrixFreePsiVectorComponent = matrixFreePsiVectorComponent;
  d_matrixFreeAdjointVectorComponent = matrixFreeAdjointVectorComponent;
  d_matrixFreePotVectorComponent = matrixFreePotVectorComponent;
  d_matrixFreeQuadratureComponentAdjointRhs =
      matrixFreeQuadratureComponentAdjointRhs;
  d_matrixFreeQuadratureComponentPot = matrixFreeQuadratureComponentPot;
  d_isComputeDiagonalA = isComputeDiagonalA;
  d_isComputeShapeFunction = isComputeShapeFunction;
  d_dftParams = &dftParams;
  d_inverseDFTParams = &inverseDFTParams;

  d_getForceCounter = 0;

  d_numElectrons = d_dftClassPtr->getNumElectrons();

  d_elpaScala = d_dftClassPtr->getElpaScalaManager();

#ifdef DFTFE_WITH_DEVICE
  d_subspaceIterationSolverDevice =
      d_dftClassPtr->getSubspaceIterationSolverDevice();
#endif

  d_subspaceIterationSolverHost =
      d_dftClassPtr->getSubspaceIterationSolverHost();

  d_adjointTol = d_inverseDFTParams->inverseAdjointInitialTol;
  d_adjointMaxIterations = d_inverseDFTParams->inverseAdjointMaxIterations;
  d_maxChebyPasses = 100; // This is hard coded
  d_fractionalOccupancyTol = d_inverseDFTParams->inverseFractionOccTol;

  d_degeneracyTol = d_inverseDFTParams->inverseDegeneracyTol;
  const unsigned int numKPoints = d_kpointWeights.size();
  d_wantedLower.resize(numSpins * numKPoints);
  d_unwantedUpper.resize(numSpins * numKPoints);
  d_unwantedLower.resize(numSpins * numKPoints);
  d_eigenValues.resize(numKPoints,
                       std::vector<double>(d_numSpins * d_numEigenValues, 0.0));
  d_residualNormWaveFunctions.resize(
      numSpins * numKPoints, std::vector<double>(d_numEigenValues, 0.0));

  d_fractionalOccupancy.resize(
      d_numKPoints, std::vector<double>(d_numSpins * d_numEigenValues, 0.0));

  d_numLocallyOwnedCellsParent =
      d_basisOperationsParentPtr[d_matrixFreePsiVectorComponent]->nCells();
  d_numLocallyOwnedCellsChild = d_basisOperationsChildPtr->nCells();

  d_numCellBlockSizeParent =
      std::min(d_numCellBlockSizeParent, d_numLocallyOwnedCellsParent);
  d_numCellBlockSizeChild =
      std::min(d_numCellBlockSizeChild, d_numLocallyOwnedCellsChild);

  d_basisOperationsParentPtr[d_matrixFreePsiVectorComponent]->reinit(
      d_numEigenValues, d_numCellBlockSizeParent,
      d_matrixFreeQuadratureComponentAdjointRhs, true, false);
  d_basisOperationsParentPtr[d_matrixFreeAdjointVectorComponent]->reinit(
      d_numEigenValues, d_numCellBlockSizeParent,
      d_matrixFreeQuadratureComponentAdjointRhs, true, false);

  d_basisOperationsChildPtr->reinit(1, d_numCellBlockSizeChild,
                                    d_matrixFreeQuadratureComponentPot, true,
                                    false);

  std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
  dftfe::dftUtils::createBandParallelizationIndices(
      d_mpi_comm_interband, d_numEigenValues, bandGroupLowHighPlusOneIndices);

  unsigned int BVec = std::min(d_dftParams->chebyWfcBlockSize,
                               bandGroupLowHighPlusOneIndices[1]);

  d_basisOperationsParentPtr[d_matrixFreePsiVectorComponent]
      ->createScratchMultiVectors(BVec, 1);
  if (d_numEigenValues % BVec != 0) {
    d_basisOperationsParentPtr[d_matrixFreePsiVectorComponent]
        ->createScratchMultiVectors(d_numEigenValues % BVec, 1);
  }

  d_multiVectorAdjointProblem.reinit(
      d_BLASWrapperPtr,
      d_basisOperationsParentPtr[d_matrixFreePsiVectorComponent],
      *d_kohnShamClass, *d_constraintMatrixHomogeneousPsi,
      d_matrixFreePsiVectorComponent, d_matrixFreeQuadratureComponentAdjointRhs,
      isComputeDiagonalA);

  d_matrixFreeDataParent =
      &d_basisOperationsParentPtr[d_matrixFreePsiVectorComponent]
           ->matrixFreeData();
  d_matrixFreeDataChild = &d_basisOperationsChildPtr->matrixFreeData();

  dftfe::linearAlgebra::MultiVector<double, dftfe::utils::MemorySpace::HOST>
      dummyPotVec;

  dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      d_matrixFreeDataChild->get_vector_partitioner(
          matrixFreePotVectorComponent),
      1, dummyPotVec);

  std::vector<dealii::types::global_dof_index> fullFlattenedMapChild;
  dftfe::vectorTools::computeCellLocalIndexSetMap(
      dummyPotVec.getMPIPatternP2P(), *d_matrixFreeDataChild,
      matrixFreePotVectorComponent, 1, fullFlattenedMapChild);

  d_fullFlattenedMapChild.resize(fullFlattenedMapChild.size());
  d_fullFlattenedMapChild.copyFrom(fullFlattenedMapChild);

  d_dofHandlerParent =
      &d_matrixFreeDataParent->get_dof_handler(d_matrixFreePsiVectorComponent);
  ;
  d_dofHandlerChild =
      &d_matrixFreeDataChild->get_dof_handler(d_matrixFreePotVectorComponent);
  ;

  d_constraintsMatrixDataInfoPot.initialize(
      d_basisOperationsChildPtr->matrixFreeData().get_vector_partitioner(
          d_matrixFreePotVectorComponent),
      *d_constraintMatrixPot);

  dealii::IndexSet locally_relevant_dofs_;
  dealii::DoFTools::extract_locally_relevant_dofs(*d_dofHandlerParent,
                                                  locally_relevant_dofs_);

  const dealii::IndexSet &locally_owned_dofs_ =
      d_dofHandlerParent->locally_owned_dofs();
  dealii::IndexSet ghost_indices_ = locally_relevant_dofs_;
  ghost_indices_.subtract_set(locally_owned_dofs_);

  dftfe::distributedCPUVec<double> tempVec = dftfe::distributedCPUVec<double>(
      locally_owned_dofs_, ghost_indices_, d_mpi_comm_domain);
  d_potParentQuadDataForce.resize(d_numSpins);
  d_potParentQuadDataSolveEigen.resize(d_numSpins);

  const dealii::Quadrature<3> &quadratureRuleChild =
      d_matrixFreeDataChild->get_quadrature(d_matrixFreeQuadratureComponentPot);
  const unsigned int numQuadraturePointsPerCellChild =
      quadratureRuleChild.size();

  d_mapQuadIdsToProcId.resize(d_numLocallyOwnedCellsChild *
                              numQuadraturePointsPerCellChild);

  for (dftfe::size_type i = 0;
       i < d_numLocallyOwnedCellsChild * numQuadraturePointsPerCellChild; i++) {
    d_mapQuadIdsToProcId.data()[i] = i;
  }

  d_basisOperationsParentHostPtr[d_matrixFreeAdjointVectorComponent]->reinit(
      d_numEigenValues, d_numCellBlockSizeParent,
      d_matrixFreeQuadratureComponentAdjointRhs, true, false);

  d_sumPsiAdjointChildQuadPartialDataMemorySpace.resize(
      d_numLocallyOwnedCellsChild * numQuadraturePointsPerCellChild);
  //    d_parentCellJxW =
  //    d_basisOperationsParentHostPtr[d_matrixFreeAdjointVectorComponent]->JxW();

  //    d_solutionPotVecForWritingInParentNodesMFVec.resize(d_numSpins);
  //    d_solutionPotVecForWritingInParentNodes.resize(d_numSpins);
  //    for (unsigned int iSpin = 0; iSpin < d_numSpins; iSpin++)
  //      {
  //        vectorTools::createDealiiVector<double>(
  //          d_matrixFreeDataParent->get_vector_partitioner(
  //            d_matrixFreePsiVectorComponent),
  //          1,
  //          d_solutionPotVecForWritingInParentNodesMFVec[iSpin]);
  //        d_solutionPotVecForWritingInParentNodes[iSpin].reinit(tempVec);
  //      }

  if (isComputeShapeFunction) {
    preComputeChildShapeFunction();
    preComputeParentJxW();
  }

  //d_kohnShamClass->resetExtPotHamFlag();
  //d_kohnShamClass->setVEffExternalPotCorrToZero();
  //d_kohnShamClass->computeCellHamiltonianMatrixExtPotContribution();

  const dealii::Quadrature<3> &quadratureRuleParent =
      d_matrixFreeDataParent->get_quadrature(
          d_matrixFreeQuadratureComponentAdjointRhs);
  const unsigned int numQuadraturePointsPerCellParent =
      quadratureRuleParent.size();

  d_uValsHost.resize(d_numLocallyOwnedCellsParent *
                     numQuadraturePointsPerCellParent);
  d_uValsMemSpace.resize(d_numLocallyOwnedCellsParent *
                         numQuadraturePointsPerCellParent);

  rhoValues.resize(
      d_numSpins,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
          numQuadraturePointsPerCellParent * d_numLocallyOwnedCellsParent));

  rhoValuesSpinPolarized.resize(
      d_numSpins,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
          numQuadraturePointsPerCellParent * d_numLocallyOwnedCellsParent));

  gradRhoValues.resize(
      d_numSpins,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
          3.0 * numQuadraturePointsPerCellParent *
          d_numLocallyOwnedCellsParent));

  gradRhoValuesSpinPolarized.resize(
      d_numSpins,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
          3.0 * numQuadraturePointsPerCellParent *
          d_numLocallyOwnedCellsParent));
  rhoDiff.resize(
      d_numSpins,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
          d_numLocallyOwnedCellsParent * numQuadraturePointsPerCellParent));
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTSolverFunction<FEOrder, FEOrderElectro, memorySpace>::
    writeVxcDataToFile(std::vector<dftfe::distributedCPUVec<double>> &pot,
                       unsigned int counter) {
  const unsigned int poolId =
      dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_interpool);
  const unsigned int bandGroupId =
      dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_interband);
  const unsigned int minPoolId =
      dealii::Utilities::MPI::min(poolId, d_mpi_comm_interpool);
  const unsigned int minBandGroupId =
      dealii::Utilities::MPI::min(bandGroupId, d_mpi_comm_interband);

  if (poolId == minPoolId && bandGroupId == minBandGroupId) {
    auto local_range = pot[0].locally_owned_elements();
    std::map<dealii::types::global_dof_index, dealii::Point<3, double>>
        dof_coord_child;
    dealii::DoFTools::map_dofs_to_support_points<3, 3>(
        dealii::MappingQ1<3, 3>(), *d_dofHandlerChild, dof_coord_child);
    dealii::types::global_dof_index numberDofsChild =
        d_dofHandlerChild->n_dofs();

    std::vector<std::shared_ptr<dftfe::dftUtils::CompositeData>> data(0);

    const std::string filename = d_inverseDFTParams->vxcDataFolder + "/" +
                                 d_inverseDFTParams->fileNameWriteVxcPostFix +
                                 "_" + std::to_string(counter);
    for (dealii::types::global_dof_index iNode = 0; iNode < numberDofsChild;
         iNode++) {
      if (local_range.is_element(iNode)) {
        std::vector<double> nodeVals(0);
        nodeVals.push_back(iNode);
        nodeVals.push_back(dof_coord_child[iNode][0]);
        nodeVals.push_back(dof_coord_child[iNode][1]);
        nodeVals.push_back(dof_coord_child[iNode][2]);

        nodeVals.push_back(pot[0][iNode]);
        if (d_numSpins == 2) {
          nodeVals.push_back(pot[1][iNode]);
        }
        data.push_back(std::make_shared<dftfe::dftUtils::NodalData>(nodeVals));
      }
    }
    std::vector<dftfe::dftUtils::CompositeData *> dataRawPtrs(data.size());
    for (unsigned int i = 0; i < data.size(); ++i)
      dataRawPtrs[i] = data[i].get();
    dftfe::dftUtils::MPIWriteOnFile().writeData(dataRawPtrs, filename,
                                                d_mpi_comm_domain);
  }
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTSolverFunction<FEOrder, FEOrderElectro, memorySpace>::
    getForceVector(std::vector<dftfe::distributedCPUVec<double>> &pot,
                   std::vector<dftfe::distributedCPUVec<double>> &force,
                   std::vector<double> &loss) {
  d_computingTimerStandard.reset();
#if defined(DFTFE_WITH_DEVICE)
  if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
    dftfe::utils::deviceSynchronize();
#endif
  MPI_Barrier(d_mpi_comm_domain);
  d_computingTimerStandard.enter_subsection("Get Force Vector");
  pcout << "Inside force vector \n";
  for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin) {
    pot[iSpin].update_ghost_values();
    // d_constraintsMatrixDataInfoPot.distribute(pot[iSpin], 1);
    d_constraintMatrixPot->distribute(pot[iSpin]);
    pot[iSpin].update_ghost_values();
  }
#if defined(DFTFE_WITH_DEVICE)
  if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
    dftfe::utils::deviceSynchronize();
#endif
  MPI_Barrier(d_mpi_comm_domain);
  d_computingTimerStandard.enter_subsection("SolveEigen in inverse call");
  this->solveEigen(pot);
#if defined(DFTFE_WITH_DEVICE)
  if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
    dftfe::utils::deviceSynchronize();
#endif
  MPI_Barrier(d_mpi_comm_domain);
  d_computingTimerStandard.leave_subsection("SolveEigen in inverse call");
  const dftfe::utils::MemoryStorage<dftfe::dataTypes::number, memorySpace>
      &eigenVectorsMemSpace = d_dftClassPtr->getEigenVectors();

  const std::vector<std::vector<double>> &eigenValuesHost =
      d_dftClassPtr->getEigenValues();
  const double fermiEnergy = d_dftClassPtr->getFermiEnergy();
  unsigned int numLocallyOwnedDofs = d_dofHandlerParent->n_locally_owned_dofs();
  unsigned int numDofsPerCell = d_dofHandlerParent->get_fe().dofs_per_cell;

  const dealii::Quadrature<3> &quadratureRuleParent =
      d_matrixFreeDataParent->get_quadrature(
          d_matrixFreeQuadratureComponentAdjointRhs);
  const unsigned int numQuadraturePointsPerCellParent =
      quadratureRuleParent.size();

  const dealii::Quadrature<3> &quadratureRuleChild =
      d_matrixFreeDataChild->get_quadrature(d_matrixFreeQuadratureComponentPot);
  const unsigned int numQuadraturePointsPerCellChild =
      quadratureRuleChild.size();
  const unsigned int numTotalQuadraturePointsChild =
      d_numLocallyOwnedCellsChild * numQuadraturePointsPerCellChild;

  typename dealii::DoFHandler<3>::active_cell_iterator cellPtr =
      d_dofHandlerParent->begin_active();
  typename dealii::DoFHandler<3>::active_cell_iterator endcellPtr =
      d_dofHandlerParent->end();

  //
  // @note: Assumes:
  // 1. No constraint magnetization and hence the fermi energy up and down are
  // set to the same value (=fermi energy)
  // 2. No spectrum splitting
  //
#if defined(DFTFE_WITH_DEVICE)
  if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
    dftfe::utils::deviceSynchronize();
#endif
  MPI_Barrier(d_mpi_comm_domain);
  d_computingTimerStandard.enter_subsection("Calculating Density");
  dftfe::computeRhoFromPSI<dftfe::dataTypes::number>(
      &eigenVectorsMemSpace, &eigenVectorsMemSpace, d_numEigenValues,
      d_numEigenValues, eigenValuesHost, fermiEnergy,
      fermiEnergy, // fermi energy up
      fermiEnergy, // fermi energy down
      d_basisOperationsParentPtr[d_matrixFreePsiVectorComponent],
      d_BLASWrapperPtr,
      d_matrixFreePsiVectorComponent,            // matrixFreeDofhandlerIndex
      d_matrixFreeQuadratureComponentAdjointRhs, // quadratureIndex
      d_kpointWeights, rhoValues, gradRhoValues, false, d_mpi_comm_parent,
      d_mpi_comm_interpool, d_mpi_comm_interband, *d_dftParams,
      false // spectrum splitting
  );

#if defined(DFTFE_WITH_DEVICE)
  if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
    dftfe::utils::deviceSynchronize();
#endif
  MPI_Barrier(d_mpi_comm_domain);
  d_computingTimerStandard.leave_subsection("Calculating Density");

  force.resize(d_numSpins);

  //
  // @note: d_rhoTargetQuadDataHost for spin unpolarized case stores only the
  // rho_up (which is also equals tp rho_down) and not the total rho.
  // Accordingly, while taking the difference with the KS rho, we use half of
  // the total KS rho
  //

  cellPtr = d_dofHandlerParent->begin_active();
  endcellPtr = d_dofHandlerParent->end();
  for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin) {
    if (d_numSpins == 1) {
      unsigned int iCell = 0;
      for (; cellPtr != endcellPtr; ++cellPtr) {
        if (cellPtr->is_locally_owned()) {
          for (unsigned int iQuad = 0; iQuad < numQuadraturePointsPerCellParent;
               ++iQuad) {
            rhoDiff[iSpin]
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad] =
                (d_rhoTargetQuadDataHost[iSpin]
                     .data()[iCell * numQuadraturePointsPerCellParent + iQuad] -
                 0.5 * rhoValues[iSpin]
                           .data()[iCell * numQuadraturePointsPerCellParent +
                                   iQuad]);
          }
          iCell++;
        }
      }
    } else {
      unsigned int iCell = 0;
      for (; cellPtr != endcellPtr; ++cellPtr) {
        if (cellPtr->is_locally_owned()) {
          for (unsigned int iQuad = 0; iQuad < numQuadraturePointsPerCellParent;
               ++iQuad) {
            // TODO check the spin polarised case
            rhoDiff[iSpin]
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad] =
                (d_rhoTargetQuadDataHost[iSpin]
                     .data()[iCell * numQuadraturePointsPerCellParent + iQuad] -
                 rhoValuesSpinPolarized[iSpin]
                     .data()[iCell * numQuadraturePointsPerCellParent +
                             iQuad]); // TODO check the spin polarised case
          }
          iCell++;
        }
      }
    }
  }

  std::vector<double> lossUnWeighted(d_numSpins, 0.0);
  std::vector<double> errorInVxc(d_numSpins, 0.0);

  for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin) {
    d_computingTimerStandard.enter_subsection("Create Force Vector");
#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
    MPI_Barrier(d_mpi_comm_domain);
    dftfe::vectorTools::createDealiiVector<double>(
        d_matrixFreeDataChild->get_vector_partitioner(
            d_matrixFreePotVectorComponent),
        1, force[iSpin]);

    force[iSpin] = 0.0;

#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
    MPI_Barrier(d_mpi_comm_domain);
    d_computingTimerStandard.leave_subsection("Create Force Vector");

    sumPsiAdjointChildQuadData.resize(numTotalQuadraturePointsChild);
    std::fill(sumPsiAdjointChildQuadData.begin(),
              sumPsiAdjointChildQuadData.end(), 0.0);
    sumPsiAdjointChildQuadDataPartial.resize(numTotalQuadraturePointsChild);
    std::fill(sumPsiAdjointChildQuadDataPartial.begin(),
              sumPsiAdjointChildQuadDataPartial.end(), 0.0);

    loss[iSpin] = 0.0;
    errorInVxc[iSpin] = 0.0;

#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
    MPI_Barrier(d_mpi_comm_domain);
    d_computingTimerStandard.enter_subsection("Interpolate To Parent Mesh");
    d_transferDataPtr->interpolateMesh2DataToMesh1QuadPoints(
        d_BLASWrapperHostPtr, pot[iSpin], 1, d_fullFlattenedMapChild,
        d_potParentQuadDataForce[iSpin],
        d_resizeMemSpaceVecDuringInterpolation);
#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
    MPI_Barrier(d_mpi_comm_domain);
    d_computingTimerStandard.leave_subsection("Interpolate To Parent Mesh");

    d_computingTimerStandard.enter_subsection("Compute Rho vectors");
    for (unsigned int iCell = 0; iCell < d_numLocallyOwnedCellsParent;
         iCell++) {
      for (unsigned int iQuad = 0; iQuad < numQuadraturePointsPerCellParent;
           ++iQuad) {
        d_uValsHost.data()[iCell * numQuadraturePointsPerCellParent + iQuad] =
            rhoDiff[iSpin]
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad] *
            d_weightQuadDataHost[iSpin]
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad];

        lossUnWeighted[iSpin] +=
            rhoDiff[iSpin]
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad] *
            rhoDiff[iSpin]
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad] *
            d_parentCellJxW[iCell * numQuadraturePointsPerCellParent + iQuad];

        loss[iSpin] +=
            rhoDiff[iSpin]
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad] *
            rhoDiff[iSpin]
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad] *
            d_weightQuadDataHost[iSpin]
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad] *
            d_parentCellJxW[iCell * numQuadraturePointsPerCellParent + iQuad];
      }
    }

    d_uValsMemSpace.copyFrom(d_uValsHost);

    const unsigned int defaultBlockSize = d_dftParams->chebyWfcBlockSize;

#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
    MPI_Barrier(d_mpi_comm_domain);
    d_computingTimerStandard.leave_subsection("Compute Rho vectors");
    for (unsigned int iKPoint = 0; iKPoint < d_numKPoints; ++iKPoint) {
      //            pcout << " kpoint loop before adjoint " << iKPoint
      //                  << " forceVector\n";
      d_kohnShamClass->reinitkPointSpinIndex(iKPoint, iSpin);
      unsigned int jvec = 0;
      while (jvec < d_numEigenValues) {
        d_computingTimerStandard.enter_subsection("Initialize Block Vectors");

        unsigned int currentBlockSize =
            std::min(defaultBlockSize, d_numEigenValues - jvec);

        bool acceptCurrentBlockSize = false;

        while (!acceptCurrentBlockSize) {
          //
          // check if last vector of this block and first vector of
          // next block are degenerate
          //
          unsigned int idThisBlockLastVec = jvec + currentBlockSize - 1;
          if (idThisBlockLastVec + 1 != d_numEigenValues) {
            const double diffEigen =
                std::abs(eigenValuesHost[iKPoint][d_numEigenValues * iSpin +
                                                  idThisBlockLastVec] -
                         eigenValuesHost[iKPoint][d_numEigenValues * iSpin +
                                                  idThisBlockLastVec + 1]);
            if (diffEigen < d_degeneracyTol) {
              currentBlockSize--;
            } else {
              acceptCurrentBlockSize = true;
            }
          } else {
            acceptCurrentBlockSize = true;
          }
        }

        if (currentBlockSize != d_previousBlockSize) {

          pcout << " block size not matching in inverse DFT Solver functions\n";

          dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
              d_matrixFreeDataParent->get_vector_partitioner(
                  d_matrixFreePsiVectorComponent),
              currentBlockSize, psiBlockVecMemSpace);

          dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
              d_matrixFreeDataParent->get_vector_partitioner(
                  d_matrixFreePsiVectorComponent),
              currentBlockSize,
              multiVectorAdjointOutputWithPsiConstraintsMemSpace);

          adjointInhomogenousDirichletValuesMemSpace.reinit(
              multiVectorAdjointOutputWithPsiConstraintsMemSpace);

          constraintsMatrixPsiDataInfo.initialize(
              d_matrixFreeDataParent->get_vector_partitioner(
                  d_matrixFreePsiVectorComponent),
              *d_constraintMatrixHomogeneousPsi);

          dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
              d_matrixFreeDataParent->get_vector_partitioner(
                  d_matrixFreeAdjointVectorComponent),
              currentBlockSize,
              multiVectorAdjointOutputWithAdjointConstraintsMemSpace);

          constraintsMatrixAdjointDataInfo.initialize(
              d_matrixFreeDataParent->get_vector_partitioner(
                  d_matrixFreeAdjointVectorComponent),
              *d_constraintMatrixHomogeneousAdjoint);

          std::shared_ptr<
              dftfe::utils::mpi::MPIPatternP2P<dftfe::utils::MemorySpace::HOST>>
              psiVecMPIP2P = std::make_shared<dftfe::utils::mpi::MPIPatternP2P<
                  dftfe::utils::MemorySpace::HOST>>(
                  psiBlockVecMemSpace.getMPIPatternP2P()
                      ->getLocallyOwnedRange(),
                  psiBlockVecMemSpace.getMPIPatternP2P()->getGhostIndices(),
                  d_mpi_comm_domain);

          dftfe::vectorTools::computeCellLocalIndexSetMap(
              psiVecMPIP2P, *d_matrixFreeDataParent,
              d_matrixFreePsiVectorComponent, currentBlockSize,
              fullFlattenedArrayCellLocalProcIndexIdMapPsiHost);

          std::shared_ptr<
              dftfe::utils::mpi::MPIPatternP2P<dftfe::utils::MemorySpace::HOST>>
              adjVecMPIP2P = std::make_shared<dftfe::utils::mpi::MPIPatternP2P<
                  dftfe::utils::MemorySpace::HOST>>(
                  multiVectorAdjointOutputWithAdjointConstraintsMemSpace
                      .getMPIPatternP2P()
                      ->getLocallyOwnedRange(),
                  multiVectorAdjointOutputWithAdjointConstraintsMemSpace
                      .getMPIPatternP2P()
                      ->getGhostIndices(),
                  d_mpi_comm_domain);

          dftfe::vectorTools::computeCellLocalIndexSetMap(
              adjVecMPIP2P, *d_matrixFreeDataParent,
              d_matrixFreeAdjointVectorComponent, currentBlockSize,
              fullFlattenedArrayCellLocalProcIndexIdMapAdjointHost);

          fullFlattenedArrayCellLocalProcIndexIdMapPsiMemSpace.resize(
              fullFlattenedArrayCellLocalProcIndexIdMapPsiHost.size());
          fullFlattenedArrayCellLocalProcIndexIdMapPsiMemSpace.copyFrom(
              fullFlattenedArrayCellLocalProcIndexIdMapPsiHost);

          fullFlattenedArrayCellLocalProcIndexIdMapAdjointMemSpace.resize(
              fullFlattenedArrayCellLocalProcIndexIdMapAdjointHost.size());
          fullFlattenedArrayCellLocalProcIndexIdMapAdjointMemSpace.copyFrom(
              fullFlattenedArrayCellLocalProcIndexIdMapAdjointHost);

          d_resizeMemSpaceBlockSizeChildQuad = true;
        }

#if defined(DFTFE_WITH_DEVICE)
        if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
          dftfe::utils::deviceSynchronize();
#endif
        MPI_Barrier(d_mpi_comm_domain);
        d_computingTimerStandard.leave_subsection("Initialize Block Vectors");

#if defined(DFTFE_WITH_DEVICE)
        if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
          dftfe::utils::deviceSynchronize();
#endif
        MPI_Barrier(d_mpi_comm_domain);
        d_computingTimerStandard.enter_subsection("Set Up MINRES");
        //
        // @note We assume that there is only homogenous Dirichlet BC
        //
        adjointInhomogenousDirichletValuesMemSpace.setValue(0.0);

        multiVectorAdjointOutputWithPsiConstraintsMemSpace.setValue(0.0);

        multiVectorAdjointOutputWithAdjointConstraintsMemSpace.setValue(0.0);

        d_BLASWrapperPtr->stridedCopyToBlockConstantStride(
            currentBlockSize, d_numEigenValues, numLocallyOwnedDofs, jvec,
            eigenVectorsMemSpace.begin() +
                (d_numSpins * iKPoint + iSpin) * d_numEigenValues,
            psiBlockVecMemSpace.begin());

        std::vector<double> effectiveOrbitalOccupancy;
        std::vector<std::vector<unsigned int>> degeneracyMap(0);
        effectiveOrbitalOccupancy.resize(currentBlockSize);
        degeneracyMap.resize(currentBlockSize);
        std::vector<double> shiftValues;
        shiftValues.resize(currentBlockSize);

        for (unsigned int iBlock = 0; iBlock < currentBlockSize; iBlock++) {
          shiftValues[iBlock] =
              eigenValuesHost[iKPoint]
                             [d_numEigenValues * iSpin + iBlock + jvec];

          effectiveOrbitalOccupancy[iBlock] =
              d_fractionalOccupancy[iKPoint]
                                   [d_numEigenValues * iSpin + iBlock + jvec] *
              d_kpointWeights[iKPoint];
        }

        evaluateDegeneracyMap(shiftValues, degeneracyMap, d_degeneracyTol);

        d_multiVectorAdjointProblem.updateInputPsi(
            psiBlockVecMemSpace, effectiveOrbitalOccupancy, d_uValsMemSpace,
            degeneracyMap, shiftValues, currentBlockSize);

        double adjoinTolForThisIteration = 5.0 * d_tolForChebFiltering;
        d_adjointTol = std::min(d_adjointTol, adjoinTolForThisIteration);

#if defined(DFTFE_WITH_DEVICE)
        if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
          dftfe::utils::deviceSynchronize();
#endif
        MPI_Barrier(d_mpi_comm_domain);
        d_computingTimerStandard.leave_subsection("Set Up MINRES");

#if defined(DFTFE_WITH_DEVICE)
        if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
          dftfe::utils::deviceSynchronize();
#endif
        MPI_Barrier(d_mpi_comm_domain);
        d_computingTimerStandard.enter_subsection("MINRES Solve");
        d_multiVectorLinearMINRESSolver.solve(
            d_multiVectorAdjointProblem, d_BLASWrapperPtr,
            multiVectorAdjointOutputWithPsiConstraintsMemSpace,
            adjointInhomogenousDirichletValuesMemSpace, numLocallyOwnedDofs,
            currentBlockSize, d_adjointTol, d_adjointMaxIterations,
            d_dftParams->verbosity,
            true); // distributeFlag

#if defined(DFTFE_WITH_DEVICE)
        if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
          dftfe::utils::deviceSynchronize();
#endif
        MPI_Barrier(d_mpi_comm_domain);
        d_computingTimerStandard.leave_subsection("MINRES Solve");

#if defined(DFTFE_WITH_DEVICE)
        if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
          dftfe::utils::deviceSynchronize();
#endif
        MPI_Barrier(d_mpi_comm_domain);
        d_computingTimerStandard.enter_subsection("copy vec");

        d_BLASWrapperPtr->xcopy(
            currentBlockSize * numLocallyOwnedDofs,
            multiVectorAdjointOutputWithPsiConstraintsMemSpace.data(), 1,
            multiVectorAdjointOutputWithAdjointConstraintsMemSpace.data(), 1);

        multiVectorAdjointOutputWithAdjointConstraintsMemSpace
            .updateGhostValues();
        constraintsMatrixAdjointDataInfo.distribute(
            multiVectorAdjointOutputWithAdjointConstraintsMemSpace);

        multiVectorAdjointOutputWithAdjointConstraintsMemSpace
            .updateGhostValues();
        psiBlockVecMemSpace.updateGhostValues();

#if defined(DFTFE_WITH_DEVICE)
        if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
          dftfe::utils::deviceSynchronize();
#endif
        MPI_Barrier(d_mpi_comm_domain);
        d_computingTimerStandard.leave_subsection("copy vec");
#if defined(DFTFE_WITH_DEVICE)
        if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
          dftfe::utils::deviceSynchronize();
#endif
        MPI_Barrier(d_mpi_comm_domain);
        d_computingTimerStandard.enter_subsection(
            "interpolate parent data to child quad");

        pcout << "d_resizeMemSpaceBlockSizeChildQuad in inverse dft Solver "
                 "func = "
              << d_resizeMemSpaceBlockSizeChildQuad << "\n";

        d_transferDataPtr->interpolateMesh1DataToMesh2QuadPoints(
            d_BLASWrapperPtr, psiBlockVecMemSpace, currentBlockSize,
            fullFlattenedArrayCellLocalProcIndexIdMapPsiMemSpace,
            d_psiChildQuadDataMemorySpace, d_resizeMemSpaceBlockSizeChildQuad);

        d_transferDataPtr->interpolateMesh1DataToMesh2QuadPoints(
            d_BLASWrapperPtr,
            multiVectorAdjointOutputWithAdjointConstraintsMemSpace,
            currentBlockSize,
            fullFlattenedArrayCellLocalProcIndexIdMapAdjointMemSpace,
            d_adjointChildQuadDataMemorySpace,
            d_resizeMemSpaceBlockSizeChildQuad);
#if defined(DFTFE_WITH_DEVICE)
        if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
          dftfe::utils::deviceSynchronize();
#endif
        MPI_Barrier(d_mpi_comm_domain);
        d_computingTimerStandard.leave_subsection(
            "interpolate parent data to child quad");

        d_computingTimerStandard.enter_subsection("Compute P_i Psi_i");
        d_sumPsiAdjointChildQuadPartialDataMemorySpace.setValue(0.0);
        d_BLASWrapperPtr->addVecOverContinuousIndex(
            numTotalQuadraturePointsChild, currentBlockSize,
            d_psiChildQuadDataMemorySpace.data(),
            d_adjointChildQuadDataMemorySpace.data(),
            d_sumPsiAdjointChildQuadPartialDataMemorySpace.data());

        sumPsiAdjointChildQuadDataPartial.copyFrom(
            d_sumPsiAdjointChildQuadPartialDataMemorySpace);

        for (unsigned int iQuad = 0; iQuad < numTotalQuadraturePointsChild;
             ++iQuad) {
          sumPsiAdjointChildQuadData[iQuad] +=
              sumPsiAdjointChildQuadDataPartial[iQuad];
        }
        jvec += currentBlockSize;

        d_previousBlockSize = currentBlockSize;

#if defined(DFTFE_WITH_DEVICE)
        if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
          dftfe::utils::deviceSynchronize();
#endif
        MPI_Barrier(d_mpi_comm_domain);
        d_computingTimerStandard.leave_subsection("Compute P_i Psi_i");
      } // block loop
    }   // kpoint loop

    // Assumes the block size is 1
    // if that changes, change the d_flattenedArrayCellChildCellMap

#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
    MPI_Barrier(d_mpi_comm_domain);
    d_computingTimerStandard.enter_subsection("Integrate With Shape function");
    integrateWithShapeFunctionsForChildData(force[iSpin],
                                            sumPsiAdjointChildQuadData);
#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
    MPI_Barrier(d_mpi_comm_domain);
    d_computingTimerStandard.leave_subsection("Integrate With Shape function");

    pcout << "force norm = " << force[iSpin].l2_norm() << "\n";

    d_constraintMatrixPot->set_zero(force[iSpin]);
    force[iSpin].zero_out_ghosts();
  } // spin loop

  d_computingTimerStandard.enter_subsection("Post process");
  if ((d_getForceCounter % d_inverseDFTParams->writeVxcFrequency == 0) &&
      (d_inverseDFTParams->writeVxcData)) {
    writeVxcDataToFile(pot, d_getForceCounter);
  }
  MPI_Allreduce(MPI_IN_PLACE, &loss[0], d_numSpins, MPI_DOUBLE, MPI_SUM,
                d_mpi_comm_domain);

  MPI_Allreduce(MPI_IN_PLACE, &lossUnWeighted[0], d_numSpins, MPI_DOUBLE,
                MPI_SUM, d_mpi_comm_domain);

  for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin) {
    pcout << " iter = " << d_getForceCounter
          << " loss unweighted = " << lossUnWeighted[iSpin] << "\n";
    pcout << " iter = " << d_getForceCounter
          << " vxc norm = " << pot[iSpin].l2_norm() << "\n";
  }

  d_lossPreviousIteration = loss[0];
  if (d_numSpins == 2) {
    d_lossPreviousIteration = std::min(d_lossPreviousIteration, loss[1]);
  }

  for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin) {
    d_constraintMatrixPot->set_zero(pot[iSpin]);
    pot[iSpin].zero_out_ghosts();
  }
  d_getForceCounter++;
  d_resizeMemSpaceVecDuringInterpolation = false;
  d_resizeMemSpaceBlockSizeChildQuad = false;

  d_computingTimerStandard.leave_subsection("Post process");
#if defined(DFTFE_WITH_DEVICE)
  if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
    dftfe::utils::deviceSynchronize();
#endif
  MPI_Barrier(d_mpi_comm_domain);
  d_computingTimerStandard.leave_subsection("Get Force Vector");
  d_computingTimerStandard.print_summary();
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTSolverFunction<FEOrder, FEOrderElectro, memorySpace>::solveEigen(
    const std::vector<dftfe::distributedCPUVec<double>> &pot) {

  pcout << "Inside solve eigen\n";

  const dealii::Quadrature<3> &quadratureRuleParent =
      d_matrixFreeDataParent->get_quadrature(
          d_matrixFreeQuadratureComponentAdjointRhs);
  const unsigned int numQuadraturePointsPerCellParent =
      quadratureRuleParent.size();
  const unsigned int numTotalQuadraturePoints =
      numQuadraturePointsPerCellParent * d_numLocallyOwnedCellsParent;

  bool isFirstFilteringPass = (d_getForceCounter == 0) ? true : false;
  std::vector<std::vector<std::vector<double>>> residualNorms(
      d_numSpins,
      std::vector<std::vector<double>>(
          d_numKPoints, std::vector<double>(d_numEigenValues, 0.0)));

  double maxResidual = 0.0;
  unsigned int iPass = 0;
  const double chebyTol = d_dftParams->chebyshevTolerance;
  if (d_getForceCounter > 3) {
    double tolPreviousIter = d_tolForChebFiltering;
    d_tolForChebFiltering = std::min(chebyTol, d_lossPreviousIteration / 10.0);
    d_tolForChebFiltering = std::min(d_tolForChebFiltering, tolPreviousIter);
  } else {
    d_tolForChebFiltering = 1e-10;
  }

  for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin) {
    d_computingTimerStandard.enter_subsection(
        "interpolate child data to parent quad");
    d_transferDataPtr->interpolateMesh2DataToMesh1QuadPoints(
        d_BLASWrapperHostPtr, pot[iSpin], 1, d_fullFlattenedMapChild,
        d_potParentQuadDataSolveEigen[iSpin],
        d_resizeMemSpaceVecDuringInterpolation);
    d_computingTimerStandard.leave_subsection(
        "interpolate child data to parent quad");
    std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
    potKSQuadData(
        d_numSpins,
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
            d_numLocallyOwnedCellsParent * numQuadraturePointsPerCellParent));

    for (unsigned int iCell = 0; iCell < d_numLocallyOwnedCellsParent;
         ++iCell) {
      for (unsigned int iQuad = 0; iQuad < numQuadraturePointsPerCellParent;
           ++iQuad) {
        potKSQuadData[iSpin]
            .data()[iCell * numQuadraturePointsPerCellParent + iQuad] =
            d_potBaseQuadDataHost[iSpin]
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad] +
            d_potParentQuadDataSolveEigen
                [iSpin][iCell * numQuadraturePointsPerCellParent + iQuad];
      }
    }

    d_computingTimerStandard.enter_subsection("setVEff inverse");
    d_kohnShamClass->setVEff(potKSQuadData, iSpin);
    d_computingTimerStandard.leave_subsection("setVEff inverse");

    d_computingTimerStandard.enter_subsection("computeHamiltonianMatrix");
    d_kohnShamClass->computeCellHamiltonianMatrix();
    d_computingTimerStandard.leave_subsection("computeHamiltonianMatrix");
  }

  do {
    pcout << " inside iPass of chebFil = " << iPass << "\n";
    for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin) {
      for (unsigned int iKpoint = 0; iKpoint < d_numKPoints; ++iKpoint) {
        const unsigned int kpointSpinId = iSpin * d_numKPoints + iKpoint;
        d_kohnShamClass->reinitkPointSpinIndex(iKpoint, iSpin);
        d_computingTimerStandard.enter_subsection(
            "kohnShamEigenSpaceCompute inverse");

        if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST) {
          d_dftClassPtr->kohnShamEigenSpaceCompute(
              iSpin, iKpoint, *d_kohnShamClass, *d_elpaScala,
              *d_subspaceIterationSolverHost, residualNorms[iSpin][iKpoint],
              true,  // compute residual
              false, // spectrum splitting
              false, // mixed precision
              false  // is first SCF
          );
        }

#ifdef DFTFE_WITH_DEVICE
        if constexpr (memorySpace == dftfe::utils::MemorySpace::DEVICE) {
          d_dftClassPtr->kohnShamEigenSpaceCompute(
              iSpin, iKpoint, *d_kohnShamClass, *d_elpaScala,
              *d_subspaceIterationSolverDevice, residualNorms[iSpin][iKpoint],
              true,  // compute residual
              false, // spectrum splitting
              false, // mixed precision
              false  // is first SCF
          );
        }
#endif
        d_computingTimerStandard.leave_subsection(
            "kohnShamEigenSpaceCompute inverse");
      }
    }

    const std::vector<std::vector<double>> &eigenValuesHost =
        d_dftClassPtr->getEigenValues();
    d_dftClassPtr->compute_fermienergy(eigenValuesHost, d_numElectrons);
    const double fermiEnergy = d_dftClassPtr->getFermiEnergy();
    maxResidual = 0.0;
    for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin) {
      for (unsigned int iKpoint = 0; iKpoint < d_numKPoints; ++iKpoint) {
        // pcout << "compute partial occupancy eigen\n";
        for (unsigned int iEig = 0; iEig < d_numEigenValues; ++iEig) {
          const double eigenValue =
              eigenValuesHost[iKpoint][d_numEigenValues * iSpin + iEig];
          d_fractionalOccupancy[iKpoint][d_numEigenValues * iSpin + iEig] =
              dftfe::dftUtils::getPartialOccupancy(
                  eigenValue, fermiEnergy, dftfe::C_kb, d_dftParams->TVal);

          if (d_dftParams->constraintMagnetization) {
            d_fractionalOccupancy[iKpoint][d_numEigenValues * iSpin + iEig] =
                1.0;
            if (iSpin == 0) {
              if (eigenValue > fermiEnergy) // fermi energy up
                d_fractionalOccupancy[iKpoint]
                                     [d_numEigenValues * iSpin + iEig] = 0.0;
            } else if (iSpin == 1) {
              if (eigenValue > fermiEnergy) // fermi energy down
                d_fractionalOccupancy[iKpoint]
                                     [d_numEigenValues * iSpin + iEig] = 0.0;
            }
          }

          if (d_fractionalOccupancy[iKpoint][d_numEigenValues * iSpin + iEig] >
              d_fractionalOccupancyTol) {
            if (residualNorms[iSpin][iKpoint][iEig] > maxResidual)
              maxResidual = residualNorms[iSpin][iKpoint][iEig];
          }
        }
      }
    }
    iPass++;
  } while (maxResidual > d_tolForChebFiltering && iPass < d_maxChebyPasses);

  pcout << " maxRes = " << maxResidual << " iPass = " << iPass << "\n";
}

// TODO changed for debugging purposes
template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTSolverFunction<FEOrder, FEOrderElectro, memorySpace>::
    setInitialGuess(const std::vector<dftfe::distributedCPUVec<double>> &pot,
                    const std::vector<std::vector<std::vector<double>>>
                        &targetPotValuesParentQuadData) {
  d_pot = pot;
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTSolverFunction<FEOrder, FEOrderElectro,
                              memorySpace>::preComputeChildShapeFunction() {
  // Quadrature for AX multiplication will FEOrderElectro+1
  const dealii::Quadrature<3> &quadratureRuleChild =
      d_matrixFreeDataChild->get_quadrature(d_matrixFreeQuadratureComponentPot);
  const unsigned int numQuadraturePointsPerCellChild =
      quadratureRuleChild.size();
  // std::cout<<" numQuadraturePointsPerCellChild =
  // "<<numQuadraturePointsPerCellChild<<"\n";
  const unsigned int numTotalQuadraturePointsChild =
      d_numLocallyOwnedCellsChild * numQuadraturePointsPerCellChild;

  // std::cout<<" numTotalQuadraturePointsChild =
  // "<<numTotalQuadraturePointsChild<<"\n";
  dealii::FEValues<3> fe_valuesChild(
      d_dofHandlerChild->get_fe(), quadratureRuleChild,
      dealii::update_values | dealii::update_JxW_values);

  const unsigned int numberDofsPerElement =
      d_dofHandlerChild->get_fe().dofs_per_cell;

  // std::cout<<" numberDofsPerElement = "<<numberDofsPerElement<<"\n";
  //
  // resize data members
  //

  d_childCellJxW.resize(numTotalQuadraturePointsChild);

  // std::cout<<" d_childCellJxW = "<<d_childCellJxW.size()<<"\n";
  typename dealii::DoFHandler<3>::active_cell_iterator
      cell = d_dofHandlerChild->begin_active(),
      endc = d_dofHandlerChild->end();
  unsigned int iElem = 0;
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned()) {
      fe_valuesChild.reinit(cell);
      if (iElem == 0) {
        // For the reference cell initalize the shape function values
        d_childCellShapeFunctionValue.resize(numberDofsPerElement *
                                             numQuadraturePointsPerCellChild);

        // std::cout<<" d_childCellShapeFunctionValue =
        // "<<d_childCellShapeFunctionValue.size()<<"\n";
        for (unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode) {
          for (unsigned int q_point = 0;
               q_point < numQuadraturePointsPerCellChild; ++q_point) {
            d_childCellShapeFunctionValue[numQuadraturePointsPerCellChild *
                                              iNode +
                                          q_point] =
                fe_valuesChild.shape_value(iNode, q_point);
          }
        }
      }

      for (unsigned int q_point = 0; q_point < numQuadraturePointsPerCellChild;
           ++q_point) {
        d_childCellJxW[(iElem * numQuadraturePointsPerCellChild) + q_point] =
            fe_valuesChild.JxW(q_point);
      }
      iElem++;
    }
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTSolverFunction<FEOrder, FEOrderElectro,
                              memorySpace>::preComputeParentJxW() {
  // Quadrature for AX multiplication will FEOrderElectro+1
  const dealii::Quadrature<3> &quadratureRuleParent =
      d_matrixFreeDataParent->get_quadrature(
          d_matrixFreeQuadratureComponentAdjointRhs);
  const unsigned int numQuadraturePointsPerCellParent =
      quadratureRuleParent.size();
  const unsigned int numTotalQuadraturePointsParent =
      d_numLocallyOwnedCellsParent * numQuadraturePointsPerCellParent;

  dealii::FEValues<3> fe_valuesParent(
      d_dofHandlerParent->get_fe(), quadratureRuleParent,
      dealii::update_values | dealii::update_JxW_values);

  const unsigned int numberDofsPerElement =
      d_dofHandlerParent->get_fe().dofs_per_cell;

  //
  // resize data members
  //

  d_parentCellJxW.resize(numTotalQuadraturePointsParent);

  typename dealii::DoFHandler<3>::active_cell_iterator
      cell = d_dofHandlerParent->begin_active(),
      endc = d_dofHandlerParent->end();
  unsigned int iElem = 0;
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned()) {
      fe_valuesParent.reinit(cell);

      if (iElem == 0) {
        // For the reference cell initalize the shape function values
        d_shapeFunctionValueParent.resize(numberDofsPerElement *
                                          numQuadraturePointsPerCellParent);

        for (unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode) {
          for (unsigned int q_point = 0;
               q_point < numQuadraturePointsPerCellParent; ++q_point) {
            d_shapeFunctionValueParent[iNode + q_point * numberDofsPerElement] =
                fe_valuesParent.shape_value(iNode, q_point);
          }
        }
      }
      for (unsigned int q_point = 0; q_point < numQuadraturePointsPerCellParent;
           ++q_point) {
        d_parentCellJxW[(iElem * numQuadraturePointsPerCellParent) + q_point] =
            fe_valuesParent.JxW(q_point);
      }
      iElem++;
    }
}
template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTSolverFunction<FEOrder, FEOrderElectro, memorySpace>::
    integrateWithShapeFunctionsForChildData(
        dftfe::distributedCPUVec<double> &outputVec,
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
            &quadInputData) {
  const dealii::Quadrature<3> &quadratureRuleChild =
      d_matrixFreeDataChild->get_quadrature(d_matrixFreeQuadratureComponentPot);
  const unsigned int numQuadraturePointsPerCellChild =
      quadratureRuleChild.size();
  const unsigned int numTotalQuadraturePointsChild =
      d_numLocallyOwnedCellsChild * numQuadraturePointsPerCellChild;

  const double alpha = 1.0;
  const double beta = 0.0;
  const unsigned int inc = 1;
  const unsigned int blockSizeInput = 1;
  char doNotTans = 'N';
  // pcout << " inside integrateWithShapeFunctionsForChildData doNotTans = "
  //      << doNotTans << "\n";

  const unsigned int numberDofsPerElement =
      d_dofHandlerChild->get_fe().dofs_per_cell;

  std::vector<double> cellLevelNodalOutput(numberDofsPerElement);
  std::vector<double> cellLevelQuadInput(numQuadraturePointsPerCellChild);
  std::vector<dealii::types::global_dof_index> localDofIndices(
      numberDofsPerElement);

  typename dealii::DoFHandler<3>::active_cell_iterator
      cell = d_dofHandlerChild->begin_active(),
      endc = d_dofHandlerChild->end();
  unsigned int iElem = 0;
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned()) {
      cell->get_dof_indices(localDofIndices);
      std::fill(cellLevelNodalOutput.begin(), cellLevelNodalOutput.end(), 0.0);

      std::copy(quadInputData.begin() +
                    (iElem * numQuadraturePointsPerCellChild),
                quadInputData.begin() +
                    ((iElem + 1) * numQuadraturePointsPerCellChild),
                cellLevelQuadInput.begin());

      for (unsigned int q_point = 0; q_point < numQuadraturePointsPerCellChild;
           ++q_point) {
        cellLevelQuadInput[q_point] =
            cellLevelQuadInput[q_point] *
            d_childCellJxW[(iElem * numQuadraturePointsPerCellChild) + q_point];
      }

      dftfe::dgemm_(&doNotTans, &doNotTans, &blockSizeInput,
                    &numberDofsPerElement, &numQuadraturePointsPerCellChild,
                    &alpha, &cellLevelQuadInput[0], &blockSizeInput,
                    &d_childCellShapeFunctionValue[0],
                    &numQuadraturePointsPerCellChild, &beta,
                    &cellLevelNodalOutput[0], &blockSizeInput);

      d_constraintMatrixPot->distribute_local_to_global(
          cellLevelNodalOutput, localDofIndices, outputVec);

      iElem++;
    }
  outputVec.compress(dealii::VectorOperation::add);
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
std::vector<dftfe::distributedCPUVec<double>>
InverseDFTSolverFunction<FEOrder, FEOrderElectro,
                         memorySpace>::getInitialGuess() const {
  return d_pot;
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTSolverFunction<FEOrder, FEOrderElectro, memorySpace>::
    setSolution(const std::vector<dftfe::distributedCPUVec<double>> &pot) {
  d_pot = pot;
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTSolverFunction<FEOrder, FEOrderElectro, memorySpace>::dotProduct(
    const dftfe::distributedCPUVec<double> &vec1,
    const dftfe::distributedCPUVec<double> &vec2, unsigned int blockSize,
    std::vector<double> &outputDot) {
  outputDot.resize(blockSize);
  std::fill(outputDot.begin(), outputDot.end(), 0.0);
  for (unsigned int iNode = 0; iNode < vec1.local_size(); iNode++) {
    outputDot[iNode % blockSize] +=
        vec1.local_element(iNode) * vec2.local_element(iNode);
  }
  MPI_Allreduce(MPI_IN_PLACE, &outputDot[0], blockSize, MPI_DOUBLE, MPI_SUM,
                d_mpi_comm_domain);
}

template class InverseDFTSolverFunction<2, 2, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTSolverFunction<3, 3, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTSolverFunction<4, 4, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTSolverFunction<5, 5, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTSolverFunction<6, 6, dftfe::utils::MemorySpace::HOST>;
#ifdef DFTFE_WITH_DEVICE
template class InverseDFTSolverFunction<2, 2,
                                        dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTSolverFunction<3, 3,
                                        dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTSolverFunction<4, 4,
                                        dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTSolverFunction<5, 5,
                                        dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTSolverFunction<6, 6,
                                        dftfe::utils::MemorySpace::DEVICE>;
#endif

} // end of namespace invDFT
