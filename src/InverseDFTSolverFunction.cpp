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
#include <energyCalculator.h>
#include <map>
#include <vector>
#ifdef DFTFE_WITH_DEVICE
#include <DeviceAPICalls.h>
#endif

#include <xc.h>
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
  d_tolForChebFiltering = 1e-8; // starting cheb tolerance
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTSolverFunction<FEOrder, FEOrderElectro, memorySpace>::reinit(
    const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoTargetQuadDataHost,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
    &rhoTargetChildQuad,
    const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &weightQuadDataHost,
    const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &potBaseQuadDataHost,
    std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &vxcLDAQuadData,
    const std::vector<double> &quadCoordinatesParent,
    dftfe::dftClass<memorySpace> &dftClass,
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
    dftfe::KohnShamDFTBaseOperator<memorySpace> &kohnShamClass,
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
    const std::vector<double> &applyDirichletBCForVxcChildQuad,
    const bool isComputeDiagonalA, const bool isComputeShapeFunction,
    const dftfe::dftParameters &dftParams,
    const inverseDFTParameters &inverseDFTParams) {
  d_rhoTargetQuadDataHost = rhoTargetQuadDataHost;
    d_rhoTargetChildQuadHost = rhoTargetChildQuad;
  d_weightQuadDataHost = weightQuadDataHost;
  d_potBaseQuadDataHost = potBaseQuadDataHost;

  d_BLASWrapperHostPtr = BLASWrapperHostPtr;
  d_BLASWrapperPtr = BLASWrapperPtr;
  d_basisOperationsParentPtr = basisOperationsParentPtr;
  d_basisOperationsParentHostPtr = basisOperationsParentHostPtr;
  d_basisOperationsChildPtr = basisOperationsChildPtr;

  d_vxcLDAQuadDataPtr = &vxcLDAQuadData;
  d_quadCoordinatesParentPtr = &quadCoordinatesParent[0];
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

    d_applyDirichletBCForVxcChildQuad = applyDirichletBCForVxcChildQuad;
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
  d_maxChebyPasses = d_inverseDFTParams->maxChebPasses; // This is hard coded
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

  if(d_inverseDFTParams->keepFractionalOccupancyFixed)
  {
	  for( unsigned int iWave = 0; iWave < d_numElectrons/2 -1; iWave++)
	  {
		  d_fractionalOccupancy[0][iWave] = 1.0;
	  }

	  d_fractionalOccupancy[0][d_numElectrons/2 - 1 ] = d_inverseDFTParams->fractionalOccupancyForHomoLevel;
	   d_fractionalOccupancy[0][d_numElectrons/2] = 1.0 - d_inverseDFTParams->fractionalOccupancyForHomoLevel;

  
	   pcout<<" Printing fractional occupanct \n";
	   for(unsigned int iWave = 0; iWave < d_numEigenValues; iWave++)
	   {
		   pcout<<"iWave = "<<iWave <<" orb Occ = "<<d_fractionalOccupancy[0][iWave]<<"\n";
	   }
  }

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
      *d_kohnShamClass, *d_constraintMatrixHomogeneousPsi, d_dftParams->TVal,
      d_matrixFreePsiVectorComponent, d_matrixFreeQuadratureComponentAdjointRhs,
      isComputeDiagonalA);

  d_matrixFreeDataParent =
      &d_basisOperationsParentPtr[d_matrixFreePsiVectorComponent]
           ->matrixFreeData();
  d_matrixFreeDataChild = &d_basisOperationsChildPtr->matrixFreeData();

constraintsMatrixPsiDataInfo.initialize(
        d_matrixFreeDataParent->get_vector_partitioner
        (d_matrixFreePsiVectorComponent),
        *d_constraintMatrixHomogeneousPsi);

dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
        d_matrixFreeDataParent->get_vector_partitioner(
        d_matrixFreePsiVectorComponent),
        d_numEigenValues, psiBlockVecMemSpace);

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
        d_matrixFreePsiVectorComponent, d_numEigenValues,
        fullFlattenedArrayCellLocalProcIndexIdMapPsiHost);

fractionalOccupanciesSqrtHost.resize(d_numEigenValues);
fractionalOccupanciesSqrtMemspace.resize(d_numEigenValues);

fullFlattenedArrayCellLocalProcIndexIdMapPsiMemSpace.resize(
        fullFlattenedArrayCellLocalProcIndexIdMapPsiHost.size());
fullFlattenedArrayCellLocalProcIndexIdMapPsiMemSpace.copyFrom(
        fullFlattenedArrayCellLocalProcIndexIdMapPsiHost);

  dftfe::linearAlgebra::MultiVector<double, dftfe::utils::MemorySpace::HOST>
      dummyPotVec;

  dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      d_matrixFreeDataChild->get_vector_partitioner(
          matrixFreePotVectorComponent),
      1, dummyPotVec);

  std::vector<dftfe::uInt> fullFlattenedMapChild;
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

  for (dftfe::uInt i = 0;
       i < d_numLocallyOwnedCellsChild * numQuadraturePointsPerCellChild; i++) {
    d_mapQuadIdsToProcId.data()[i] = i;
  }

  d_basisOperationsParentHostPtr[d_matrixFreeAdjointVectorComponent]->reinit(
      d_numEigenValues, d_numCellBlockSizeParent,
      d_matrixFreeQuadratureComponentAdjointRhs, true, false);

  d_sumPsiAdjointChildQuadPartialDataMemorySpace.resize(
      d_numLocallyOwnedCellsChild * numQuadraturePointsPerCellChild);

  if (isComputeShapeFunction) {
    preComputeChildShapeFunction();
    preComputeParentJxW();
  }

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

  rhoDiffNoPrefac.resize(
      d_numSpins,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
          d_numLocallyOwnedCellsParent * numQuadraturePointsPerCellParent));

  d_kohnShamClass->resetExtPotHamFlag();
  if ((d_dftParams->isPseudopotential || d_dftParams->smearedNuclearCharges)) {

    d_kohnShamClass->computeVEffExternalPotCorr(d_dftClassPtr->getPseudoVLoc());
  } else {
    d_kohnShamClass->setVEffExternalPotCorrToZero();
  }

  d_tauValues.resize(d_numLocallyOwnedCellsParent *
                     numQuadraturePointsPerCellParent);
  /***
   *
   * computing Vxc LDA
   *
   */

  d_tolForChebFiltering =
                std::min(d_dftParams->chebyshevTolerance,
                         d_inverseDFTParams->initialTolForChebFiltering);
}

// interpolate nodal data to quadrature values using FEEvaluation
//
template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTSolverFunction<FEOrder, FEOrderElectro, memorySpace>::
    interpolateElectroNodalDataToQuadratureDataGeneral(
        const std::shared_ptr<dftfe::basis::FEBasisOperations<
            double, double, dftfe::utils::MemorySpace::HOST>>
            &basisOperationsPtr,
        const unsigned int dofHandlerId, const unsigned int quadratureId,
        const dftfe::distributedCPUVec<double> &nodalField,
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
            &quadratureValueData,
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
            &quadratureGradValueData,
        const bool isEvaluateGradData) {
  basisOperationsPtr->reinit(0, 0, quadratureId, false);
  const unsigned int nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();
  const unsigned int nCells = basisOperationsPtr->nCells();
  quadratureValueData.clear();
  quadratureValueData.resize(nQuadsPerCell * nCells);
  nodalField.update_ghost_values();
  if (isEvaluateGradData) {
    quadratureGradValueData.clear();
    quadratureGradValueData.resize(3 * nQuadsPerCell * nCells);
  }
  dealii::FEEvaluation<
      3, FEOrderElectro,
      C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(), 1, double>
      feEvalObj(basisOperationsPtr->matrixFreeData(), dofHandlerId,
                quadratureId);
  // AssertThrow(nodalField.partitioners_are_globally_compatible(*matrixFreeData.get_vector_partitioner(dofHandlerId)),
  //        dealii::ExcMessage("DFT-FE Error: mismatch in
  //        partitioner/dofHandler."));

  AssertThrow(
      basisOperationsPtr->matrixFreeData()
              .get_quadrature(quadratureId)
              .size() == nQuadsPerCell,
      dealii::ExcMessage("DFT-FE Error: mismatch in quadrature rule usage in "
                         "interpolateNodalDataToQuadratureData."));

  dealii::DoFHandler<3>::active_cell_iterator subCellPtr;
  for (unsigned int cell = 0;
       cell < basisOperationsPtr->matrixFreeData().n_cell_batches(); ++cell) {
    feEvalObj.reinit(cell);
    feEvalObj.read_dof_values_plain(nodalField);

    auto evalFlags = dealii::EvaluationFlags::values;
    if (isEvaluateGradData)
      evalFlags = evalFlags | dealii::EvaluationFlags::gradients;
    feEvalObj.evaluate(evalFlags);

    for (unsigned int iSubCell = 0;
         iSubCell <
         basisOperationsPtr->matrixFreeData().n_active_entries_per_cell_batch(
             cell);
         ++iSubCell) {
      subCellPtr = basisOperationsPtr->matrixFreeData().get_cell_iterator(
          cell, iSubCell, dofHandlerId);
      dealii::CellId subCellId = subCellPtr->id();
      unsigned int cellIndex = basisOperationsPtr->cellIndex(subCellId);

      double *tempVec = quadratureValueData.data() + cellIndex * nQuadsPerCell;

      for (unsigned int q_point = 0; q_point < nQuadsPerCell; ++q_point) {
        tempVec[q_point] = feEvalObj.get_value(q_point)[iSubCell];
      }
      if (isEvaluateGradData) {
        double *tempVec2 =
            quadratureGradValueData.data() + 3 * cellIndex * nQuadsPerCell;

        for (unsigned int q_point = 0; q_point < nQuadsPerCell; ++q_point) {
          const dealii::Tensor<1, 3, dealii::VectorizedArray<double>>
              &gradVals = feEvalObj.get_gradient(q_point);
          tempVec2[3 * q_point + 0] = gradVals[0][iSubCell];
          tempVec2[3 * q_point + 1] = gradVals[1][iSubCell];
          tempVec2[3 * q_point + 2] = gradVals[2][iSubCell];
        }
      }
    }
  }
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTSolverFunction<FEOrder, FEOrderElectro, memorySpace>::
    writeParentMeshQuadDataToFile(
        const std::vector<dftfe::utils::MemoryStorage<
            double, dftfe::utils::MemorySpace::HOST>> &deltaVxcQuadData,
        const std::vector<dftfe::utils::MemoryStorage<
            double, dftfe::utils::MemorySpace::HOST>> &vxcLDAQuadData,
        const double *quadCoords, const std::string fileName) {

  const unsigned int poolId =
      dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_interpool);
  const unsigned int bandGroupId =
      dealii::Utilities::MPI::this_mpi_process(d_mpi_comm_interband);
  const unsigned int minPoolId =
      dealii::Utilities::MPI::min(poolId, d_mpi_comm_interpool);
  const unsigned int minBandGroupId =
      dealii::Utilities::MPI::min(bandGroupId, d_mpi_comm_interband);

  if (poolId == minPoolId && bandGroupId == minBandGroupId) {
    const dealii::Quadrature<3> &quadratureRuleParent =
        d_matrixFreeDataParent->get_quadrature(
            d_matrixFreeQuadratureComponentAdjointRhs);
    const unsigned int numQuadraturePointsPerCellParent =
        quadratureRuleParent.size();

    dealii::types::global_dof_index numberQuadPts =
        numQuadraturePointsPerCellParent * d_numLocallyOwnedCellsParent;

    std::vector<std::shared_ptr<dftfe::dftUtils::CompositeData>> data(0);

    for (dealii::types::global_dof_index iQuad = 0; iQuad < numberQuadPts;
         iQuad++) {
      std::vector<double> quadVals(0);
      quadVals.push_back(iQuad);
      quadVals.push_back(*(quadCoords + iQuad * 3 + 0));
      quadVals.push_back(*(quadCoords + iQuad * 3 + 1));
      quadVals.push_back(*(quadCoords + iQuad * 3 + 2));

      quadVals.push_back(d_inverseDFTParams->factorForLDAVxc *
                             vxcLDAQuadData[0][iQuad] +
                         deltaVxcQuadData[0][iQuad]);
      if (d_numSpins == 2) {
        quadVals.push_back(d_inverseDFTParams->factorForLDAVxc *
                               vxcLDAQuadData[1][iQuad] +
                           deltaVxcQuadData[1][iQuad]);
      }
      data.push_back(std::make_shared<dftfe::dftUtils::NodalData>(quadVals));
    }
    std::vector<dftfe::dftUtils::CompositeData *> dataRawPtrs(data.size());
    for (unsigned int i = 0; i < data.size(); ++i)
      dataRawPtrs[i] = data[i].get();
    dftfe::dftUtils::MPIWriteOnFile().writeData(dataRawPtrs, fileName,
                                                d_mpi_comm_domain);
  }
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTSolverFunction<FEOrder, FEOrderElectro, memorySpace>::
    writeChildMeshDataToFile(
        const std::vector<dftfe::distributedCPUVec<double>> &pot,
        const std::string fileName) {

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
    dftfe::dftUtils::MPIWriteOnFile().writeData(dataRawPtrs, fileName,
                                                d_mpi_comm_domain);
  }
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTSolverFunction<FEOrder, FEOrderElectro, memorySpace>::
    writeVxcDataToFile(const std::vector<dftfe::distributedCPUVec<double>> &pot,
                       const unsigned int counter) {

  const std::string filename = d_inverseDFTParams->vxcDataFolder + "/" +
                               d_inverseDFTParams->fileNameWriteVxcPostFix +
                               "_" + std::to_string(counter);

  writeChildMeshDataToFile(pot, filename);
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

    d_transferDataPtr->interpolateMesh2DataToMesh1QuadPoints(
            d_BLASWrapperHostPtr,
            pot[0], 1,
            d_fullFlattenedMapChild,
            d_potParentQuadDataForce[0], 1, 1, 0,
            d_resizeMemSpaceVecDuringInterpolation);

    d_dftClassPtr->solveWithInputLDAVxc(d_potParentQuadDataForce,d_tolForChebFiltering);

  const std::vector<std::vector<double>> &eigenValuesHost =
      d_dftClassPtr->getEigenValues();
  const double fermiEnergy = d_dftClassPtr->getFermiEnergy();

  for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin)
    for (unsigned int iKPoint = 0; iKPoint < d_numKPoints; ++iKPoint) {
      for (unsigned int iWave = 0; iWave < d_numEigenValues; iWave++) {
        double deriFermi = dftfe::dftUtils::getPartialOccupancyDer(
            eigenValuesHost[iKPoint][d_numEigenValues * iSpin + iWave],
            fermiEnergy, dftfe::C_kb, d_dftParams->TVal);
        // pcout << " derivative of iSpin = " << iSpin << " kPoint = " <<
        // iKPoint
        //      << " iWave = " << iWave << " : " << deriFermi << "\n";
      }
    }
#if defined(DFTFE_WITH_DEVICE)
  if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
    dftfe::utils::deviceSynchronize();
#endif
  MPI_Barrier(d_mpi_comm_domain);
  d_computingTimerStandard.leave_subsection("SolveEigen in inverse call");
  const dftfe::utils::MemoryStorage<dftfe::dataTypes::number, memorySpace>
      &eigenVectorsMemSpace = d_dftClassPtr->getEigenVectors();

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

  auto dftBasisOp = d_dftClassPtr->getBasisOperationsMemSpace();

  std::vector<double> kPointCoords;
  kPointCoords.resize(3, 0.0);

  dftfe::computeRhoFromPSI<dftfe::dataTypes::number, memorySpace>(
      &eigenVectorsMemSpace, d_numEigenValues, d_dftClassPtr->getPartialOccupancies(),
      dftBasisOp,
      // d_basisOperationsParentPtr[d_matrixFreePsiVectorComponent],
      d_BLASWrapperPtr, d_dftClassPtr->getDensityDofHandlerIndex(),
      d_dftClassPtr->getDensityQuadratureId(),
      // d_matrixFreePsiVectorComponent,            // matrixFreeDofhandlerIndex
      // d_matrixFreeQuadratureComponentAdjointRhs, // quadratureIndex
      kPointCoords, d_kpointWeights, rhoValues, gradRhoValues, d_tauValues,
      true, true, d_mpi_comm_parent, d_mpi_comm_interpool, d_mpi_comm_interband,
      *d_dftParams);

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
		  double preFactor = (d_rhoTargetQuadDataHost[iSpin]
                     .data()[iCell * numQuadraturePointsPerCellParent + iQuad])/(d_rhoTargetQuadDataHost[iSpin]
                     .data()[iCell * numQuadraturePointsPerCellParent + iQuad] + d_inverseDFTParams->rhoTolForConstraints );
           rhoDiffNoPrefac[iSpin]
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad] =
                (d_rhoTargetQuadDataHost[iSpin]
                     .data()[iCell * numQuadraturePointsPerCellParent + iQuad] -
                 0.5 * rhoValues[iSpin]
                           .data()[iCell * numQuadraturePointsPerCellParent +
                                   iQuad]);
	   
		  rhoDiff[iSpin]
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad] =
                        rhoDiffNoPrefac[iSpin]
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad]*preFactor;
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
	    //
	    rhoDiffNoPrefac[iSpin]
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad] =
		(rhoValuesSpinPolarized[iSpin]
                        .data()[iCell * numQuadraturePointsPerCellParent +
                                iQuad] -
                                d_rhoTargetQuadDataHost[iSpin]
                     .data()[iCell * numQuadraturePointsPerCellParent + iQuad]
                 );

            rhoDiff[iSpin]
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad] =
                (rhoValuesSpinPolarized[iSpin]
                        .data()[iCell * numQuadraturePointsPerCellParent +
                                iQuad] -
                                d_rhoTargetQuadDataHost[iSpin]
                     .data()[iCell * numQuadraturePointsPerCellParent + iQuad]
                 ); // TODO check the spin polarised case
          }
          iCell++;
        }
      }
    }
  }

  std::vector<double> l1ErrorInDensityWithPrefac(d_numSpins, 0.0);
  std::vector<double> lossWithPrefac(d_numSpins, 0.0);
  std::vector<double> lossUnWeighted(d_numSpins, 0.0);
  std::vector<double> errorInVxc(d_numSpins, 0.0);
  std::vector<double> l1ErrorInDensity(d_numSpins, 0.0);

  double intRhoVEff = 0.0;
  double intRho_targetVeff = 0.0; 
  double intRhoVc = 0.0; 
  double intRho_targetVc = 0.0;
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

    sumPsiAdjointChildQuadDataOldRoute.resize(numTotalQuadraturePointsChild);
    std::fill(sumPsiAdjointChildQuadDataOldRoute.begin(),
              sumPsiAdjointChildQuadDataOldRoute.end(), 0.0);

    loss[iSpin] = 0.0;
    errorInVxc[iSpin] = 0.0;
    l1ErrorInDensity[iSpin] = 0.0;

#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
    MPI_Barrier(d_mpi_comm_domain);
    d_computingTimerStandard.enter_subsection("Interpolate To Parent Mesh");
    /*
      d_transferDataPtr->interpolateMesh2DataToMesh1QuadPoints(
        d_BLASWrapperHostPtr, pot[iSpin], 1, d_fullFlattenedMapChild,
        d_potParentQuadDataForce[iSpin], 1, 1, 0,
        d_resizeMemSpaceVecDuringInterpolation);
     */
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

        l1ErrorInDensity[iSpin] += std::abs(
            rhoDiffNoPrefac[iSpin]
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad] *
            d_parentCellJxW[iCell * numQuadraturePointsPerCellParent + iQuad]);
        
  l1ErrorInDensityWithPrefac[iSpin] += std::abs(
            rhoDiff[iSpin]
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad] *
            d_parentCellJxW[iCell * numQuadraturePointsPerCellParent + iQuad]);

	lossWithPrefac[iSpin] +=
		rhoDiff[iSpin]
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad] *
            rhoDiff[iSpin]
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad] *
            d_parentCellJxW[iCell * numQuadraturePointsPerCellParent + iQuad];

	lossUnWeighted[iSpin] +=
            rhoDiffNoPrefac[iSpin]
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad] *
            rhoDiffNoPrefac[iSpin]
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad] *
            d_parentCellJxW[iCell * numQuadraturePointsPerCellParent + iQuad];

        loss[iSpin] +=
            rhoDiffNoPrefac[iSpin]
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad] *
            rhoDiffNoPrefac[iSpin]
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad] *
            d_weightQuadDataHost[iSpin]
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad] *
            d_parentCellJxW[iCell * numQuadraturePointsPerCellParent + iQuad];
/*
        intRhoVEff +=
            d_potKSQuadData[iSpin]
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad] *
            rhoValues[iSpin]
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad] *
            d_parentCellJxW
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad];
  */
	/*
        intRho_targetVeff += 2.0* 
	d_potKSQuadData[iSpin]
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad] *
            d_rhoTargetQuadDataHost[iSpin]
                     .data()[iCell * numQuadraturePointsPerCellParent + iQuad] *
		d_parentCellJxW
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad];
*/
        intRhoVc += 
	    d_potParentQuadDataForce
                        [iSpin][iCell * numQuadraturePointsPerCellParent + iQuad] *
			rhoValues[iSpin]
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad] *
            d_parentCellJxW
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad];

	intRho_targetVc += 2.0 *
                d_potParentQuadDataForce        
		[iSpin][iCell * numQuadraturePointsPerCellParent + iQuad] * 
		d_rhoTargetQuadDataHost[iSpin]
                     .data()[iCell * numQuadraturePointsPerCellParent + iQuad] *
            d_parentCellJxW
                .data()[iCell * numQuadraturePointsPerCellParent + iQuad];

      }
    }

      dftfe::distributedCPUVec<double> rhoDiffVec;
      dftfe::vectorTools::createDealiiVector<double>(
              d_matrixFreeDataParent->get_vector_partitioner(d_matrixFreePsiVectorComponent),
              1, rhoDiffVec);
      
      auto dftBasisOpHost = d_dftClassPtr->getBasisOperationsHost();
      d_dftClassPtr->l2ProjectionQuadToNodal(
              dftBasisOpHost, *d_constraintMatrixHomogeneousPsi, d_matrixFreePsiVectorComponent,
              d_dftClassPtr->getDensityQuadratureId(), rhoDiff[iSpin], rhoDiffVec);

      rhoDiffVec.update_ghost_values();
      d_constraintMatrixHomogeneousPsi->distribute(rhoDiffVec);
      rhoDiffVec.update_ghost_values();

      dftfe::linearAlgebra::MultiVector<double, memorySpace> psiBlockVecMemSpaceSingleVec;
      std::vector<unsigned int>
              fullFlattenedArrayCellLocalProcIndexIdMapPsiHostSingleVector;

      dftfe::utils::MemoryStorage<unsigned int, dftfe::utils::MemorySpace::HOST>
              fullFlattenedArrayCellLocalProcIndexIdMapPsiHostSingleVectorMemSpace;

      dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
              d_matrixFreeDataParent->get_vector_partitioner(
                      d_matrixFreePsiVectorComponent),
              1, psiBlockVecMemSpaceSingleVec);

      std::shared_ptr<
      dftfe::utils::mpi::MPIPatternP2P<dftfe::utils::MemorySpace::HOST>>
              psiVecMPIP2P = std::make_shared<dftfe::utils::mpi::MPIPatternP2P<
                             dftfe::utils::MemorySpace::HOST>>(
                                     psiBlockVecMemSpaceSingleVec.getMPIPatternP2P()
                                             ->getLocallyOwnedRange(),
                                             psiBlockVecMemSpaceSingleVec.getMPIPatternP2P()->getGhostIndices(),
                                             d_mpi_comm_domain);

      dftfe::vectorTools::computeCellLocalIndexSetMap(
              psiVecMPIP2P, *d_matrixFreeDataParent,
              d_matrixFreePsiVectorComponent, 1,
              fullFlattenedArrayCellLocalProcIndexIdMapPsiHostSingleVector);
      fullFlattenedArrayCellLocalProcIndexIdMapPsiHostSingleVectorMemSpace.
      resize(fullFlattenedArrayCellLocalProcIndexIdMapPsiHostSingleVector.size());

        fullFlattenedArrayCellLocalProcIndexIdMapPsiHostSingleVectorMemSpace.copyFrom
        (fullFlattenedArrayCellLocalProcIndexIdMapPsiHostSingleVector);

      d_transferDataPtr->interpolateMesh1DataToMesh2QuadPoints(
              d_BLASWrapperHostPtr, rhoDiffVec, 1,
              fullFlattenedArrayCellLocalProcIndexIdMapPsiHostSingleVectorMemSpace,
              sumPsiAdjointChildQuadDataOldRoute, 1, //TODO this is hard coded to cpus.
              1, 0,
              true);


      const std::vector<std::vector<double>> &
      partialOccupanciesHost = d_dftClassPtr->getPartialOccupancies();


      for( unsigned int iWave = 0; iWave< d_numEigenValues; iWave++)
      {
          fractionalOccupanciesSqrtHost[iWave] =
                  std::sqrt(partialOccupanciesHost[0][iWave])
                  - 1.0;
      }

      psiBlockVecMemSpace.setValue(0.0);
      fractionalOccupanciesSqrtMemspace.copyFrom(fractionalOccupanciesSqrtHost);
      d_BLASWrapperPtr->stridedCopyToBlockConstantStride(
              d_numEigenValues, d_numEigenValues, numLocallyOwnedDofs, 0,
              eigenVectorsMemSpace.begin(),
              psiBlockVecMemSpace.begin());

      psiBlockVecMemSpace.updateGhostValues();
      constraintsMatrixPsiDataInfo.distribute(psiBlockVecMemSpace);
      psiBlockVecMemSpace.updateGhostValues();
      d_psiChildQuadDataMemorySpace.setValue(0.0);
      d_transferDataPtr->interpolateMesh1DataToMesh2QuadPoints(
              d_BLASWrapperPtr, psiBlockVecMemSpace, d_numEigenValues,
              fullFlattenedArrayCellLocalProcIndexIdMapPsiMemSpace,
              d_psiChildQuadDataMemorySpace, d_numEigenValues,
              d_numEigenValues, 0,
              d_resizeMemSpaceBlockSizeChildQuad);

      d_BLASWrapperPtr->stridedBlockScaleAndAddColumnWise(
              d_numEigenValues,
              numTotalQuadraturePointsChild,
              d_psiChildQuadDataMemorySpace.data(),
              fractionalOccupanciesSqrtMemspace.data(),
      d_psiChildQuadDataMemorySpace.data());


      d_sumPsiAdjointChildQuadPartialDataMemorySpace.setValue(0.0);
      d_BLASWrapperPtr->addVecOverContinuousIndex(
              numTotalQuadraturePointsChild, d_numEigenValues,
              d_psiChildQuadDataMemorySpace.data(),
              d_psiChildQuadDataMemorySpace.data(),
              d_sumPsiAdjointChildQuadPartialDataMemorySpace.data());

      sumPsiAdjointChildQuadDataPartial.copyFrom(
              d_sumPsiAdjointChildQuadPartialDataMemorySpace);

      double diffIntegChild = 0.0; 
      double rhoInputIntegChild = 0.0;
      double rhoTargetIntegChild = 0.0;
      double l1errorNewRoute = 0.0;
      double l2errorNewRoute = 0.0;
      for(unsigned int iPoint = 0; iPoint < sumPsiAdjointChildQuadData.size(); iPoint++)
      {
	      double preFactor = (d_rhoTargetChildQuadHost[iPoint])/(d_rhoTargetChildQuadHost[iPoint] + 10.0*d_inverseDFTParams->rhoTolForConstraints );

	      sumPsiAdjointChildQuadData[iPoint] = d_applyDirichletBCForVxcChildQuad[iPoint]*(d_rhoTargetChildQuadHost[iPoint] - sumPsiAdjointChildQuadDataPartial[iPoint])*2.0;
	      diffIntegChild += sumPsiAdjointChildQuadData[iPoint]*d_childCellJxW[iPoint] ;
	      rhoInputIntegChild += sumPsiAdjointChildQuadDataPartial[iPoint]*d_childCellJxW[iPoint];
	      rhoTargetIntegChild+= d_rhoTargetChildQuadHost[iPoint]*d_childCellJxW[iPoint];
      
	      l1errorNewRoute += std::abs(sumPsiAdjointChildQuadDataOldRoute[iPoint] - sumPsiAdjointChildQuadData[iPoint])*d_childCellJxW[iPoint] ;
	      l2errorNewRoute += (sumPsiAdjointChildQuadDataOldRoute[iPoint] - sumPsiAdjointChildQuadData[iPoint])*
		      (sumPsiAdjointChildQuadDataOldRoute[iPoint] - sumPsiAdjointChildQuadData[iPoint])*
		      d_childCellJxW[iPoint];
      
      }

     MPI_Allreduce(MPI_IN_PLACE, &diffIntegChild, 1, MPI_DOUBLE, MPI_SUM,
                d_mpi_comm_domain);
    MPI_Allreduce(MPI_IN_PLACE, &rhoInputIntegChild, 1, MPI_DOUBLE, MPI_SUM,
                d_mpi_comm_domain);
   MPI_Allreduce(MPI_IN_PLACE, &rhoTargetIntegChild, 1, MPI_DOUBLE, MPI_SUM,
                d_mpi_comm_domain);

   MPI_Allreduce(MPI_IN_PLACE, &l1errorNewRoute, 1, MPI_DOUBLE, MPI_SUM,
                d_mpi_comm_domain);
   MPI_Allreduce(MPI_IN_PLACE, &l2errorNewRoute, 1, MPI_DOUBLE, MPI_SUM,
                d_mpi_comm_domain);

   l2errorNewRoute = std::sqrt(l2errorNewRoute);

   pcout<<" diffIntegChild = "<<diffIntegChild<<"\n";
   pcout<<" rhoInputIntegChild = "<<rhoInputIntegChild<<"\n";
   pcout<<" rhoTargetIntegChild = "<<rhoTargetIntegChild<<"\n";

   pcout<<" l1errorNewRoute = "<<l1errorNewRoute<<"\n";
   pcout<<" l2errorNewRoute = "<<l2errorNewRoute<<"\n";

      d_uValsMemSpace.copyFrom(d_uValsHost);

    const unsigned int defaultBlockSize = d_dftParams->chebyWfcBlockSize;

#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
    MPI_Barrier(d_mpi_comm_domain);
    d_computingTimerStandard.leave_subsection("Compute Rho vectors");

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
    force[iSpin].zero_out_ghost_values();
  } // spin loop

  d_computingTimerStandard.enter_subsection("Post process");
  if ((d_getForceCounter % d_inverseDFTParams->writeVxcFrequency == 0) &&
      (d_inverseDFTParams->writeVxcData)) {
    computeEnergyMetrics();
    writeVxcDataToFile(pot, d_getForceCounter);

    const std::string quadDataFilename =
        d_inverseDFTParams->vxcDataFolder + "/" +
        d_inverseDFTParams->fileNameWriteVxcPostFix + "_denistyParentQuad_" +
        std::to_string(d_getForceCounter);
    /*
          writeParentMeshQuadDataToFile(
                          d_potParentQuadDataSolveEigen,
                          *d_vxcLDAQuadDataPtr,
                          d_quadCoordinatesParentPtr,
                          quadDataFilename);
      */
    // d_dftClassPtr->outputWfc(quadDataFilename);
  }
  MPI_Allreduce(MPI_IN_PLACE, &loss[0], d_numSpins, MPI_DOUBLE, MPI_SUM,
                d_mpi_comm_domain);

  MPI_Allreduce(MPI_IN_PLACE, &lossUnWeighted[0], d_numSpins, MPI_DOUBLE,
                MPI_SUM, d_mpi_comm_domain);

  MPI_Allreduce(MPI_IN_PLACE, &l1ErrorInDensity[0], d_numSpins, MPI_DOUBLE,
                MPI_SUM, d_mpi_comm_domain);

  MPI_Allreduce(MPI_IN_PLACE, &lossWithPrefac[0], d_numSpins, MPI_DOUBLE,
                MPI_SUM, d_mpi_comm_domain);

  MPI_Allreduce(MPI_IN_PLACE, &l1ErrorInDensityWithPrefac[0], d_numSpins, MPI_DOUBLE,
                MPI_SUM, d_mpi_comm_domain);
  MPI_Allreduce(MPI_IN_PLACE, &intRhoVEff, 1, MPI_DOUBLE, MPI_SUM,
                d_mpi_comm_domain);
  MPI_Allreduce(MPI_IN_PLACE, &intRho_targetVeff, 1, MPI_DOUBLE, MPI_SUM,
                d_mpi_comm_domain);
  MPI_Allreduce(MPI_IN_PLACE, &intRhoVc, 1, MPI_DOUBLE, MPI_SUM,
                d_mpi_comm_domain);
  MPI_Allreduce(MPI_IN_PLACE, &intRho_targetVc, 1, MPI_DOUBLE, MPI_SUM,
                d_mpi_comm_domain);
  
  for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin) {
    pcout << " iter = " << d_getForceCounter
          << " loss unweighted = " << lossUnWeighted[iSpin] << "\n";
    pcout << " intRhoVEff = " << intRhoVEff << "\n";
    pcout << " intRho_targetVeff = "<<intRho_targetVeff<<"\n";
    pcout << " intRhoVc = "<<intRhoVc<<"\n";
    pcout << " intRho_targetVc = "<<intRho_targetVc<<"\n";
    pcout << " iter = " << d_getForceCounter
          << " l1 error = " << l1ErrorInDensity[iSpin] << "\n";
    pcout << " iter = " << d_getForceCounter
          << " vxc norm = " << pot[iSpin].l2_norm() << "\n";
  pcout<<" iter = " << d_getForceCounter<< " l1ErrorInDensityWithPrefac = "<<l1ErrorInDensityWithPrefac[iSpin]<<"\n";
  pcout<<" lossWithPrefac = "<<lossWithPrefac[iSpin]<<"\n";
  
  }


  d_lossPreviousIteration = loss[0];
 
  const double chebyTol = std::min(d_dftParams->chebyshevTolerance,
                         d_inverseDFTParams->initialTolForChebFiltering);

  double tolPreviousIter = d_tolForChebFiltering;
  d_tolForChebFiltering = std::min(
                chebyTol, d_lossPreviousIteration /
                          d_inverseDFTParams->adaptiveFactorForChebFiltering);
  
  d_tolForChebFiltering = std::min(d_tolForChebFiltering, tolPreviousIter);
  
  pcout<<" Solving the exact exchange to "<<d_tolForChebFiltering<<" tolerance \n";
  if (d_numSpins == 2) {
    d_lossPreviousIteration = std::min(d_lossPreviousIteration, loss[1]);
  }

  for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin) {
    d_constraintMatrixPot->set_zero(pot[iSpin]);
    pot[iSpin].zero_out_ghost_values();
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
} // namespace invDFT

template <unsigned int FEOrder, unsigned int FEOrderElectro,
        dftfe::utils::MemorySpace memorySpace>
void InverseDFTSolverFunction<FEOrder, FEOrderElectro, memorySpace>::solveEigen(
        const std::vector<dftfe::distributedCPUVec<double>> &pot) {

    pcout << "Inside solve eigen\n";

    d_kohnShamClass->setFlagForApplyFock(true);
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
        d_tolForChebFiltering = std::min(
                chebyTol, d_lossPreviousIteration /
                          d_inverseDFTParams->adaptiveFactorForChebFiltering);
        d_tolForChebFiltering = std::min(d_tolForChebFiltering, tolPreviousIter);
    } else {
        d_tolForChebFiltering =
                std::min(d_dftParams->chebyshevTolerance,
                         d_inverseDFTParams->initialTolForChebFiltering);
    }

    pcout << " Chebyshev filtering is solved to " << d_tolForChebFiltering
          << " tolerance \n";

    d_potKSQuadData.resize(
            d_numSpins,
            dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
                    d_numLocallyOwnedCellsParent * numQuadraturePointsPerCellParent));

    for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin) {
        d_computingTimerStandard.enter_subsection(
                "interpolate child data to parent quad");
        d_transferDataPtr->interpolateMesh2DataToMesh1QuadPoints(
                d_BLASWrapperHostPtr, pot[iSpin], 1, d_fullFlattenedMapChild,
                d_potParentQuadDataSolveEigen[iSpin], 1, 1, 0,
                d_resizeMemSpaceVecDuringInterpolation);
        d_computingTimerStandard.leave_subsection(
                "interpolate child data to parent quad");

        for (unsigned int iCell = 0; iCell < d_numLocallyOwnedCellsParent;
             ++iCell) {
            for (unsigned int iQuad = 0; iQuad < numQuadraturePointsPerCellParent;
                 ++iQuad) {
                d_potKSQuadData[iSpin]
                        .data()[iCell * numQuadraturePointsPerCellParent + iQuad] =
                        d_potBaseQuadDataHost[iSpin]
                                .data()[iCell * numQuadraturePointsPerCellParent + iQuad] +
                        d_potParentQuadDataSolveEigen
                        [iSpin][iCell * numQuadraturePointsPerCellParent + iQuad] +
                        d_inverseDFTParams->factorForLDAVxc *
                        (*(d_vxcLDAQuadDataPtr))
                        [iSpin][iCell * numQuadraturePointsPerCellParent + iQuad];
            }
        }

        d_computingTimerStandard.enter_subsection("setVEff inverse");
        d_kohnShamClass->setVEff(d_potKSQuadData, iSpin);
        d_computingTimerStandard.leave_subsection("setVEff inverse");

        d_computingTimerStandard.enter_subsection("computeHamiltonianMatrix");
        d_kohnShamClass->computeCellHamiltonianMatrix();
        d_computingTimerStandard.leave_subsection("computeHamiltonianMatrix");
    }

    double maxResidualAfterACEUpdate = 0.0;
    unsigned int iPassACE = 0;
    do {
        iPass = 0;

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
              0,     // numberRayleighRitzAvoidancePasses
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
            unsigned int homoLevel = 0;
            for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin) {
                for (unsigned int iKpoint = 0; iKpoint < d_numKPoints; ++iKpoint) {
                    // pcout << "compute partial occupancy eigen\n";
                    for (unsigned int iEig = 0; iEig < d_numEigenValues; ++iEig) {
                        const double eigenValue =
                                eigenValuesHost[iKpoint][d_numEigenValues * iSpin + iEig];
                        if(! d_inverseDFTParams->keepFractionalOccupancyFixed)
			{
				d_fractionalOccupancy[iKpoint][d_numEigenValues * iSpin + iEig] =
                                dftfe::dftUtils::getPartialOccupancy(
                                        eigenValue, fermiEnergy, dftfe::C_kb, d_dftParams->TVal);

			}
                        if (eigenValue < fermiEnergy + 1e-3) {
                            if (homoLevel < iEig) {
                                homoLevel = iEig;
                            }
                        }
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

                        if ((d_fractionalOccupancy[iKpoint][d_numEigenValues * iSpin + iEig] >
                             d_fractionalOccupancyTol) ||
                            (iEig <=
                             homoLevel + d_inverseDFTParams->additionalEigenStatesSolved)) {
                            if (residualNorms[iSpin][iKpoint][iEig] > maxResidual)
                                maxResidual = residualNorms[iSpin][iKpoint][iEig];
                        }
                    }
                }
            }
            iPass++;
        } while (maxResidual > d_tolForChebFiltering && iPass < d_maxChebyPasses);

        pcout << " maxRes = " << maxResidual << " iPass = " << iPass << "\n";

        d_kohnShamClass->getExactExchangePtr()->updateACEOperator(d_dftClassPtr->getEigenVectors()
                ,d_fractionalOccupancy,
                                                                  true);

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
              0, // numberRayleighRitzAvoidancePasses
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
        maxResidualAfterACEUpdate = 0.0;
        unsigned int homoLevel = 0;
        for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin) {
            for (unsigned int iKpoint = 0; iKpoint < d_numKPoints; ++iKpoint) {
                // pcout << "compute partial occupancy eigen\n";
                for (unsigned int iEig = 0; iEig < d_numEigenValues; ++iEig) {
                    const double eigenValue =
                            eigenValuesHost[iKpoint][d_numEigenValues * iSpin + iEig];
                    
		    if(! d_inverseDFTParams->keepFractionalOccupancyFixed)
                        {

		    d_fractionalOccupancy[iKpoint][d_numEigenValues * iSpin + iEig] =
                            dftfe::dftUtils::getPartialOccupancy(
                                    eigenValue, fermiEnergy, dftfe::C_kb, d_dftParams->TVal);

			}
                    if (eigenValue < fermiEnergy + 1e-3) {
                        if (homoLevel < iEig) {
                            homoLevel = iEig;
                        }
                    }
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

                    if ((d_fractionalOccupancy[iKpoint][d_numEigenValues * iSpin + iEig] >
                         d_fractionalOccupancyTol) ||
                        (iEig <=
                         homoLevel + d_inverseDFTParams->additionalEigenStatesSolved)) {
                        if (residualNorms[iSpin][iKpoint][iEig] > maxResidualAfterACEUpdate)
                            maxResidualAfterACEUpdate = residualNorms[iSpin][iKpoint][iEig];
                    }
                }
            }
        }

        iPassACE++;
        pcout<<" maxRes ACE at iPass = "<<iPassACE<<" = "<<maxResidualAfterACEUpdate<<"\n";
    } while ((maxResidualAfterACEUpdate > d_tolForChebFiltering*10) && (iPassACE < d_maxChebyPasses));
    pcout << " maxRes ACE = " << maxResidualAfterACEUpdate << " iPass = " << iPassACE << "\n";
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
  for (unsigned int iNode = 0; iNode < vec1.locally_owned_size(); iNode++) {
    outputDot[iNode % blockSize] +=
        vec1.local_element(iNode) * vec2.local_element(iNode);
  }
  MPI_Allreduce(MPI_IN_PLACE, &outputDot[0], blockSize, MPI_DOUBLE, MPI_SUM,
                d_mpi_comm_domain);
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTSolverFunction<FEOrder, FEOrderElectro,
                              memorySpace>::computeEnergyMetrics() {
  // compute KE
  double kineticEnergy = d_dftClassPtr->computeAndPrintKE(d_tauValues[0]);

  // compute electrostatic energy

  double totalElectrostaticEnergy = computeElectrostaticEnergy(rhoValues[0]);

  // compute XC energy

  {
    xc_func_type funcXLDA, funcCLDA;
    int exceptParamX = xc_func_init(&funcXLDA, XC_LDA_X, XC_UNPOLARIZED);
    int exceptParamC = xc_func_init(&funcCLDA, XC_LDA_C_PW, XC_UNPOLARIZED);
    double xcLDAEnergy =
        computeLDAEnergy(rhoValues[0], "LDA-PW", funcXLDA, funcCLDA);
  }

  {
    xc_func_type funcXGGA, funcCGGA;
    int exceptParamX = xc_func_init(&funcXGGA, XC_GGA_X_PBE, XC_UNPOLARIZED);
    int exceptParamC = xc_func_init(&funcCGGA, XC_GGA_C_PBE, XC_UNPOLARIZED);
    double xcGGAEnergy = computeGGAEnergy(rhoValues[0], gradRhoValues[0],
                                          "GGA-PBE", funcXGGA, funcCGGA);
  }

  {
    xc_func_type funcXMGGA, funcCMGGA;
    int exceptParamX =
        xc_func_init(&funcXMGGA, XC_MGGA_X_R2SCAN, XC_UNPOLARIZED);
    int exceptParamC =
        xc_func_init(&funcCMGGA, XC_MGGA_C_R2SCAN, XC_UNPOLARIZED);
    double xcMGGAEnergy =
        computeMGGAEnergy(rhoValues[0], gradRhoValues[0], d_tauValues[0],
                          "MGGA-R2SCAN", funcXMGGA, funcCMGGA);
  }

  {
    xc_func_type funcXMGGA, funcCMGGA;
    int exceptParamX = xc_func_init(&funcXMGGA, XC_MGGA_X_SCAN, XC_UNPOLARIZED);
    int exceptParamC = xc_func_init(&funcCMGGA, XC_MGGA_C_SCAN, XC_UNPOLARIZED);
    double xcMGGAEnergy =
        computeMGGAEnergy(rhoValues[0], gradRhoValues[0], d_tauValues[0],
                          "MGGA-SCAN", funcXMGGA, funcCMGGA);
  }
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
double InverseDFTSolverFunction<FEOrder, FEOrderElectro, memorySpace>::
    computeElectrostaticEnergy(
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
            &totalRhoValues) {
  dftfe::distributedCPUVec<double> vTotElectroNodal;

  auto basisOperationsElectroHost =
      d_dftClassPtr->getBasisOperationsElectroHost();

  auto matrixFreeElectro = d_dftClassPtr->getMatrixFreeDataElectro();

  unsigned int mfVectorComponent = d_dftClassPtr->getElectroDofHandlerIndex();

  unsigned int mfRhsId = d_dftClassPtr->getElectroQuadratureRhsId();

  unsigned int mfAXId = d_dftClassPtr->getElectroQuadratureAxId();

  auto constraintMatrix = d_dftClassPtr->getConstraintsVectorElectro();

  auto atomNodeToChargeMap = d_dftClassPtr->getAtomNodeToChargeMap();

  auto bQuadValuesAllAtoms = d_dftClassPtr->getBQuadValuesAllAtoms();

  unsigned int smearedChargeQuadId =
      d_dftClassPtr->getSmearedChargeQuadratureIdElectro();

  dftfe::vectorTools::createDealiiVector<double>(
      matrixFreeElectro.get_vector_partitioner(mfVectorComponent), 1,
      vTotElectroNodal);

  basisOperationsElectroHost->reinit(1, 1, mfRhsId, true);

  vTotElectroNodal = 0.0;

  dftfe::poissonSolverProblem<FEOrderElectro> poissonSolverObj(
      d_mpi_comm_domain);

  if (d_dftParams->multipoleBoundaryConditions) {
    d_dftClassPtr->computeMultipoleMoments(basisOperationsElectroHost, mfRhsId,
                                           totalRhoValues,
                                           &(bQuadValuesAllAtoms));
    d_dftClassPtr->updatePRefinedConstraints();
  }

  poissonSolverObj.reinit(
      basisOperationsElectroHost, vTotElectroNodal, *constraintMatrix,
      mfVectorComponent, mfRhsId, mfAXId, atomNodeToChargeMap,
      bQuadValuesAllAtoms, smearedChargeQuadId, totalRhoValues,
      true, // isComputeDiagonalA
      d_dftParams->periodicX && d_dftParams->periodicY &&
          d_dftParams->periodicZ &&
          !d_dftParams->pinnedNodeForPBC, // isComputeMeanValueConstraint
      d_dftParams->smearedNuclearCharges, // smearedNuclearCharges
      true,                               // isRhoValues
      false,                              // isGradSmearedChargeRhs
      0,                                  // smearedChargeGradientComponentId
      false, false, true);

  dftfe::dealiiLinearSolver dealiiLinearSolverObj(
      d_mpi_comm_parent, d_mpi_comm_domain, dftfe::dealiiLinearSolver::CG);

  dealiiLinearSolverObj.solve(
      poissonSolverObj, d_dftParams->absLinearSolverTolerance,
      d_dftParams->maxLinearSolverIterations, d_dftParams->verbosity);

  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> dummy;
  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      vTotElectroQuad;
  interpolateElectroNodalDataToQuadratureDataGeneral(
      basisOperationsElectroHost, mfVectorComponent, mfRhsId, vTotElectroNodal,
      vTotElectroQuad, dummy);

  double electrostaticEnergyTotPot =
      0.5 *
      dftfe::internalEnergy::computeFieldTimesDensity(
          basisOperationsElectroHost, mfRhsId, vTotElectroQuad, totalRhoValues);

  double totalelectrostaticEnergyPot =
      dealii::Utilities::MPI::sum(electrostaticEnergyTotPot, d_mpi_comm_domain);

  const double nuclearElectrostaticEnergy =
      dftfe::internalEnergy::nuclearElectrostaticEnergyLocal(
          vTotElectroNodal, d_dftClassPtr->getLocalVselfs(),
          bQuadValuesAllAtoms, d_dftClassPtr->getbCellNonTrivialAtomIds(),
          basisOperationsElectroHost->getDofHandler(),
          basisOperationsElectroHost->matrixFreeData().get_quadrature(mfRhsId),
          basisOperationsElectroHost->matrixFreeData().get_quadrature(
              d_dftClassPtr->getSmearedChargeQuadratureIdElectro()),
          atomNodeToChargeMap, d_dftParams->smearedNuclearCharges);

  double totalNuclearElectrostaticEnergy = dealii::Utilities::MPI::sum(
      nuclearElectrostaticEnergy, d_mpi_comm_domain);

  const double allElectronElectrostaticEnergy =
      (totalelectrostaticEnergyPot + totalNuclearElectrostaticEnergy);

  pcout<<" Total electrostatic energy without nuclear energy = "<<totalelectrostaticEnergyPot<<"\n";
  pcout << " Total electrostatic energy = " << allElectronElectrostaticEnergy
        << "\n";
  return allElectronElectrostaticEnergy;
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
double InverseDFTSolverFunction<FEOrder, FEOrderElectro, memorySpace>::
    computeLDAEnergy(
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
            &totalRhoValues,
        std::string functional, xc_func_type funcXLDA, xc_func_type funcCLDA) {
  const dealii::Quadrature<3> &quadratureRuleParent =
      d_matrixFreeDataParent->get_quadrature(
          d_matrixFreeQuadratureComponentAdjointRhs);
  const unsigned int numQuadraturePointsPerCellParent =
      quadratureRuleParent.size();

  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      excEnergyDensityVal;
  excEnergyDensityVal.resize(d_numLocallyOwnedCellsParent *
                             numQuadraturePointsPerCellParent);

  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      corrEnergyDensityVal;
  corrEnergyDensityVal.resize(d_numLocallyOwnedCellsParent *
                              numQuadraturePointsPerCellParent);

  xc_lda_exc(&funcXLDA,
             d_numLocallyOwnedCellsParent * numQuadraturePointsPerCellParent,
             totalRhoValues.data(), excEnergyDensityVal.data());

  xc_lda_exc(&funcCLDA,
             d_numLocallyOwnedCellsParent * numQuadraturePointsPerCellParent,
             totalRhoValues.data(), corrEnergyDensityVal.data());

  d_basisOperationsParentHostPtr[d_matrixFreePsiVectorComponent]->reinit(
      d_numEigenValues, d_numCellBlockSizeParent,
      d_matrixFreeQuadratureComponentAdjointRhs, false, false);

  auto JxWData =
      d_basisOperationsParentHostPtr[d_matrixFreePsiVectorComponent]->JxW();

  double ExcEnergyLocal = 0.0;
  for (unsigned int i = 0;
       i < d_numLocallyOwnedCellsParent * numQuadraturePointsPerCellParent;
       i++) {
    ExcEnergyLocal +=
        (totalRhoValues.data()[i]) *
        (excEnergyDensityVal.data()[i] + corrEnergyDensityVal.data()[i]) *
        (JxWData[i]);
  }

  double totalExcEnergy =
      dealii::Utilities::MPI::sum(ExcEnergyLocal, d_mpi_comm_domain);

  pcout << " EXC energy for " << functional << " = " << totalExcEnergy << "\n";

  return totalExcEnergy;
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
double InverseDFTSolverFunction<FEOrder, FEOrderElectro, memorySpace>::
    computeGGAEnergy(
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
            &totalRhoValues,
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
            &totalGradRhoValues,
        std::string functional, xc_func_type funcXGGA, xc_func_type funcCGGA) {
  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      totalSigmaRhoValues;

  const dealii::Quadrature<3> &quadratureRuleParent =
      d_matrixFreeDataParent->get_quadrature(
          d_matrixFreeQuadratureComponentAdjointRhs);
  const unsigned int numQuadraturePointsPerCellParent =
      quadratureRuleParent.size();

  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      excEnergyDensityVal;
  excEnergyDensityVal.resize(d_numLocallyOwnedCellsParent *
                             numQuadraturePointsPerCellParent);

  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      corrEnergyDensityVal;
  corrEnergyDensityVal.resize(d_numLocallyOwnedCellsParent *
                              numQuadraturePointsPerCellParent);

  totalSigmaRhoValues.resize(d_numLocallyOwnedCellsParent *
                             numQuadraturePointsPerCellParent);
  std::fill(totalSigmaRhoValues.begin(), totalSigmaRhoValues.end(), 0.0);

  for (unsigned int i = 0;
       i < d_numLocallyOwnedCellsParent * numQuadraturePointsPerCellParent;
       i++) {
    totalSigmaRhoValues.data()[i] = (totalGradRhoValues.data()[i * 3 + 0] *
                                         totalGradRhoValues.data()[i * 3 + 0] +
                                     totalGradRhoValues.data()[i * 3 + 1] *
                                         totalGradRhoValues.data()[i * 3 + 1] +
                                     totalGradRhoValues.data()[i * 3 + 2] *
                                         totalGradRhoValues.data()[i * 3 + 2]);
  }

  xc_gga_exc(&funcXGGA,
             d_numLocallyOwnedCellsParent * numQuadraturePointsPerCellParent,
             totalRhoValues.data(), totalSigmaRhoValues.data(),
             excEnergyDensityVal.data());

  xc_gga_exc(&funcCGGA,
             d_numLocallyOwnedCellsParent * numQuadraturePointsPerCellParent,
             totalRhoValues.data(), totalSigmaRhoValues.data(),
             corrEnergyDensityVal.data());

  d_basisOperationsParentHostPtr[d_matrixFreePsiVectorComponent]->reinit(
      d_numEigenValues, d_numCellBlockSizeParent,
      d_matrixFreeQuadratureComponentAdjointRhs, false, false);

  auto JxWData =
      d_basisOperationsParentHostPtr[d_matrixFreePsiVectorComponent]->JxW();

  double ExcEnergyLocal = 0.0;
  for (unsigned int i = 0;
       i < d_numLocallyOwnedCellsParent * numQuadraturePointsPerCellParent;
       i++) {
    ExcEnergyLocal +=
        (totalRhoValues.data()[i]) *
        (excEnergyDensityVal.data()[i] + corrEnergyDensityVal.data()[i]) *
        (JxWData[i]);
  }

  double totalExcEnergy =
      dealii::Utilities::MPI::sum(ExcEnergyLocal, d_mpi_comm_domain);

  pcout << " EXC energy for " << functional << " = " << totalExcEnergy << "\n";

  return totalExcEnergy;
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
double InverseDFTSolverFunction<FEOrder, FEOrderElectro, memorySpace>::
    computeMGGAEnergy(
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
            &totalRhoValues,
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
            &totalGradRhoValues,
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
            &kineticEnergyDensityValues,
        std::string functional, xc_func_type funcXMGGA,
        xc_func_type funcCMGGA) {
  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      totalSigmaRhoValues;

  const dealii::Quadrature<3> &quadratureRuleParent =
      d_matrixFreeDataParent->get_quadrature(
          d_matrixFreeQuadratureComponentAdjointRhs);
  const unsigned int numQuadraturePointsPerCellParent =
      quadratureRuleParent.size();

  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      excEnergyDensityVal;
  excEnergyDensityVal.resize(d_numLocallyOwnedCellsParent *
                             numQuadraturePointsPerCellParent);

  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      corrEnergyDensityVal;
  corrEnergyDensityVal.resize(d_numLocallyOwnedCellsParent *
                              numQuadraturePointsPerCellParent);

  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> tauValue;
  tauValue.resize(d_numLocallyOwnedCellsParent *
                  numQuadraturePointsPerCellParent);
  std::fill(tauValue.begin(), tauValue.end(), 0.0);

  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      rhoLaplacianQuadValues;
  rhoLaplacianQuadValues.resize(d_numLocallyOwnedCellsParent *
                                numQuadraturePointsPerCellParent);

  // TODO setting rhoLaplacianQuadValues to zero
  std::fill(rhoLaplacianQuadValues.begin(), rhoLaplacianQuadValues.end(), 0.0);

  totalSigmaRhoValues.resize(d_numLocallyOwnedCellsParent *
                             numQuadraturePointsPerCellParent);
  std::fill(totalSigmaRhoValues.begin(), totalSigmaRhoValues.end(), 0.0);

  for (unsigned int i = 0;
       i < d_numLocallyOwnedCellsParent * numQuadraturePointsPerCellParent;
       i++) {
    totalSigmaRhoValues.data()[i] = (totalGradRhoValues.data()[i * 3 + 0] *
                                         totalGradRhoValues.data()[i * 3 + 0] +
                                     totalGradRhoValues.data()[i * 3 + 1] *
                                         totalGradRhoValues.data()[i * 3 + 1] +
                                     totalGradRhoValues.data()[i * 3 + 2] *
                                         totalGradRhoValues.data()[i * 3 + 2]);
  }

  xc_mgga_exc(&funcXMGGA,
              d_numLocallyOwnedCellsParent * numQuadraturePointsPerCellParent,
              totalRhoValues.data(), totalSigmaRhoValues.data(),
              rhoLaplacianQuadValues.data(), kineticEnergyDensityValues.data(),
              excEnergyDensityVal.data());

  xc_mgga_exc(&funcCMGGA,
              d_numLocallyOwnedCellsParent * numQuadraturePointsPerCellParent,
              totalRhoValues.data(), totalSigmaRhoValues.data(),
              rhoLaplacianQuadValues.data(), kineticEnergyDensityValues.data(),
              corrEnergyDensityVal.data());

  d_basisOperationsParentHostPtr[d_matrixFreePsiVectorComponent]->reinit(
      d_numEigenValues, d_numCellBlockSizeParent,
      d_matrixFreeQuadratureComponentAdjointRhs, false, false);

  auto JxWData =
      d_basisOperationsParentHostPtr[d_matrixFreePsiVectorComponent]->JxW();

  double ExcEnergyLocal = 0.0;
  for (unsigned int i = 0;
       i < d_numLocallyOwnedCellsParent * numQuadraturePointsPerCellParent;
       i++) {
    ExcEnergyLocal +=
        (totalRhoValues.data()[i]) *
        (excEnergyDensityVal.data()[i] + corrEnergyDensityVal.data()[i]) *
        (JxWData[i]);
  }

  double totalExcEnergy =
      dealii::Utilities::MPI::sum(ExcEnergyLocal, d_mpi_comm_domain);

  pcout << " EXC energy for " << functional << " = " << totalExcEnergy << "\n";

  return totalExcEnergy;
}

template class InverseDFTSolverFunction<2, 2, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTSolverFunction<2, 3, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTSolverFunction<2, 4, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTSolverFunction<3, 3, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTSolverFunction<3, 4, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTSolverFunction<3, 5, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTSolverFunction<3, 6, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTSolverFunction<4, 4, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTSolverFunction<4, 5, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTSolverFunction<4, 6, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTSolverFunction<4, 7, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTSolverFunction<5, 5, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTSolverFunction<5, 6, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTSolverFunction<5, 7, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTSolverFunction<5, 8, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTSolverFunction<6, 6, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTSolverFunction<6, 7, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTSolverFunction<6, 8, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTSolverFunction<6, 9, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTSolverFunction<7, 7, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTSolverFunction<7, 8, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTSolverFunction<7, 9, dftfe::utils::MemorySpace::HOST>;

#ifdef DFTFE_WITH_DEVICE
template class InverseDFTSolverFunction<2, 2,
                                        dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTSolverFunction<2, 3,
                                        dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTSolverFunction<2, 4,
                                        dftfe::utils::MemorySpace::DEVICE>;

template class InverseDFTSolverFunction<3, 3,
                                        dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTSolverFunction<3, 4,
                                        dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTSolverFunction<3, 5,
                                        dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTSolverFunction<3, 6,
                                        dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTSolverFunction<4, 4,
                                        dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTSolverFunction<4, 5,
                                        dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTSolverFunction<4, 6,
                                        dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTSolverFunction<4, 7,
                                        dftfe::utils::MemorySpace::DEVICE>;

template class InverseDFTSolverFunction<5, 5,
                                        dftfe::utils::MemorySpace::DEVICE>;

template class InverseDFTSolverFunction<5, 6,
                                        dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTSolverFunction<5, 7,
                                        dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTSolverFunction<5, 8,
                                        dftfe::utils::MemorySpace::DEVICE>;

template class InverseDFTSolverFunction<6, 6,
                                        dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTSolverFunction<6, 7,
                                        dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTSolverFunction<6, 8,
                                        dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTSolverFunction<6, 9,
                                        dftfe::utils::MemorySpace::DEVICE>;

template class InverseDFTSolverFunction<7, 7,
                                        dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTSolverFunction<7, 8,
                                        dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTSolverFunction<7, 9,
                                        dftfe::utils::MemorySpace::DEVICE>;
#endif

} // end of namespace invDFT
