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

#include "InverseDFTEngine.h"
#include "BFGSInverseDFTSolver.h"
#include "CompositeData.h"
#include "InverseDFTSolverFunction.h"
#include "MPIWriteOnFile.h"
#include "NodalData.h"
#include "dftUtils.h"
#include <densityCalculator.h>
#include <gaussianFunctionManager.h>
#include <xc.h>
namespace invDFT {
namespace {
double realPart(const double x) { return x; }

double realPart(const std::complex<double> x) { return x.real(); }

double complexConj(const double x) { return x; }

std::complex<double> complexConj(const std::complex<double> x) {
  return std::conj(x);
}

float realPart(const float x) { return x; }

float realPart(const std::complex<float> x) { return x.real(); }

float complexConj(const float x) { return x; }

std::complex<float> complexConj(const std::complex<float> x) {
  return std::conj(x);
}

struct coordinateValues {
  double iNode;
  double xcoord;
  double ycoord;
  double zcoord;
  double value0;
  double value1;
};

struct less_than_key {
  inline bool operator()(const coordinateValues &lhs,
                         const coordinateValues &rhs) {
    double tol = 1e-6;
    if (lhs.iNode - rhs.iNode < -tol) {
      return true;
    }
    return false;

    double xdiff = lhs.xcoord - rhs.xcoord;
    double ydiff = lhs.ycoord - rhs.ycoord;
    double zdiff = lhs.zcoord - rhs.zcoord;
    if (xdiff < -tol)
      return true;
    if (xdiff > tol)
      return false;

    if (ydiff < -tol)
      return true;
    if (ydiff > tol)
      return false;

    if (zdiff < -tol)
      return true;
    if (zdiff > tol)
      return false;

    return false;
    // AssertThrow(
    //   (std::abs(xdiff) > tol) || (std::abs(ydiff) > tol) ||
    //   (std::abs(zdiff) > tol), ExcMessage(
    //     "DFT-FE error:  coordinates of two different vertices in Vxc are
    //     close to tol`"));
  }
};

//    auto comp = [](const coordinateValues& lhs, const coordinateValues&
//    rhs){
//      double tol = 1e-6;
//
//      double xdiff = lhs.xcoord - rhs.xcoord;
//      double ydiff = lhs.ycoord - rhs.ycoord;
//      double zdiff = lhs.zcoord - rhs.zcoord;
//      if(xdiff < -tol)
//        return true;
//      if(xdiff > tol)
//        return false;
//
//          if(ydiff < -tol)
//            return true;
//          if(ydiff > tol)
//            return false;
//
//              if(zdiff < -tol)
//                return true;
//              if(zdiff > tol)
//                return false;
//
//              AssertThrow(
//                (std::abs(xdiff) > tol) || (std::abs(ydiff) > tol) ||
//                (std::abs(zdiff) > tol), ExcMessage(
//                  "DFT-FE error:  coordinates of two different vertices in
//                  Vxc are close to tol`"));
//    };

} // namespace

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
InverseDFTEngine<FEOrder, FEOrderElectro, memorySpace>::InverseDFTEngine(
    dftfe::dftBase &dft, dftfe::dftParameters &dftParams,
    inverseDFTParameters &inverseDFTParams, const MPI_Comm &mpi_comm_parent,
    const MPI_Comm &mpi_comm_domain, const MPI_Comm &mpi_comm_bandgroup,
    const MPI_Comm &mpi_comm_interpool)
    : d_mpiComm_domain(mpi_comm_domain), d_mpiComm_parent(mpi_comm_parent),
      d_mpiComm_bandgroup(mpi_comm_bandgroup),
      d_mpiComm_interpool(mpi_comm_interpool),
      n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_domain)),
      this_mpi_process(
          dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain)),
      pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0)),
      d_dftParams(dftParams), d_inverseDFTParams(inverseDFTParams),
      d_triaManagerVxc(mpi_comm_parent, mpi_comm_domain, mpi_comm_interpool,
                       mpi_comm_bandgroup, dftParams, d_inverseDFTParams),
      d_dofHandlerTriaVxc(),
      d_gaussQuadVxc(2) // TODO this hard coded to Gauss 2x2x2 rule which is
                        // sufficient as the vxc mesh is taken to be linear FE.
                        // Read from params file for generality
{

  d_rhoTargetTolForConstraints = d_inverseDFTParams.rhoTolForConstraints;

  d_dftBaseClass =
      ((dftfe::dftClass<FEOrder, FEOrderElectro, memorySpace> *)&dft);

  d_dftMatrixFreeData = &(d_dftBaseClass->getMatrixFreeData());

  d_dftDensityDoFHandlerIndex = d_dftBaseClass->getDensityDofHandlerIndex();

  d_dftQuadIndex = d_dftBaseClass->getDensityQuadratureId();
  d_gaussQuadAdjoint = &(d_dftMatrixFreeData->get_quadrature(d_dftQuadIndex));

  d_numSpins = 1 + d_dftParams.spinPolarized;
  d_kpointWeights = d_dftBaseClass->getKPointWeights();
  d_numEigenValues = d_dftBaseClass->getNumEigenValues();
  d_numKPoints = d_kpointWeights.size();
  // get the triangulation manager of the DFTClass
  d_dftTriaManager = d_dftBaseClass->getTriangulationManager();

  d_dofHandlerDFTClass =
      &d_dftMatrixFreeData->get_dof_handler(d_dftDensityDoFHandlerIndex);
  d_constraintDFTClass = d_dftBaseClass->getDensityConstraint();

  d_dftMatrixFreeDataElectro = &(d_dftBaseClass->getMatrixFreeDataElectro());
  d_dftElectroDoFHandlerIndex = d_dftBaseClass->getElectroDofHandlerIndex();
  d_dofHandlerElectroDFTClass =
      &d_dftMatrixFreeDataElectro->get_dof_handler(d_dftElectroDoFHandlerIndex);
  d_dftElectroRhsQuadIndex = d_dftBaseClass->getElectroQuadratureRhsId();
  d_dftElectroAxQuadIndex = d_dftBaseClass->getElectroQuadratureAxId();

  d_basisOperationsHost = d_dftBaseClass->getBasisOperationsHost();

  d_basisOperationsMemSpace = d_dftBaseClass->getBasisOperationsMemSpace();

  d_basisOperationsElectroHost =
      d_dftBaseClass->getBasisOperationsElectroHost();

  d_basisOperationsElectroMemSpace =
      d_dftBaseClass->getBasisOperationsElectroMemSpace();

  d_blasWrapperMemSpace = d_dftBaseClass->getBLASWrapperMemSpace();

  d_blasWrapperHost = d_dftBaseClass->getBLASWrapperHost();
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
InverseDFTEngine<FEOrder, FEOrderElectro, memorySpace>::~InverseDFTEngine() {
  // delete d_triaManagerVxcPtr;
}
template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTEngine<FEOrder, FEOrderElectro,
                      memorySpace>::createParentChildDofManager() {

  dftfe::dftParameters dftParamsVxc(d_dftParams);

  dftParamsVxc.innerAtomBallRadius = d_inverseDFTParams.VxcInnerDomain;
  dftParamsVxc.meshSizeInnerBall = d_inverseDFTParams.VxcInnerMeshSize;

  dftfe::triangulationManager dftTriaManagerVxc(
      d_mpiComm_parent, d_mpiComm_domain, d_mpiComm_interpool,
      d_mpiComm_bandgroup, 1, dftParamsVxc);

  // TODO does not assume periodic BCs.
  std::vector<std::vector<double>> atomLocations =
      d_dftBaseClass->getAtomLocationsCart();

  const std::vector<std::vector<double>> &imageAtomsCart =
      d_dftBaseClass->getImageAtomLocationsCart();

  const std::vector<int> &imageIdsTrunc = d_dftBaseClass->getImageAtomIDs();

  const std::vector<double> &nearestAtomDist =
      d_dftBaseClass->getNearestAtomDistance();
  const std::vector<std::vector<double>> &domainBound =
      d_dftBaseClass->getCell();

  dftTriaManagerVxc.generateSerialUnmovedAndParallelMovedUnmovedMesh(
      atomLocations, imageAtomsCart, imageIdsTrunc, nearestAtomDist,
      domainBound,
      false); // generateSerialTria

  // const parallel::distributed::Triangulation<3> &parallelMeshUnmoved =
  //   d_dftTriaManager->getParallelMeshUnmoved();
  // const parallel::distributed::Triangulation<3> &parallelMeshMoved =
  //   d_dftTriaManager->getParallelMeshMoved();
  /*
            d_triaManagerVxcPtr = new TriangulationManagerVxc(d_mpiComm_parent,
                         d_mpiComm_domain,
                         d_mpiComm_interpool,
                         d_mpiComm_bandgroup,
                         dftParamsVxc,
                         dealii::parallel::distributed::Triangulation<3,
     3>::default_setting); // set this to
                                                                                               // dealii::parallel::distributed::Triangulation<3, 3>::no_automatic_repartitioning
                                                                                               // If you want no repartitioning
  */

  MPI_Barrier(d_mpiComm_domain);
  double meshStart = MPI_Wtime();
  //
  // @note This is compatible with only non-periodic boundary conditions as
  // imageAtomLocations is not considered
  //
  d_triaManagerVxc.generateParallelUnmovedMeshVxc(atomLocations,
                                                  dftTriaManagerVxc);

  MPI_Barrier(d_mpiComm_domain);
  double meshEnd = MPI_Wtime();
  // TODO this function has been commented out
  // d_triaManagerVxc.generateParallelMovedMeshVxc(parallelMeshUnmoved,
  //                                              parallelMeshMoved);

  // parallelMeshMoved = parallelMeshUnmoved;
  MPI_Barrier(d_mpiComm_domain);
  double meshMoveEnd = MPI_Wtime();
  // const parallel::distributed::Triangulation<3> &parallelMeshMovedVxc =
  //  d_triaManagerVxc.getParallelMovedMeshVxc();

  const parallel::distributed::Triangulation<3> &parallelMeshUnmovedVxc =
      d_triaManagerVxc.getParallelUnmovedMeshVxc();

  //  d_dofHandlerTriaVxc.reinit(parallelMeshMovedVxc);

  d_dofHandlerTriaVxc.reinit(parallelMeshUnmovedVxc);

  // TODO this hard coded to linear FE (which should be the usual case).
  // Read it from params file for generality
  const dealii::FE_Q<3> finite_elementVxc(1);

  d_dofHandlerTriaVxc.distribute_dofs(finite_elementVxc);

  dealii::IndexSet locallyRelevantDofsVxc;

  dealii::DoFTools::extract_locally_relevant_dofs(d_dofHandlerTriaVxc,
                                                  locallyRelevantDofsVxc);

  d_constraintMatrixVxc.clear();
  d_constraintMatrixVxc.reinit(locallyRelevantDofsVxc);
  dealii::DoFTools::make_hanging_node_constraints(d_dofHandlerTriaVxc,
                                                  d_constraintMatrixVxc);
  d_constraintMatrixVxc.close();

  typename MatrixFree<3>::AdditionalData additional_data;
  // comment this if using deal ii version 9
  // additional_data.mpi_communicator = d_mpiCommParent;
  additional_data.tasks_parallel_scheme =
      MatrixFree<3>::AdditionalData::partition_partition;

  //    additional_data.mapping_update_flags =
  //      update_values | update_JxW_values;

  additional_data.mapping_update_flags =
      update_values | update_gradients | update_JxW_values;

  d_matrixFreeDataVxc.reinit(dealii::MappingQ1<3, 3>(), d_dofHandlerTriaVxc,
                             d_constraintMatrixVxc, d_gaussQuadVxc,
                             additional_data);

  std::vector<const dealii::AffineConstraints<double> *> constraintsVector;
  constraintsVector.resize(1);
  constraintsVector[0] = &d_constraintMatrixVxc;
  d_basisOperationsChildHostPtr =
      std::make_shared<dftfe::basis::FEBasisOperations<
          dftfe::dataTypes::number, double, dftfe::utils::MemorySpace::HOST>>(
          d_blasWrapperHost);

  std::vector<unsigned int> quadratureIDVec;
  quadratureIDVec.resize(1);
  quadratureIDVec[0] = 0;

  std::vector<dftfe::basis::UpdateFlags> updateFlags;
  updateFlags.resize(1);
  updateFlags[0] = dftfe::basis::update_jxw | dftfe::basis::update_values |
                   dftfe::basis::update_transpose;
  d_basisOperationsChildHostPtr->init(d_matrixFreeDataVxc, constraintsVector, 0,
                                      quadratureIDVec, updateFlags);

  d_dofHandlerVxcIndex = 0;
  d_quadVxcIndex = 0;

  /*
      unsigned int maxRelativeRefinement = 0;
      d_triaManagerVxc.computeMapBetweenParentAndChildMesh(
        parallelMeshMoved,
        parallelMeshMovedVxc,
        d_mapParentCellsToChild,
        d_mapParentCellToChildCellsIter,
        d_mapChildCellsToParent,
        maxRelativeRefinement);

      std::cout << " max relative refinement = " << maxRelativeRefinement <<
     "\n";
  */
  MPI_Barrier(d_mpiComm_domain);
  double constraintsEnd = MPI_Wtime();
  /*
      d_inverseDftDoFManagerObjPtr =
     std::make_shared<TransferDataBetweenMeshesCompatiblePartitioning>(*d_dftMatrixFreeData,
                                       d_dftDensityDoFHandlerIndex,
                                       d_dftQuadIndex,
                                       d_matrixFreeDataVxc,
                                       d_dofHandlerVxcIndex,
                                       d_quadVxcIndex,
                                       d_mapParentCellsToChild,
                                       d_mapParentCellToChildCellsIter,
                                       d_mapChildCellsToParent,
                                       maxRelativeRefinement,
                                       d_dftParams.useDevice);
  */

  d_inverseDftDoFManagerObjPtr = std::make_shared<
      dftfe::TransferDataBetweenMeshesIncompatiblePartitioning<memorySpace>>(
      *d_dftMatrixFreeData, d_dftDensityDoFHandlerIndex, d_dftQuadIndex,
      d_matrixFreeDataVxc, d_dofHandlerVxcIndex, d_quadVxcIndex,
      d_dftParams.verbosity, d_mpiComm_domain,
      d_inverseDFTParams.useMemOptForTransfer);
  MPI_Barrier(d_mpiComm_domain);
  double createMapEnd = MPI_Wtime();

  pcout << " time for mesh generation = " << meshEnd - meshStart
        << " move mesh = " << meshMoveEnd - meshEnd
        << " constraints = " << constraintsEnd - meshMoveEnd
        << " map gen = " << createMapEnd - constraintsEnd << "\n";
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTEngine<FEOrder, FEOrderElectro, memorySpace>::
    setInitialDensityFromGaussian(
        const std::vector<dftfe::utils::MemoryStorage<
            double, dftfe::utils::MemorySpace::HOST>> &rhoValuesFeSpin) {
  // Quadrature for AX multiplication will FEOrderElectro+1
  const dealii::Quadrature<3> &quadratureRuleParent =
      d_dftMatrixFreeData->get_quadrature(d_dftQuadIndex);
  const unsigned int numQuadraturePointsPerCellParent =
      quadratureRuleParent.size();
  unsigned int totalLocallyOwnedCellsParent =
      d_dftMatrixFreeData->n_physical_cells();

  const unsigned int numTotalQuadraturePointsParent =
      totalLocallyOwnedCellsParent * numQuadraturePointsPerCellParent;

  const dealii::DoFHandler<3> *dofHandlerParent =
      &d_dftMatrixFreeData->get_dof_handler(d_dftDensityDoFHandlerIndex);
  dealii::FEValues<3> fe_valuesParent(
      dofHandlerParent->get_fe(), quadratureRuleParent,
      dealii::update_JxW_values | dealii::update_quadrature_points);

  const unsigned int numberDofsPerElement =
      dofHandlerParent->get_fe().dofs_per_cell;

  //
  // resize data members
  //

  std::vector<double> quadJxWValues(numTotalQuadraturePointsParent, 0.0);
  d_quadCoordinatesParent.resize(numTotalQuadraturePointsParent * 3, 0.0);
  typename dealii::DoFHandler<3>::active_cell_iterator
      cell = dofHandlerParent->begin_active(),
      endc = dofHandlerParent->end();
  unsigned int iElem = 0;
  unsigned int quadPtNo = 0;
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned()) {
      fe_valuesParent.reinit(cell);

      for (unsigned int q_point = 0; q_point < numQuadraturePointsPerCellParent;
           ++q_point) {
        quadJxWValues[(iElem * numQuadraturePointsPerCellParent) + q_point] =
            fe_valuesParent.JxW(q_point);
        dealii::Point<3, double> qPointVal =
            fe_valuesParent.quadrature_point(q_point);
        unsigned int qPointCoordIndex =
            ((iElem * numQuadraturePointsPerCellParent) + q_point) * 3;
        d_quadCoordinatesParent[qPointCoordIndex + 0] = qPointVal[0];
        d_quadCoordinatesParent[qPointCoordIndex + 1] = qPointVal[1];
        d_quadCoordinatesParent[qPointCoordIndex + 2] = qPointVal[2];
      }
      iElem++;
    }

  std::vector<std::string> densityMatPrimaryFileNames;
  densityMatPrimaryFileNames.push_back(
      d_inverseDFTParams.densityMatPrimaryFileNameSpinUp);
  if (d_numSpins == 2) {
    densityMatPrimaryFileNames.push_back(
        d_inverseDFTParams.densityMatPrimaryFileNameSpinDown);
  }

  gaussianFunctionManager gaussianFuncManPrimaryObj(
      densityMatPrimaryFileNames,             // densityMatFilenames
      d_inverseDFTParams.gaussianAtomicCoord, // atomicCoordsFilename
      'A',                                    // unit
      d_mpiComm_parent, d_mpiComm_domain);

  unsigned int gaussQuadIndex = 0;
  gaussianFuncManPrimaryObj.evaluateForQuad(
      &d_quadCoordinatesParent[0], &quadJxWValues[0], numTotalQuadraturePointsParent,
      true,  // evalBasis,
      false, // evalBasisDerivatives,
      false, // evalBasisDoubleDerivatives,
      true,  // evalSMat,
      true,  // normalizeBasis,
      gaussQuadIndex, d_inverseDFTParams.sMatrixName);

  std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      rhoGaussianPrimary;
  rhoGaussianPrimary.resize(
      d_numSpins,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
          totalLocallyOwnedCellsParent * numQuadraturePointsPerCellParent));

  for (unsigned int iSpin = 0; iSpin < d_numSpins; iSpin++) {
    gaussianFuncManPrimaryObj.getRhoValue(gaussQuadIndex, iSpin,
                                          rhoGaussianPrimary[iSpin].data());
  }

if (d_inverseDFTParams.useLb94InInitialguess)
{
	const dftfe::utils::MemoryStorage<dftfe::dataTypes::number, memorySpace>
      &eigenVectorsMemSpace = d_dftBaseClass->getEigenVectors();

	const std::vector<std::vector<double>> &eigenValuesHost =
      d_dftBaseClass->getEigenValues();
  const double fermiEnergy = d_dftBaseClass->getFermiEnergy();

  auto dftBasisOp = d_dftBaseClass->getBasisOperationsMemSpace();

  std::vector< dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>> rhoValues, gradRhoValues;
  rhoValues.resize(
      d_numSpins,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
          numQuadraturePointsPerCellParent * totalLocallyOwnedCellsParent));

  gradRhoValues.resize(
      d_numSpins,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
          3.0 * numQuadraturePointsPerCellParent *
          totalLocallyOwnedCellsParent));

	dftfe::computeRhoFromPSI<dftfe::dataTypes::number>(
      &eigenVectorsMemSpace, &eigenVectorsMemSpace, d_numEigenValues,
      d_numEigenValues, eigenValuesHost, fermiEnergy,
      fermiEnergy, // fermi energy up
      fermiEnergy, // fermi energy down
      dftBasisOp,
      // d_basisOperationsParentPtr[d_matrixFreePsiVectorComponent],
      d_blasWrapperMemSpace, d_dftBaseClass->getDensityDofHandlerIndex(),
      d_dftBaseClass->getDensityQuadratureId(),
      // d_matrixFreePsiVectorComponent,            // matrixFreeDofhandlerIndex
      // d_matrixFreeQuadratureComponentAdjointRhs, // quadratureIndex
      d_kpointWeights, rhoValues, gradRhoValues, true, d_mpiComm_parent,
      d_mpiComm_interpool , d_mpiComm_bandgroup , d_dftParams,
      false // spectrum splitting
	);

  if (d_numSpins == 1) {
  //  std::vector<double> qpointCoord(3, 0.0);
  //  std::vector<double> gradVal(3, 0.0);

    d_sigmaGradRhoTarget.resize(totalLocallyOwnedCellsParent *
                                numQuadraturePointsPerCellParent);
    std::fill(d_sigmaGradRhoTarget.begin(), d_sigmaGradRhoTarget.end(), 0.0);
    // TODO uncomment this after testing
    for (unsigned int iCell = 0; iCell < totalLocallyOwnedCellsParent;
         iCell++) {
      for (unsigned int q_point = 0; q_point < numQuadraturePointsPerCellParent;
           ++q_point) {
        unsigned int qPointId =
            (iCell * numQuadraturePointsPerCellParent) + q_point;
        unsigned int qPointCoordIndex = qPointId * 3;
/*
        qpointCoord[0] = d_quadCoordinatesParent[qPointCoordIndex + 0];
        qpointCoord[1] = d_quadCoordinatesParent[qPointCoordIndex + 1];
        qpointCoord[2] = d_quadCoordinatesParent[qPointCoordIndex + 2];

        gaussianFuncManPrimaryObj.getRhoGradient(&qpointCoord[0], 0, gradVal);
*/
        d_sigmaGradRhoTarget[qPointId] =
             gradRhoValues[0].data()[qPointCoordIndex + 0] *gradRhoValues[0].data()[qPointCoordIndex + 0] +
	     gradRhoValues[0].data()[qPointCoordIndex + 1] *gradRhoValues[0].data()[qPointCoordIndex + 1] +
	     gradRhoValues[0].data()[qPointCoordIndex + 2] * gradRhoValues[0].data()[qPointCoordIndex + 2];

        //                if ( d_sigmaGradRhoTarget[qPointId] > 1e8)
        //                  {
        //                    std::cout<<" Large value of d_sigmaGradRhoTarget
        //                    found at "<<qpointCoord[0]<<" "<<qpointCoord[1]<<"
        //                    "<<qpointCoord[2]<<"\n";
        //                  }
      }
    }
  }
  if (d_numSpins == 2) {
    std::vector<double> qpointCoord(3, 0.0);
    std::vector<double> gradValSpinUp(3, 0.0);
    std::vector<double> gradValSpinDown(3, 0.0);
    d_sigmaGradRhoTarget.resize(3 * totalLocallyOwnedCellsParent *
                                numQuadraturePointsPerCellParent);

    std::fill(d_sigmaGradRhoTarget.begin(), d_sigmaGradRhoTarget.end(), 0.0);
    for (unsigned int iCell = 0; iCell < totalLocallyOwnedCellsParent;
         iCell++) {
      for (unsigned int q_point = 0; q_point < numQuadraturePointsPerCellParent;
           ++q_point) {

        unsigned int qPointId =
            (iCell * numQuadraturePointsPerCellParent) + q_point;
        unsigned int qPointCoordIndex = qPointId * 3;

        qpointCoord[0] = d_quadCoordinatesParent[qPointCoordIndex + 0];
        qpointCoord[1] = d_quadCoordinatesParent[qPointCoordIndex + 1];
        qpointCoord[2] = d_quadCoordinatesParent[qPointCoordIndex + 2];

        gaussianFuncManPrimaryObj.getRhoGradient(&qpointCoord[0], 0,
                                                 gradValSpinUp);
        gaussianFuncManPrimaryObj.getRhoGradient(&qpointCoord[0], 1,
                                                 gradValSpinDown);

        d_sigmaGradRhoTarget[3 * qPointId + 0] =
            gradValSpinUp[0] * gradValSpinUp[0] +
            gradValSpinUp[1] * gradValSpinUp[1] +
            gradValSpinUp[2] * gradValSpinUp[2];

        d_sigmaGradRhoTarget[3 * qPointId + 1] =
            gradValSpinUp[0] * gradValSpinDown[0] +
            gradValSpinUp[1] * gradValSpinDown[1] +
            gradValSpinUp[2] * gradValSpinDown[2];

        d_sigmaGradRhoTarget[3 * qPointId + 2] =
            gradValSpinDown[0] * gradValSpinDown[0] +
            gradValSpinDown[1] * gradValSpinDown[1] +
            gradValSpinDown[2] * gradValSpinDown[2];
      }
    }
  }

  auto sigmaGradIt = std::max_element(d_sigmaGradRhoTarget.begin(),
                                      d_sigmaGradRhoTarget.end());
  double maxSigmaGradVal = *sigmaGradIt;

  MPI_Allreduce(MPI_IN_PLACE, &maxSigmaGradVal, 1, MPI_DOUBLE, MPI_MAX,
                d_mpiComm_domain);

  pcout << " Max vlaue of sigmaGradVal = " << maxSigmaGradVal << "\n";
}
  std::vector<std::string> densityMatDFTFileNames;
  densityMatDFTFileNames.push_back(
      d_inverseDFTParams.densityMatDFTFileNameSpinUp);
  if (d_numSpins == 2) {
    densityMatDFTFileNames.push_back(
        d_inverseDFTParams.densityMatDFTFileNameSpinDown);
  }

  gaussianFunctionManager gaussianFuncManDFTObj(
      densityMatDFTFileNames,                 // densityMatFilenames
      d_inverseDFTParams.gaussianAtomicCoord, // atomicCoordsFilename
      'A',                                    // unit
      d_mpiComm_parent, d_mpiComm_domain);

  gaussianFuncManDFTObj.evaluateForQuad(
      &d_quadCoordinatesParent[0], &quadJxWValues[0], numTotalQuadraturePointsParent,
      true,  // evalBasis,
      false, // evalBasisDerivatives,
      false, // evalBasisDoubleDerivatives,
      true,  // evalSMat,
      true,  // normalizeBasis,
      gaussQuadIndex, d_inverseDFTParams.sMatrixName);

  std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      rhoGaussianDFT;
  rhoGaussianDFT.resize(
      d_numSpins,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
          totalLocallyOwnedCellsParent * numQuadraturePointsPerCellParent));

  std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      rhoDiffMemStorage;
  rhoDiffMemStorage.resize(
      d_numSpins,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
          totalLocallyOwnedCellsParent * numQuadraturePointsPerCellParent));
  for (unsigned int iSpin = 0; iSpin < d_numSpins; iSpin++) {
    gaussianFuncManDFTObj.getRhoValue(gaussQuadIndex, iSpin,
                                      rhoGaussianDFT[iSpin].data());
  }

  d_rhoTarget.resize(
      d_numSpins,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
          totalLocallyOwnedCellsParent * numQuadraturePointsPerCellParent));
  for (unsigned int iSpin = 0; iSpin < d_numSpins; iSpin++) {
    cell = dofHandlerParent->begin_active();
    iElem = 0;

    if (d_inverseDFTParams.useDeltaRhoCorrection) {
      for (; cell != endc; ++cell)
        if (cell->is_locally_owned()) {
          for (unsigned int iQuad = 0; iQuad < numQuadraturePointsPerCellParent;
               iQuad++) {
            unsigned int index =
                iElem * numQuadraturePointsPerCellParent + iQuad;
            d_rhoTarget[iSpin][index] = rhoGaussianPrimary[iSpin][index] -
                                        rhoGaussianDFT[iSpin][index] +
                                        rhoValuesFeSpin[iSpin][index];
          }
          iElem++;
        }
    } else {
      for (; cell != endc; ++cell)
        if (cell->is_locally_owned()) {
          for (unsigned int iQuad = 0; iQuad < numQuadraturePointsPerCellParent;
               iQuad++) {
            unsigned int index =
                iElem * numQuadraturePointsPerCellParent + iQuad;
            d_rhoTarget[iSpin][index] = rhoGaussianPrimary[iSpin][index];
          }
          iElem++;
        }
    }

    for (; cell != endc; ++cell)
      if (cell->is_locally_owned()) {
        for (unsigned int iQuad = 0; iQuad < numQuadraturePointsPerCellParent;
             iQuad++) {
          unsigned int index = iElem * numQuadraturePointsPerCellParent + iQuad;
          d_rhoTarget[iSpin][index] = rhoGaussianPrimary[iSpin][index] -
                                      rhoGaussianDFT[iSpin][index] +
                                      rhoValuesFeSpin[iSpin][index];
        }
        iElem++;
      }
  }

  cell = dofHandlerParent->begin_active();
  iElem = 0;
  double rhoSumGaussian = 0.0;
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned()) {
      const dealii::CellId cellId = cell->id();
      for (unsigned int iQuad = 0; iQuad < numQuadraturePointsPerCellParent;
           iQuad++) {
        unsigned int index = iElem * numQuadraturePointsPerCellParent + iQuad;
        rhoDiffMemStorage[0][index] =
            rhoGaussianDFT[0][index] - rhoValuesFeSpin[0][index];

        if ((d_rhoTarget[0][iElem * numQuadraturePointsPerCellParent + iQuad] <
             0.0) &&
            (d_rhoTarget[0][iElem * numQuadraturePointsPerCellParent + iQuad] >
             -1e-7)) {
          d_rhoTarget[0][iElem * numQuadraturePointsPerCellParent + iQuad] =
              rhoGaussianPrimary[0][index];
        }
        if (d_rhoTarget[0][iElem * numQuadraturePointsPerCellParent + iQuad] <
            0.0) {
          unsigned int qPointCoordIndex =
              ((iElem * numQuadraturePointsPerCellParent) + iQuad) * 3;

          std::cout << " qPoint = (" << d_quadCoordinatesParent[qPointCoordIndex + 0]
                    << "," << d_quadCoordinatesParent[qPointCoordIndex + 1] << ","
                    << d_quadCoordinatesParent[qPointCoordIndex + 2]
                    << ") RHO IS NEGATIVE!!!!!!!!!!\n";
          std::cout << "primary = " << rhoGaussianPrimary[0][index]
                    << " secondary = " << rhoGaussianDFT[0][index]
                    << " Fe = " << rhoValuesFeSpin[0][index] << "\n";
        }
        rhoSumGaussian +=
            d_rhoTarget[0][iElem * numQuadraturePointsPerCellParent + iQuad] *
            quadJxWValues[(iElem * numQuadraturePointsPerCellParent) + iQuad];
      }
      iElem++;
    }
  MPI_Allreduce(MPI_IN_PLACE, &rhoSumGaussian, 1, MPI_DOUBLE, MPI_SUM,
                d_mpiComm_domain);
  pcout << " Sum of all rho target = " << rhoSumGaussian << "\n";

  dftfe::distributedCPUVec<double> rhoGPVec, rhoGSVec, rhoFLVec, rhoDiffVec;

  dftfe::vectorTools::createDealiiVector<double>(
      d_dftMatrixFreeData->get_vector_partitioner(d_dftDensityDoFHandlerIndex),
      1, rhoGPVec);
  rhoGPVec = 0.0;
  rhoGSVec.reinit(rhoGPVec);
  rhoGSVec = 0.0;
  rhoFLVec.reinit(rhoGPVec);
  rhoFLVec = 0.0;
  rhoDiffVec.reinit(rhoGPVec);
  rhoDiffVec = 0.0;

  unsigned int cellSizeParentTemp = 100;
  cellSizeParentTemp =
      std::min(cellSizeParentTemp, totalLocallyOwnedCellsParent);
  d_basisOperationsHost->reinit(1,
                                cellSizeParentTemp, // cellBlockSize
                                d_dftQuadIndex,     // quadId
                                true, false);

  d_dftBaseClass->l2ProjectionQuadToNodal(
      d_basisOperationsHost, *d_constraintDFTClass, d_dftDensityDoFHandlerIndex,
      d_dftQuadIndex, rhoGaussianPrimary[0], rhoGPVec);

  pcout << " Norm of rhoGPVec before distribute = " << rhoGPVec.l2_norm()
        << "\n";

  rhoGPVec.update_ghost_values();
  d_constraintDFTClass->distribute(rhoGPVec);
  rhoGPVec.update_ghost_values();

  pcout << " Norm of rhoGPVec after distribute = " << rhoGPVec.l2_norm()
        << "\n";

  d_dftBaseClass->l2ProjectionQuadToNodal(
      d_basisOperationsHost, *d_constraintDFTClass, d_dftDensityDoFHandlerIndex,
      d_dftQuadIndex, rhoGaussianDFT[0], rhoGSVec);
  pcout << " Norm of rhoGSVec before distribute = " << rhoGSVec.l2_norm()
        << "\n";

  rhoGSVec.update_ghost_values();
  d_constraintDFTClass->distribute(rhoGSVec);
  rhoGSVec.update_ghost_values();

  pcout << " Norm of rhoGSVec after distribute = " << rhoGSVec.l2_norm()
        << "\n";

  d_dftBaseClass->l2ProjectionQuadToNodal(
      d_basisOperationsHost, *d_constraintDFTClass, d_dftDensityDoFHandlerIndex,
      d_dftQuadIndex, rhoValuesFeSpin[0], rhoFLVec);

  pcout << " Norm of rhoFLVec before distribute = " << rhoFLVec.l2_norm()
        << "\n";

  rhoFLVec.update_ghost_values();
  d_constraintDFTClass->distribute(rhoFLVec);
  rhoFLVec.update_ghost_values();

  pcout << " Norm of rhoFLVec after distribute = " << rhoFLVec.l2_norm()
        << "\n";

  d_dftBaseClass->l2ProjectionQuadToNodal(
      d_basisOperationsHost, *d_constraintDFTClass, d_dftDensityDoFHandlerIndex,
      d_dftQuadIndex, rhoDiffMemStorage[0], rhoDiffVec);

  pcout << " Norm of rhoDiffVec before distribute = " << rhoDiffVec.l2_norm()
        << "\n";

  rhoDiffVec.update_ghost_values();
  d_constraintDFTClass->distribute(rhoDiffVec);
  rhoDiffVec.update_ghost_values();

  pcout << " Norm of rhoDiffVec after distribute = " << rhoDiffVec.l2_norm()
        << "\n";
  /*
     dealii::DataOut<3> data_out_rho;

     data_out_rho.attach_dof_handler(*dofHandlerParent);

     std::string outputVecName1 = "rho Gaussian primary";
     std::string outputVecName2 = "rho Gaussian secondary";
     std::string outputVecName3 = "rho fe lda";
     std::string outputVecName4 = "rho diff";
     data_out_rho.add_data_vector(rhoGPVec,outputVecName1);
     data_out_rho.add_data_vector(rhoGSVec,outputVecName2);
     data_out_rho.add_data_vector(rhoFLVec,outputVecName3);
     data_out_rho.add_data_vector(rhoDiffVec,outputVecName4);

     data_out_rho.build_patches();
     data_out_rho.write_vtu_with_pvtu_record("./", "inputRhoData",
     0,d_mpiComm_domain,2, 4);
  */
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTEngine<FEOrder, FEOrderElectro,
                      memorySpace>::setInitialPotL2Proj() {
  unsigned int totalLocallyOwnedCellsVxc =
      d_matrixFreeDataVxc.n_physical_cells();

  const unsigned int numQuadPointsPerCellInVxc = d_gaussQuadVxc.size();

  double spinFactor = (d_dftParams.spinPolarized == 1) ? 1.0 : 2.0;
  /*
    int isSpinPolarized;
    if (d_dftParams.spinPolarized == 1) {
      isSpinPolarized = XC_POLARIZED;
    } else {
      isSpinPolarized = XC_UNPOLARIZED;
    }

    dftfe::excDensityBaseClass *excFunctionalPtrLDA, *excFunctionalPtrGGA;

    dftfe::excManager excManagerObjLDA;
    excManagerObjLDA.init(d_dftParams.xc_id,
                          // isSpinPolarized,
                          (d_dftParams.spinPolarized == 1) ? true : false,
                          0.0,   // exx factor
                          false, // scale exchange
                          1.0,   // scale exchange factor
                          true,  // computeCorrelation
                          "");

    excFunctionalPtrLDA = excManagerObjLDA.getExcDensityObj();

    dftfe::excManager excManagerObjGGA;
    excManagerObjGGA.init(8, // TODO this is experimental // X - LB , C = PBE
                             // isSpinPolarized,
                          (d_dftParams.spinPolarized == 1) ? true : false,
                          0.0,   // exx factor
                          false, // scale exchange
                          1.0,   // scale exchange factor
                          true,  // computeCorrelation
                          " ");
    excFunctionalPtrGGA = excManagerObjGGA.getExcDensityObj();
  */
  std::vector<dftfe::distributedCPUVec<double>> vxcInitialGuess;

  unsigned int locallyOwnedDofs = d_dofHandlerDFTClass->n_locally_owned_dofs();

  vxcInitialGuess.resize(d_numSpins);
  std::vector<dftfe::utils::MemoryStorage<dftfe::dataTypes::number,
                                          dftfe::utils::MemorySpace::HOST>>
      initialPotValuesChildQuad;
  initialPotValuesChildQuad.resize(d_numSpins);
  d_vxcInitialChildNodes.resize(d_numSpins);

  d_targetPotValuesParentQuadData.resize(d_numSpins);

  unsigned int totalOwnedCellsPsi = d_dftMatrixFreeData->n_physical_cells();

  const dealii::Quadrature<3> &quadratureRulePsi =
      d_dftMatrixFreeData->get_quadrature(d_dftQuadIndex);

  unsigned int numQuadPointsPerPsiCell = quadratureRulePsi.size();
  std::vector<double> rhoSpinFlattened(d_numSpins * totalOwnedCellsPsi *
                                       numQuadPointsPerPsiCell);

  d_vxcLDAQuadData.resize(
      d_numSpins,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
          totalOwnedCellsPsi * numQuadPointsPerPsiCell));

  std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
  exactPotValuesParentQuadData(
      d_numSpins,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
          totalOwnedCellsPsi * numQuadPointsPerPsiCell));

  for (unsigned int iSpin = 0; iSpin < d_numSpins; iSpin++) {
    for (unsigned int iCell = 0; iCell < totalOwnedCellsPsi; iCell++) {
      for (unsigned int iQuad = 0; iQuad < numQuadPointsPerPsiCell; iQuad++) {
        rhoSpinFlattened[(iCell * numQuadPointsPerPsiCell + iQuad) *
                             d_numSpins +
                         iSpin] =
            spinFactor *
            d_rhoTarget[iSpin][iCell * numQuadPointsPerPsiCell + iQuad];
      }
    }
  }

  dftfe::distributedCPUVec<double> rhoInputTotal;
  dftfe::vectorTools::createDealiiVector<double>(
      d_dftMatrixFreeData->get_vector_partitioner(d_dftDensityDoFHandlerIndex),
      1, rhoInputTotal);
  rhoInputTotal = 0.0;

  unsigned int totalLocallyOwnedCellsPsi =
      d_dftMatrixFreeData->n_physical_cells();

  unsigned int numLocallyOwnedDofsPsi =
      d_dofHandlerDFTClass->n_locally_owned_dofs();
  unsigned int numDofsPerCellPsi = d_dofHandlerDFTClass->get_fe().dofs_per_cell;

  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      rhoValues;
  rhoValues.resize(totalLocallyOwnedCellsPsi * numQuadPointsPerPsiCell);

  typename DoFHandler<3>::active_cell_iterator cellPsiPtr =
      d_dofHandlerDFTClass->begin_active();
  typename DoFHandler<3>::active_cell_iterator endcellPsiPtr =
      d_dofHandlerDFTClass->end();

  unsigned int iElem = 0;
  unsigned int spinIndex1 = 0;
  unsigned int spinIndex2 = 0;
  if (d_numSpins == 2) {
    spinIndex2 = 1;
  }
  for (; cellPsiPtr != endcellPsiPtr; ++cellPsiPtr) {
    if (cellPsiPtr->is_locally_owned()) {
      for (unsigned int iQuad = 0; iQuad < numQuadPointsPerPsiCell; iQuad++) {
        rhoValues[iElem * numQuadPointsPerPsiCell + iQuad] =
            d_rhoTarget[spinIndex1][iElem * numQuadPointsPerPsiCell + iQuad] +
            d_rhoTarget[spinIndex2][iElem * numQuadPointsPerPsiCell + iQuad];
      }
      iElem++;
    }
  }

  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      hartreeQuadData;
  computeHartreePotOnParentQuad(hartreeQuadData);

  // allocate storage for exchange potential
  std::vector<double> exchangePotentialVal(
      d_numSpins * totalOwnedCellsPsi * numQuadPointsPerPsiCell, 0.0);

  std::vector<double> corrPotentialVal(
      d_numSpins * totalOwnedCellsPsi * numQuadPointsPerPsiCell, 0.0);

  if (d_inverseDFTParams.useLb94InInitialguess)
  {
	  std::vector<double> derExchEnergyWithSigmaValDummy(
      d_sigmaGradRhoTarget.size(), 0.0);

  xc_func_type funcXGGA;
  xc_func_init(&funcXGGA, XC_GGA_X_LB,
               (d_numSpins == 2) ? XC_POLARIZED : XC_UNPOLARIZED);
  xc_gga_vxc(&funcXGGA, totalOwnedCellsPsi * numQuadPointsPerPsiCell,
             &rhoSpinFlattened[0], &d_sigmaGradRhoTarget[0],
             &exchangePotentialVal[0], &derExchEnergyWithSigmaValDummy[0]);
  }
  else
  {
	  xc_func_type funcXLDA;
	  xc_func_init(&funcXLDA, XC_LDA_X, (d_numSpins == 2) ? XC_POLARIZED : XC_UNPOLARIZED);

	  xc_lda_vxc(&funcXLDA,totalOwnedCellsPsi * numQuadPointsPerPsiCell, &rhoSpinFlattened[0], &exchangePotentialVal[0]);

  }
  xc_func_type funcCLDA;

  xc_func_init(&funcCLDA, XC_LDA_C_PW,
               (d_numSpins == 2) ? XC_POLARIZED : XC_UNPOLARIZED);
  xc_lda_vxc(&funcCLDA, totalOwnedCellsPsi * numQuadPointsPerPsiCell,
             &rhoSpinFlattened[0], &corrPotentialVal[0]);

  //  excFunctionalPtrLDA->computeDensityBasedVxc(
  //      totalOwnedCellsPsi * numQuadPointsPerPsiCell, rhoData,
  //      outputDerExchangeEnergyDummy, outputDerCorrEnergy);

  dftfe::dftUtils::constraintMatrixInfo<memorySpace>
      constraintsMatrixDataInfoPsi;
  constraintsMatrixDataInfoPsi.initialize(
      d_dftMatrixFreeData->get_vector_partitioner(d_dftDensityDoFHandlerIndex),
      *d_constraintDFTClass);

  unsigned int numElectrons = d_dftBaseClass->getNumElectrons();

  for (unsigned int spinIndex = 0; spinIndex < d_numSpins; ++spinIndex) {
    dftfe::vectorTools::createDealiiVector<double>(
        d_dftMatrixFreeData->get_vector_partitioner(
            d_dftDensityDoFHandlerIndex),
        1, vxcInitialGuess[spinIndex]);
    vxcInitialGuess[spinIndex] = 0.0;

    dftfe::vectorTools::createDealiiVector<double>(
        d_matrixFreeDataVxc.get_vector_partitioner(d_dofHandlerVxcIndex), 1,
        d_vxcInitialChildNodes[spinIndex]);
    d_vxcInitialChildNodes[spinIndex] = 0.0;

    d_targetPotValuesParentQuadData[spinIndex].resize(totalOwnedCellsPsi);

    double sumInitialValuesVxcParent = 0.0;
    dealii::DoFHandler<3>::active_cell_iterator cellPsi = d_dofHandlerDFTClass
                                                              ->begin_active(),
                                                endcPsi =
                                                    d_dofHandlerDFTClass->end();
    unsigned int iElemPsi = 0;
    for (; cellPsi != endcPsi; ++cellPsi)
      if (cellPsi->is_locally_owned()) {
        d_targetPotValuesParentQuadData[spinIndex][iElemPsi].resize(
            numQuadPointsPerPsiCell, 0.0);
        std::vector<double> cellLevelQuadInput;
        cellLevelQuadInput.resize(numQuadPointsPerPsiCell);

        for (unsigned int iQuad = 0; iQuad < numQuadPointsPerPsiCell; iQuad++) {
          double tau = d_inverseDFTParams.inverseTauForVxBc;
          double preFactor =
              rhoSpinFlattened[(iElemPsi * numQuadPointsPerPsiCell + iQuad) *
                                   d_numSpins +
                               spinIndex] /
              (rhoSpinFlattened[(iElemPsi * numQuadPointsPerPsiCell + iQuad) *
                                    d_numSpins +
                                spinIndex] +
               tau);
          double exchangeValue = exchangePotentialVal
              [(iElemPsi * numQuadPointsPerPsiCell + iQuad) * d_numSpins +
               spinIndex];
          double exchangeCorrValue =
              exchangePotentialVal[(iElemPsi * numQuadPointsPerPsiCell +
                                    iQuad) *
                                       d_numSpins +
                                   spinIndex] +
              corrPotentialVal[(iElemPsi * numQuadPointsPerPsiCell + iQuad) *
                                   d_numSpins +
                               spinIndex];

          cellLevelQuadInput[iQuad] = ((1.0 - preFactor) * exchangeValue +
                                       (preFactor)*exchangeCorrValue);

          if (d_inverseDFTParams.fermiAmaldiBC) {
            double tauBC = d_inverseDFTParams.inverseTauForFABC;
            double preFactorBC =
                rhoSpinFlattened[(iElemPsi * numQuadPointsPerPsiCell + iQuad) *
                                     d_numSpins +
                                 spinIndex] /
                (rhoSpinFlattened[(iElemPsi * numQuadPointsPerPsiCell + iQuad) *
                                      d_numSpins +
                                  spinIndex] +
                 tauBC);

            cellLevelQuadInput[iQuad] =
                preFactorBC * cellLevelQuadInput[iQuad] +
                (1.0 - preFactorBC) * (-1.0 / numElectrons) * d_inverseDFTParams.factorFermiAmaldi * 
                    hartreeQuadData[iElemPsi * numQuadPointsPerPsiCell + iQuad];
          }
          d_targetPotValuesParentQuadData[spinIndex][iElemPsi][iQuad] =
              exchangePotentialVal[(iElemPsi * numQuadPointsPerPsiCell +
                                    iQuad) *
                                       d_numSpins +
                                   spinIndex] +
              corrPotentialVal[(iElemPsi * numQuadPointsPerPsiCell + iQuad) *
                                   d_numSpins +
                               spinIndex];
          exactPotValuesParentQuadData[spinIndex][iElemPsi *
                                                      numQuadPointsPerPsiCell +
                                                  iQuad] =
              exchangePotentialVal[(iElemPsi * numQuadPointsPerPsiCell +
                                    iQuad) *
                                       d_numSpins +
                                   spinIndex] +
              corrPotentialVal[(iElemPsi * numQuadPointsPerPsiCell + iQuad) *
                                   d_numSpins +
                               spinIndex];
          d_vxcLDAQuadData[spinIndex][(
              iElemPsi * numQuadPointsPerPsiCell + iQuad)] =
              cellLevelQuadInput[iQuad];
        }
        iElemPsi++;
      }

	   d_dftBaseClass->l2ProjectionQuadToNodal(
        d_basisOperationsHost, *d_constraintDFTClass,
        d_dftDensityDoFHandlerIndex, d_dftQuadIndex,
        d_vxcLDAQuadData[spinIndex], vxcInitialGuess[spinIndex]);

    pcout << " vxcInitialGuess norm before distribute= "
          << vxcInitialGuess[spinIndex].l2_norm() << "\n";

    // vxcInitialGuess[spinIndex].update_ghost_values();
    // constraintsMatrixDataInfoPsi.distribute(vxcInitialGuess[spinIndex], 1);
    // vxcInitialGuess[spinIndex].update_ghost_values();

    vxcInitialGuess[spinIndex].update_ghost_values();
    d_constraintDFTClass->distribute(vxcInitialGuess[spinIndex]);
    vxcInitialGuess[spinIndex].update_ghost_values();

    pcout << " vxcInitialGuess norm after distribute= "
          << vxcInitialGuess[spinIndex].l2_norm() << "\n";
    dftfe::distributedCPUVec<double> exactVxcTestParent;
    exactVxcTestParent.reinit(vxcInitialGuess[spinIndex]);

    d_dftBaseClass->l2ProjectionQuadToNodal(
        d_basisOperationsHost, *d_constraintDFTClass,
        d_dftDensityDoFHandlerIndex, d_dftQuadIndex,
        exactPotValuesParentQuadData[spinIndex], exactVxcTestParent);

    pcout << " norm of exactVxcTestParent distribute = "
          << exactVxcTestParent.l2_norm() << "\n";
    // exactVxcTestParent.update_ghost_values();
    // constraintsMatrixDataInfoPsi.distribute(exactVxcTestParent, 1);
    // exactVxcTestParent.update_ghost_values();
    // pcout<<" norm of exactVxcTestParent distribute =
    // "<<exactVxcTestParent.l2_norm()<<"\n";

    exactVxcTestParent.update_ghost_values();
    d_constraintDFTClass->distribute(exactVxcTestParent);
    exactVxcTestParent.update_ghost_values();

    pcout << " norm of exactVxcTestParent distribute = "
          << exactVxcTestParent.l2_norm() << "\n";

    dftfe::utils::MemoryStorage<dftfe::global_size_type,
                                dftfe::utils::MemorySpace::HOST>
        fullFlattenedArrayCellLocalProcIndexIdMapVxcInitialParent;

    unsigned int totalOwnedCellsParent =
        d_dftMatrixFreeData->n_physical_cells();

    unsigned int numCellRangeParent = 100;
    numCellRangeParent = std::min(numCellRangeParent, totalOwnedCellsParent);
    d_basisOperationsHost->reinit(1,
                                  numCellRangeParent, // cellBlockSize
                                  d_dftQuadIndex,     // quadId
                                  false, false);

    fullFlattenedArrayCellLocalProcIndexIdMapVxcInitialParent =
        d_basisOperationsHost->getFlattenedMapsHost();

    std::vector<dftfe::utils::MemoryStorage<dftfe::dataTypes::number,
                                            dftfe::utils::MemorySpace::HOST>>
        exactPotValuesChildQuad;
    exactPotValuesChildQuad.resize(d_numSpins);

    exactPotValuesChildQuad[spinIndex].resize(totalLocallyOwnedCellsVxc *
                                              numQuadPointsPerCellInVxc);
    std::fill(exactPotValuesChildQuad[spinIndex].begin(),
              exactPotValuesChildQuad[spinIndex].end(), 0.0);

    pcout << " exactVxcTestParent norm = " << exactVxcTestParent.l2_norm()
          << "\n";
    d_inverseDftDoFManagerObjPtr->interpolateMesh1DataToMesh2QuadPoints(
        d_blasWrapperHost, exactVxcTestParent,
        1, // blockSize
        fullFlattenedArrayCellLocalProcIndexIdMapVxcInitialParent,
        exactPotValuesChildQuad[spinIndex], 1,1,0,true);

    double sumExactPotValuesChild = 0.0;
    for (unsigned int iQuad = 0;
         iQuad < totalLocallyOwnedCellsVxc * numQuadPointsPerCellInVxc;
         iQuad++) {
      sumExactPotValuesChild +=
          exactPotValuesChildQuad[spinIndex].data()[iQuad];
    }

    MPI_Allreduce(MPI_IN_PLACE, &sumExactPotValuesChild, 1,
                  dftfe::dataTypes::mpi_type_id(&sumExactPotValuesChild),
                  MPI_SUM, d_mpiComm_domain);

    pcout << " sumExactPotValuesChild = " << sumExactPotValuesChild << "\n";

    initialPotValuesChildQuad[spinIndex].resize(totalLocallyOwnedCellsVxc *
                                                numQuadPointsPerCellInVxc);
    std::fill(initialPotValuesChildQuad[spinIndex].begin(),
              initialPotValuesChildQuad[spinIndex].end(), 0.0);

    d_inverseDftDoFManagerObjPtr->interpolateMesh1DataToMesh2QuadPoints(
        d_blasWrapperHost, vxcInitialGuess[spinIndex],
        1, // blockSize
        fullFlattenedArrayCellLocalProcIndexIdMapVxcInitialParent,
        initialPotValuesChildQuad[spinIndex], 1,1,0,true);

    dealii::DoFHandler<3>::active_cell_iterator cell = d_dofHandlerTriaVxc
                                                           .begin_active(),
                                                endc =
                                                    d_dofHandlerTriaVxc.end();
    unsigned int iElem = 0;


    for(unsigned int iQuad = 0 ; iQuad < totalLocallyOwnedCellsVxc *
                                                numQuadPointsPerCellInVxc; iQuad++)
    {
	    initialPotValuesChildQuad[spinIndex].data()[iQuad] = initialPotValuesChildQuad[spinIndex].data()[iQuad]
		    *(1.0 - d_inverseDFTParams.factorForLDAVxc); 
    }

    d_dftBaseClass->l2ProjectionQuadToNodal(
        d_basisOperationsChildHostPtr, d_constraintMatrixVxc,
        d_dofHandlerVxcIndex, d_quadVxcIndex,
        initialPotValuesChildQuad[spinIndex],
        d_vxcInitialChildNodes[spinIndex]);

    d_constraintMatrixVxc.set_zero(d_vxcInitialChildNodes[spinIndex]);
    //    d_vxcInitialChildNodes[spinIndex].zero_out_ghosts();

    dftfe::distributedCPUVec<double> exactVxcTestChild;
    exactVxcTestChild.reinit(d_vxcInitialChildNodes[spinIndex]);
    d_dftBaseClass->l2ProjectionQuadToNodal(
        d_basisOperationsChildHostPtr, d_constraintMatrixVxc,
        d_dofHandlerVxcIndex, d_quadVxcIndex,
        exactPotValuesChildQuad[spinIndex], exactVxcTestChild);

    pcout << "exactVxcTestChild norm before distribute = "
          << exactVxcTestChild.l2_norm() << "\n";

    exactVxcTestChild.update_ghost_values();
    d_constraintMatrixVxc.distribute(exactVxcTestChild);
    exactVxcTestChild.update_ghost_values();

    pcout << "exactVxcTestChild norm after distribute = "
          << exactVxcTestChild.l2_norm() << "\n";
    /*
          pcout<<"writing exact vxc output\n";
          dealii::DataOut<3> data_out_vxc;

          data_out_vxc.attach_dof_handler(d_dofHandlerTriaVxc);

          std::string outputVecName1 = "exact vxc";
          data_out_vxc.add_data_vector(exactVxcTestChild, outputVecName1);

          data_out_vxc.build_patches();
          data_out_vxc.write_vtu_with_pvtu_record("./", "exactVxc",
       0,d_mpiComm_domain ,2, 4);
  */

    pcout << "writing initial vxc guess\n";
    dealii::DataOut<3> data_out_vxc;

    data_out_vxc.attach_dof_handler(*d_dofHandlerDFTClass);

    std::string outputVecName1 = "initial vxc";
    data_out_vxc.add_data_vector(vxcInitialGuess[0], outputVecName1);

    data_out_vxc.build_patches();
    data_out_vxc.write_vtu_with_pvtu_record("./", "initiaVxcGuess", 0,
                                            d_mpiComm_domain, 2, 4);
  }

  //    delete (excFunctionalPtrLDA);
  //    delete (excFunctionalPtrGGA);

  d_dftBaseClass->l2ProjectionQuadToNodal(
      d_basisOperationsHost, *d_constraintDFTClass, d_dftDensityDoFHandlerIndex,
      d_dftQuadIndex, rhoValues, rhoInputTotal);

  pcout << "norm of rhoInputTotal before constraints = "
        << rhoInputTotal.l2_norm() << "\n";
  rhoInputTotal.update_ghost_values();
  // constraintsMatrixDataInfoPsi.distribute(rhoInputTotal, 1);

  d_constraintDFTClass->distribute(rhoInputTotal);
  rhoInputTotal.update_ghost_values();
  pcout << "norm of rhoInputTotal after constraints = "
        << rhoInputTotal.l2_norm() << "\n";

  setAdjointBoundaryCondition(rhoInputTotal);
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTEngine<FEOrder, FEOrderElectro, memorySpace>::
    setAdjointBoundaryCondition(dftfe::distributedCPUVec<double> &rhoTarget) {
  /*
  pcout<<"writing rho target";
  dealii::DataOut<3> data_out;

  data_out.attach_dof_handler(*d_dofHandlerDFTClass);

  std::string outputVecName = "rhoTarget";
  data_out.add_data_vector(rhoTarget, outputVecName);

  data_out.build_patches();
  data_out.write_vtu_with_pvtu_record("./", "rhoTarget", 0,d_mpiComm_domain
  ,2, 4);
*/
  dealii::IndexSet localSet = d_dofHandlerDFTClass->locally_owned_dofs();

  dealii::IndexSet locallyRelevantDofsAdjoint;

  dealii::DoFTools::extract_locally_relevant_dofs(*d_dofHandlerDFTClass,
                                                  locallyRelevantDofsAdjoint);

  dftfe::distributedCPUVec<double> rhoTargetFullVector;
  rhoTargetFullVector.reinit(localSet, locallyRelevantDofsAdjoint,
                             d_mpiComm_domain);

  unsigned int locallyOwnedDofs = d_dofHandlerDFTClass->n_locally_owned_dofs();
  for (unsigned int iNode = 0; iNode < locallyOwnedDofs; iNode++) {
    rhoTargetFullVector.local_element(iNode) = rhoTarget.local_element(iNode);
  }
  rhoTargetFullVector.update_ghost_values();

  d_constraintMatrixAdjoint.clear();
  d_constraintMatrixAdjoint.reinit(locallyRelevantDofsAdjoint);
  dealii::DoFTools::make_hanging_node_constraints(*d_dofHandlerDFTClass,
                                                  d_constraintMatrixAdjoint);

  const unsigned int dofs_per_cell =
      d_dofHandlerDFTClass->get_fe().dofs_per_cell;
  const unsigned int faces_per_cell = dealii::GeometryInfo<3>::faces_per_cell;
  const unsigned int dofs_per_face =
      d_dofHandlerDFTClass->get_fe().dofs_per_face;

  std::vector<dealii::types::global_dof_index> cellGlobalDofIndices(
      dofs_per_cell);
  std::vector<dealii::types::global_dof_index> iFaceGlobalDofIndices(
      dofs_per_face);

  std::vector<bool> dofs_touched(d_dofHandlerDFTClass->n_dofs(), false);

  dealii::DoFHandler<3>::active_cell_iterator cell = d_dofHandlerDFTClass
                                                         ->begin_active(),
                                              endc =
                                                  d_dofHandlerDFTClass->end();
  unsigned int adjointConstraiedNodes = 0;
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned() || cell->is_ghost()) {
      cell->get_dof_indices(cellGlobalDofIndices);
      for (unsigned int iDof = 0; iDof < dofs_per_cell; iDof++) {
        const dealii::types::global_dof_index nodeId =
            cellGlobalDofIndices[iDof];
        if (dofs_touched[nodeId])
          continue;
        dofs_touched[nodeId] = true;
        if (rhoTargetFullVector[nodeId] < d_rhoTargetTolForConstraints) {
          if (!d_constraintMatrixAdjoint.is_constrained(nodeId)) {
            if (rhoTargetFullVector.in_local_range(nodeId)) {
              adjointConstraiedNodes++;
            }
            d_constraintMatrixAdjoint.add_line(nodeId);
            d_constraintMatrixAdjoint.set_inhomogeneity(nodeId, 0.0);
          } // non-hanging node check
        }
      }
      //          for (unsigned int iFace = 0; iFace < faces_per_cell;
      //          ++iFace)
      //            {
      //              const unsigned int boundaryId =
      //              cell->face(iFace)->boundary_id(); if (boundaryId == 0)
      //                {
      //                  cell->face(iFace)->get_dof_indices(iFaceGlobalDofIndices);
      //                  for (unsigned int iFaceDof = 0; iFaceDof <
      //                  dofs_per_face;
      //                       ++iFaceDof)
      //                    {
      //                      const dealii::types::global_dof_index nodeId =
      //                        iFaceGlobalDofIndices[iFaceDof];
      //                      if (dofs_touched[nodeId])
      //                        continue;
      //                      dofs_touched[nodeId] = true;
      //                      if
      //                      (!d_constraintMatrixAdjoint.is_constrained(nodeId))
      //                        {
      //                          d_constraintMatrixAdjoint.add_line(nodeId);
      //                          d_constraintMatrixAdjoint.set_inhomogeneity(nodeId,
      //                          0.0);
      //                        } // non-hanging node check
      //                    }     // Face dof loop
      //                }         // non-periodic boundary id
      //            }
    }

  d_constraintMatrixAdjoint.close();

  MPI_Allreduce(MPI_IN_PLACE, &adjointConstraiedNodes, 1,
                dftfe::dataTypes::mpi_type_id(&adjointConstraiedNodes), MPI_SUM,
                d_mpiComm_domain);

  pcout << " no of constrained adjoint from manual addition  = "
        << adjointConstraiedNodes << "\n";

  // std::cout << " num adjoint constraints iProc = " << this_mpi_process
  //          << "size = " << d_constraintMatrixAdjoint.n_constraints() << "\n";

  IndexSet locally_active_dofs;

  DoFTools::extract_locally_active_dofs(*d_dofHandlerDFTClass,
                                        locally_active_dofs);

  bool consistentConstraints =
      d_constraintMatrixAdjoint.is_consistent_in_parallel(
          Utilities::MPI::all_gather(
              d_mpiComm_domain, d_dofHandlerDFTClass->locally_owned_dofs()),
          locally_active_dofs, d_mpiComm_domain, true);

  pcout << " Are the constraints consistent across partitoners = "
        << consistentConstraints << "\n";

  typename MatrixFree<3>::AdditionalData additional_data;
  // comment this if using deal ii version 9
  // additional_data.mpi_communicator = d_mpiCommParent;
  additional_data.tasks_parallel_scheme =
      MatrixFree<3>::AdditionalData::partition_partition;

  additional_data.mapping_update_flags =
      update_values | update_gradients | update_JxW_values;

  std::vector<const dealii::DoFHandler<3> *> matrixFreeDofHandlerVectorInput;
  matrixFreeDofHandlerVectorInput.push_back(d_dofHandlerDFTClass);
  matrixFreeDofHandlerVectorInput.push_back(d_dofHandlerDFTClass);

  d_constraintsVectorAdjoint.clear();
  d_constraintsVectorAdjoint.push_back(&d_constraintMatrixAdjoint);
  d_constraintsVectorAdjoint.push_back(d_constraintDFTClass);

  std::vector<Quadrature<1>> quadratureVector(0);

  unsigned int quadRhsVal = std::cbrt(d_gaussQuadAdjoint->size());
  pcout << " rhs quad adjoint val  = " << quadRhsVal << "\n";

  quadratureVector.push_back(QGauss<1>(quadRhsVal));

  d_matrixFreeDataAdjoint.reinit(
      dealii::MappingQ1<3, 3>(), matrixFreeDofHandlerVectorInput,
      d_constraintsVectorAdjoint, quadratureVector, additional_data);

  d_adjointMFAdjointConstraints = 0;
  d_adjointMFPsiConstraints = 1;
  d_quadAdjointIndex = 0;

  d_basisOperationsAdjointHostPtr.resize(
      2,
      std::make_shared<dftfe::basis::FEBasisOperations<
          dftfe::dataTypes::number, double, dftfe::utils::MemorySpace::HOST>>(
          d_blasWrapperHost));

  std::vector<dftfe::basis::UpdateFlags> updateFlags;
  updateFlags.resize(1);
  updateFlags[0] = dftfe::basis::update_jxw | dftfe::basis::update_values |
                   dftfe::basis::update_gradients |
                   dftfe::basis::update_transpose;

  std::vector<unsigned int> quadratureVectorId;
  quadratureVectorId.resize(1);
  quadratureVectorId[0] = d_quadAdjointIndex;
  d_basisOperationsAdjointHostPtr[d_adjointMFAdjointConstraints]->init(
      d_matrixFreeDataAdjoint, d_constraintsVectorAdjoint,
      d_adjointMFAdjointConstraints, quadratureVectorId, updateFlags);

  d_basisOperationsAdjointHostPtr[d_adjointMFPsiConstraints]->init(
      d_matrixFreeDataAdjoint, d_constraintsVectorAdjoint,
      d_adjointMFPsiConstraints, quadratureVectorId, updateFlags);

  d_basisOperationsAdjointMemSpacePtr.resize(
      2,
      std::make_shared<dftfe::basis::FEBasisOperations<dftfe::dataTypes::number,
                                                       double, memorySpace>>(
          d_blasWrapperMemSpace));

  d_basisOperationsAdjointMemSpacePtr[d_adjointMFAdjointConstraints]->init(
      *(d_basisOperationsAdjointHostPtr[d_adjointMFAdjointConstraints].get()));
  d_basisOperationsAdjointMemSpacePtr[d_adjointMFPsiConstraints]->init(
      *(d_basisOperationsAdjointHostPtr[d_adjointMFPsiConstraints].get()));

  std::vector<unsigned int> constraintedDofsAdjointMF;
  constraintedDofsAdjointMF = d_matrixFreeDataAdjoint.get_constrained_dofs(
      d_adjointMFAdjointConstraints);
  unsigned int sizeLocalAdjointMF = constraintedDofsAdjointMF.size();

  unsigned int numConstraintsLocalMF = 0;
  for (unsigned int iNode = 0; iNode < sizeLocalAdjointMF; iNode++) {
    if (constraintedDofsAdjointMF[iNode] < locallyOwnedDofs) {
      numConstraintsLocalMF++;
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &numConstraintsLocalMF, 1,
                dftfe::dataTypes::mpi_type_id(&numConstraintsLocalMF), MPI_SUM,
                d_mpiComm_domain);

  MPI_Allreduce(MPI_IN_PLACE, &sizeLocalAdjointMF, 1,
                dftfe::dataTypes::mpi_type_id(&sizeLocalAdjointMF), MPI_SUM,
                d_mpiComm_domain);

  pcout << " no of constrained adjoint from MF  = " << sizeLocalAdjointMF
        << "\n";
  pcout << " no of local constrained adjoint from MF  = "
        << numConstraintsLocalMF << "\n";
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTEngine<FEOrder, FEOrderElectro, memorySpace>::setTargetDensity(
    const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoOutValues,
    const std::vector<
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
        &rhoOutValuesSpinPolarized) {
  unsigned int totalOwnedCellsElectro =
      d_dftMatrixFreeDataElectro->n_physical_cells();

  const dealii::Quadrature<3> &quadratureRule =
      d_dftMatrixFreeDataElectro->get_quadrature(d_dftElectroRhsQuadIndex);

  const unsigned int numQuadPointsElectroPerCell = quadratureRule.size();

  dealii::DoFHandler<3>::active_cell_iterator
      cellElectro = d_dofHandlerElectroDFTClass->begin_active(),
      endElectro = d_dofHandlerElectroDFTClass->end();

  d_rhoTarget.resize(d_numSpins);
  if (d_numSpins == 1) {
    unsigned int spinIndex = 0;
    d_rhoTarget[spinIndex].resize(totalOwnedCellsElectro *
                                  numQuadPointsElectroPerCell);
    unsigned int iElemElectro = 0;
    for (; cellElectro != endElectro; ++cellElectro)
      if (cellElectro->is_locally_owned()) {
        for (unsigned int iQuad = 0; iQuad < numQuadPointsElectroPerCell;
             iQuad++) {
          d_rhoTarget[spinIndex][iElemElectro * numQuadPointsElectroPerCell +
                                 iQuad] =
              0.5 *
              rhoOutValues[spinIndex]
                          [iElemElectro * numQuadPointsElectroPerCell + iQuad];
        }
        iElemElectro++;
      }
  } else {
    unsigned int spinIndex1 = 0;
    d_rhoTarget[spinIndex1].resize(totalOwnedCellsElectro *
                                   numQuadPointsElectroPerCell);

    unsigned int spinIndex2 = 0;
    d_rhoTarget[spinIndex2].resize(totalOwnedCellsElectro *
                                   numQuadPointsElectroPerCell);

    unsigned int iElemElectro = 0;
    for (; cellElectro != endElectro; ++cellElectro)
      if (cellElectro->is_locally_owned()) {
        for (unsigned int iQuad = 0; iQuad < numQuadPointsElectroPerCell;
             iQuad++) {
          d_rhoTarget[spinIndex1][iElemElectro * numQuadPointsElectroPerCell +
                                  iQuad] = rhoOutValuesSpinPolarized
              [spinIndex1][iElemElectro * numQuadPointsElectroPerCell + iQuad];

          d_rhoTarget[spinIndex2][iElemElectro * numQuadPointsElectroPerCell +
                                  iQuad] = rhoOutValuesSpinPolarized
              [spinIndex2][iElemElectro * numQuadPointsElectroPerCell + iQuad];
        }
        iElemElectro++;
      }
  }
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTEngine<FEOrder, FEOrderElectro, memorySpace>::
    computeHartreePotOnParentQuad(
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
            &hartreeQuadData) {
  unsigned int numElectrons = d_dftBaseClass->getNumElectrons();
  // set up the constraints and the matrixFreeObj
  pcout << " numElectrons = " << numElectrons << "\n";

  // TODO does not assume periodic BCs.
  std::vector<std::vector<double>> atomLocations =
      d_dftBaseClass->getAtomLocationsCart();

  double netChargeOnAtom = 0.0;
  if (d_dftParams.multipoleBoundaryConditions) {
    netChargeOnAtom =
        (double)d_inverseDFTParams.netCharge / atomLocations.size();
  }

  dealii::IndexSet locallyRelevantDofsElectro;

  dealii::DoFTools::extract_locally_relevant_dofs(*d_dofHandlerElectroDFTClass,
                                                  locallyRelevantDofsElectro);

  dealii::AffineConstraints<double> d_constraintMatrixElectroHartree;
  // TODO periodic boundary conditions are not included
  d_constraintMatrixElectroHartree.clear();
  d_constraintMatrixElectroHartree.reinit(locallyRelevantDofsElectro);
  dealii::DoFTools::make_hanging_node_constraints(
      *d_dofHandlerElectroDFTClass, d_constraintMatrixElectroHartree);

  const unsigned int dofs_per_cell =
      d_dofHandlerElectroDFTClass->get_fe().dofs_per_cell;
  const unsigned int faces_per_cell = dealii::GeometryInfo<3>::faces_per_cell;
  const unsigned int dofs_per_face =
      d_dofHandlerElectroDFTClass->get_fe().dofs_per_face;

  std::vector<dealii::types::global_dof_index> cellGlobalDofIndices(
      dofs_per_cell);
  std::vector<dealii::types::global_dof_index> iFaceGlobalDofIndices(
      dofs_per_face);

  std::vector<bool> dofs_touched(d_dofHandlerElectroDFTClass->n_dofs(), false);

  dealii::MappingQGeneric<3, 3> mapping(1);
  std::map<dealii::types::global_dof_index, dealii::Point<3, double>>
      dof_coords_electro;
  dealii::DoFTools::map_dofs_to_support_points<3, 3>(
      mapping, *d_dofHandlerElectroDFTClass, dof_coords_electro);

  dealii::DoFHandler<3>::active_cell_iterator
      cellElectro = d_dofHandlerElectroDFTClass->begin_active(),
      endElectro = d_dofHandlerElectroDFTClass->end();
  for (; cellElectro != endElectro; ++cellElectro)
    if (cellElectro->is_locally_owned() || cellElectro->is_ghost()) {
      cellElectro->get_dof_indices(cellGlobalDofIndices);
      for (unsigned int iFace = 0; iFace < faces_per_cell; ++iFace) {
        const unsigned int boundaryId = cellElectro->face(iFace)->boundary_id();
        if (boundaryId == 0) {
          cellElectro->face(iFace)->get_dof_indices(iFaceGlobalDofIndices);
          for (unsigned int iFaceDof = 0; iFaceDof < dofs_per_face;
               ++iFaceDof) {
            const dealii::types::global_dof_index nodeId =
                iFaceGlobalDofIndices[iFaceDof];
            if (dofs_touched[nodeId])
              continue;
            dofs_touched[nodeId] = true;
            if (!d_constraintMatrixElectroHartree.is_constrained(nodeId)) {
              //                          pcout << " setting constraints for = "
              //                          << nodeId
              //                                << "\n";
              //                          double rad = 0.0;
              //                          rad =
              //                          dof_coords_electro[nodeId][0]*dof_coords_electro[nodeId][0];
              //                          rad +=
              //                          dof_coords_electro[nodeId][1]*dof_coords_electro[nodeId][1];
              //                          rad +=
              //                          dof_coords_electro[nodeId][2]*dof_coords_electro[nodeId][2];
              //                          rad = std::sqrt(rad);
              //                          if( rad < 1e-6)
              //                            {
              //                              pcout<<"Errorrrrrr in
              //                              rad \n";
              //                            }
              double nodalConstraintVal = 0.0;
              for (unsigned int iAtom = 0; iAtom < atomLocations.size();
                   iAtom++) {
                double rad = 0.0;
                rad +=
                    (atomLocations[iAtom][2] - dof_coords_electro[nodeId][0]) *
                    (atomLocations[iAtom][2] - dof_coords_electro[nodeId][0]);
                rad +=
                    (atomLocations[iAtom][3] - dof_coords_electro[nodeId][1]) *
                    (atomLocations[iAtom][3] - dof_coords_electro[nodeId][1]);
                rad +=
                    (atomLocations[iAtom][4] - dof_coords_electro[nodeId][2]) *
                    (atomLocations[iAtom][4] - dof_coords_electro[nodeId][2]);
                rad = std::sqrt(rad);
                if (d_dftParams.isPseudopotential)
                  nodalConstraintVal +=
                      (atomLocations[iAtom][1] + netChargeOnAtom) / rad;
                else
                  nodalConstraintVal +=
                      (atomLocations[iAtom][0] + netChargeOnAtom) / rad;
              }
              d_constraintMatrixElectroHartree.add_line(nodeId);
              d_constraintMatrixElectroHartree.set_inhomogeneity(
                  nodeId, nodalConstraintVal);
              //                          d_constraintMatrixElectroHartree.set_inhomogeneity(nodeId,
              //                          0.0);
            } // non-hanging node check
          }   // Face dof loop
        }     // non-periodic boundary id
      }       // Face loop
    }         // cell locally owned
  d_constraintMatrixElectroHartree.close();

  const dealii::Quadrature<3> &quadratureRuleElectroRhs =
      d_dftMatrixFreeDataElectro->get_quadrature(d_dftElectroRhsQuadIndex);

  const dealii::Quadrature<3> &quadratureRuleElectroAx =
      d_dftMatrixFreeDataElectro->get_quadrature(d_dftElectroAxQuadIndex);

  std::vector<Quadrature<1>> quadratureVector(0);

  unsigned int quadRhsVal = std::cbrt(quadratureRuleElectroRhs.size());
  pcout << " first quad val  = " << quadRhsVal << "\n";

  unsigned int quadAxVal = std::cbrt(quadratureRuleElectroAx.size());
  pcout << " second quad val  = " << quadAxVal << "\n";

  quadratureVector.push_back(QGauss<1>(quadRhsVal));
  quadratureVector.push_back(QGauss<1>(quadAxVal));

  typename MatrixFree<3>::AdditionalData additional_data;
  // comment this if using deal ii version 9
  // additional_data.mpi_communicator = d_mpiCommParent;
  additional_data.tasks_parallel_scheme =
      MatrixFree<3>::AdditionalData::partition_partition;
  additional_data.mapping_update_flags =
      update_values | update_gradients | update_JxW_values;

  std::vector<const dealii::DoFHandler<3> *> matrixFreeDofHandlerVectorInput;
  matrixFreeDofHandlerVectorInput.push_back(d_dofHandlerElectroDFTClass);

  std::vector<const dealii::AffineConstraints<double> *>
      constraintsVectorElectro;

  constraintsVectorElectro.push_back(&d_constraintMatrixElectroHartree);

  // TODO check if passing the quadrature rules this way is correct
  dealii::MatrixFree<3, double> matrixFreeElectro;
  matrixFreeElectro.reinit(
      dealii::MappingQ1<3, 3>(), matrixFreeDofHandlerVectorInput,
      constraintsVectorElectro, quadratureVector, additional_data);
  unsigned int dofHandlerElectroIndex = 0;
  unsigned int quadratureElectroRhsId = 0;
  unsigned int quadratureElectroAxId = 1;

  std::map<dealii::types::global_dof_index, double> dummyAtomMap;
  std::map<dealii::CellId, std::vector<double>> dummySmearedChargeValues;

  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      totalRhoValues;

  unsigned int nLocalCellsElectro = matrixFreeElectro.n_physical_cells();

  dftfe::distributedCPUVec<double> vHartreeElectroNodal;

  dftfe::vectorTools::createDealiiVector<double>(
      matrixFreeElectro.get_vector_partitioner(dofHandlerElectroIndex), 1,
      vHartreeElectroNodal);

  vHartreeElectroNodal = 0.0;

  std::vector<unsigned int> constraintedDofsInverse;
  constraintedDofsInverse =
      matrixFreeElectro.get_constrained_dofs(dofHandlerElectroIndex);
  unsigned int sizeLocalInversePot = constraintedDofsInverse.size();

  MPI_Allreduce(MPI_IN_PLACE, &sizeLocalInversePot, 1,
                dftfe::dataTypes::mpi_type_id(&sizeLocalInversePot), MPI_SUM,
                d_mpiComm_domain);

  pcout << " no of constrained inverse  = " << sizeLocalInversePot << "\n";

  dealii::DoFHandler<3>::active_cell_iterator cell = d_dofHandlerDFTClass
                                                         ->begin_active(),
                                              endc =
                                                  d_dofHandlerDFTClass->end();
  unsigned int iElem = 0;

  const dealii::Quadrature<3> &quadratureRule =
      d_dftMatrixFreeData->get_quadrature(d_dftQuadIndex);
  const unsigned int numQuadPointsPerCell = quadratureRule.size();

  pcout << " numQuadPointsPerCell = " << numQuadPointsPerCell << "\n";
  totalRhoValues.resize(numQuadPointsPerCell * nLocalCellsElectro);
  unsigned int spinIndex1 = 0;
  unsigned int spinIndex2 = (d_numSpins == 2) ? 1 : 0;

  double sumTotalRho = 0.0;
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned()) {
      for (unsigned int iQuad = 0; iQuad < numQuadPointsPerCell; iQuad++) {
        totalRhoValues[iElem * numQuadPointsPerCell + iQuad] =
            d_rhoTarget[spinIndex1][iElem * numQuadPointsPerCell + iQuad] +
            d_rhoTarget[spinIndex2][iElem * numQuadPointsPerCell + iQuad];

        sumTotalRho += totalRhoValues[iElem * numQuadPointsPerCell + iQuad];
      }
      //          totalRhoValues[cell->id()] = cellLevelQuadInput;
      iElem++;
    }

  MPI_Allreduce(MPI_IN_PLACE, &sumTotalRho, 1,
                dftfe::dataTypes::mpi_type_id(&sumTotalRho), MPI_SUM,
                d_mpiComm_domain);

  pcout << " sum total rho = " << sumTotalRho << "\n";

  pcout << " solving possion in the compute quad hartree\n";

  std::shared_ptr<dftfe::basis::FEBasisOperations<
      dftfe::dataTypes::number, double, dftfe::utils::MemorySpace::HOST>>
      basisOpeElectroHost = std::make_shared<dftfe::basis::FEBasisOperations<
          dftfe::dataTypes::number, double, dftfe::utils::MemorySpace::HOST>>(
          d_blasWrapperHost);

  std::vector<dftfe::basis::UpdateFlags> updateFlags;
  updateFlags.resize(2);
  updateFlags[0] = dftfe::basis::update_jxw | dftfe::basis::update_values |
                   dftfe::basis::update_transpose;

  updateFlags[1] = dftfe::basis::update_jxw | dftfe::basis::update_values |
                   dftfe::basis::update_transpose;

  std::vector<unsigned int> quadratureVectorId;
  quadratureVectorId.resize(2);
  quadratureVectorId[0] = quadratureElectroRhsId;
  quadratureVectorId[1] = quadratureElectroAxId;
  basisOpeElectroHost->init(matrixFreeElectro, constraintsVectorElectro,
                            dofHandlerElectroIndex, quadratureVectorId,
                            updateFlags);

  unsigned int numCellsTempSize = 100;
  numCellsTempSize = std::min(numCellsTempSize, nLocalCellsElectro);
  basisOpeElectroHost->reinit(1, numCellsTempSize, quadratureElectroAxId);

  dftfe::poissonSolverProblem<FEOrder, FEOrderElectro> poissonSolverObj(
      d_mpiComm_domain);
  poissonSolverObj.reinit(basisOpeElectroHost, vHartreeElectroNodal,
                          d_constraintMatrixElectroHartree,
                          dofHandlerElectroIndex, quadratureElectroRhsId,
                          quadratureElectroAxId, dummyAtomMap,
                          dummySmearedChargeValues, 0, totalRhoValues,
                          true,  // isComputeDiagonalA
                          false, // isComputeMeanValueConstraints
                          false, // smearedNuclearCharges
                          true,  // isRhoValues
                          false, // isGradSmearedChargeRhs
                          0,     // smearedChargeGradientComponentId
                          false, // storeSmearedChargeRhs
                          false, // reuseSmearedChargeRhs
                          true); // reinitializeFastConstraints

  dftfe::dealiiLinearSolver dealiiLinearSolverObj(
      d_mpiComm_parent, d_mpiComm_domain, dftfe::dealiiLinearSolver::CG);

  dealiiLinearSolverObj.solve(
      poissonSolverObj, d_dftParams.absLinearSolverTolerance,
      d_dftParams.maxLinearSolverIterations, d_dftParams.verbosity);

  dftfe::dftUtils::constraintMatrixInfo<dftfe::utils::MemorySpace::HOST>
      constraintsMatrixDataInfoElectro;
  constraintsMatrixDataInfoElectro.initialize(
      matrixFreeElectro.get_vector_partitioner(dofHandlerElectroIndex),
      d_constraintMatrixElectroHartree);

  pcout << " norm of vHartreeElectroNodal before distribute = "
        << vHartreeElectroNodal.l2_norm() << "\n";

  vHartreeElectroNodal.update_ghost_values();
  constraintsMatrixDataInfoElectro.distribute(vHartreeElectroNodal, 1);
  vHartreeElectroNodal.update_ghost_values();

  pcout << " norm of vHartreeElectroNodal after distribute = "
        << vHartreeElectroNodal.l2_norm() << "\n";
  /*
       pcout<<"writing hartree pot output\n";
          dealii::DataOut<3> data_out_hartree;

          data_out_hartree.attach_dof_handler(*d_dofHandlerElectroDFTClass);

          std::string outputVecName1 = "hartree pot";
          data_out_hartree.add_data_vector(vHartreeElectroNodal,outputVecName1);


          data_out_hartree.build_patches();
          data_out_hartree.write_vtu_with_pvtu_record("./", "hartreePot",
     0,d_mpiComm_domain,2, 4);
  */

  const unsigned int numQuadPointsElectroPerCell =
      quadratureRuleElectroRhs.size();

  pcout << " numQuadPointsElectroPerCell = " << numQuadPointsElectroPerCell
        << "\n";

  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      quadratureGradValueData;
  hartreeQuadData.resize(numQuadPointsElectroPerCell * nLocalCellsElectro);

  d_dftBaseClass->interpolateElectroNodalDataToQuadratureDataGeneral(
      basisOpeElectroHost, dofHandlerElectroIndex, quadratureElectroRhsId,
      vHartreeElectroNodal, hartreeQuadData, quadratureGradValueData,
      false // isEvaluateGradData
  );
  double hartreeQuadDataSum = 0.0;
  for (unsigned int iQuad = 0;
       iQuad < numQuadPointsElectroPerCell * nLocalCellsElectro; iQuad++) {
    hartreeQuadDataSum += hartreeQuadData.data()[iQuad];
  }

  MPI_Allreduce(MPI_IN_PLACE, &hartreeQuadDataSum, 1,
                dftfe::dataTypes::mpi_type_id(&hartreeQuadDataSum), MPI_SUM,
                d_mpiComm_domain);

  pcout << " hartreeQuadDataSum = " << hartreeQuadDataSum << "\n";
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTEngine<FEOrder, FEOrderElectro,
                      memorySpace>::setPotBaseExactNuclear() {
  unsigned int numElectrons = d_dftBaseClass->getNumElectrons();
  // set up the constraints and the matrixFreeObj
  pcout << " numElectrons = " << numElectrons << "\n";

  // TODO does not assume periodic BCs.
  std::vector<std::vector<double>> atomLocations =
      d_dftBaseClass->getAtomLocationsCart();

  dealii::IndexSet locallyRelevantDofsElectro;

  dealii::DoFTools::extract_locally_relevant_dofs(*d_dofHandlerElectroDFTClass,
                                                  locallyRelevantDofsElectro);

  dealii::AffineConstraints<double> d_constraintMatrixElectroHartree;
  // TODO periodic boundary conditions are not included
  d_constraintMatrixElectroHartree.clear();
  d_constraintMatrixElectroHartree.reinit(locallyRelevantDofsElectro);
  dealii::DoFTools::make_hanging_node_constraints(
      *d_dofHandlerElectroDFTClass, d_constraintMatrixElectroHartree);

  const unsigned int dofs_per_cell =
      d_dofHandlerElectroDFTClass->get_fe().dofs_per_cell;
  const unsigned int faces_per_cell = dealii::GeometryInfo<3>::faces_per_cell;
  const unsigned int dofs_per_face =
      d_dofHandlerElectroDFTClass->get_fe().dofs_per_face;

  std::vector<dealii::types::global_dof_index> cellGlobalDofIndices(
      dofs_per_cell);
  std::vector<dealii::types::global_dof_index> iFaceGlobalDofIndices(
      dofs_per_face);

  std::vector<bool> dofs_touched(d_dofHandlerElectroDFTClass->n_dofs(), false);

  dealii::MappingQGeneric<3, 3> mapping(1);
  std::map<dealii::types::global_dof_index, dealii::Point<3, double>>
      dof_coords_electro;
  dealii::DoFTools::map_dofs_to_support_points<3, 3>(
      mapping, *d_dofHandlerElectroDFTClass, dof_coords_electro);

  dealii::DoFHandler<3>::active_cell_iterator
      cellElectro = d_dofHandlerElectroDFTClass->begin_active(),
      endElectro = d_dofHandlerElectroDFTClass->end();
  for (; cellElectro != endElectro; ++cellElectro)
    if (cellElectro->is_locally_owned() || cellElectro->is_ghost()) {
      cellElectro->get_dof_indices(cellGlobalDofIndices);
      for (unsigned int iFace = 0; iFace < faces_per_cell; ++iFace) {
        const unsigned int boundaryId = cellElectro->face(iFace)->boundary_id();
        if (boundaryId == 0) {
          cellElectro->face(iFace)->get_dof_indices(iFaceGlobalDofIndices);
          for (unsigned int iFaceDof = 0; iFaceDof < dofs_per_face;
               ++iFaceDof) {
            const dealii::types::global_dof_index nodeId =
                iFaceGlobalDofIndices[iFaceDof];
            if (dofs_touched[nodeId])
              continue;
            dofs_touched[nodeId] = true;
            if (!d_constraintMatrixElectroHartree.is_constrained(nodeId)) {
              //                          pcout << " setting constraints for = "
              //                          << nodeId
              //                                << "\n";
              //                          double rad = 0.0;
              //                          rad =
              //                          dof_coords_electro[nodeId][0]*dof_coords_electro[nodeId][0];
              //                          rad +=
              //                          dof_coords_electro[nodeId][1]*dof_coords_electro[nodeId][1];
              //                          rad +=
              //                          dof_coords_electro[nodeId][2]*dof_coords_electro[nodeId][2];
              //                          rad = std::sqrt(rad);
              //                          if( rad < 1e-6)
              //                            {
              //                              pcout<<"Errorrrrrr in
              //                              rad \n";
              //                            }
              double nodalConstraintVal = 0.0;
              for (unsigned int iAtom = 0; iAtom < atomLocations.size();
                   iAtom++) {
                double rad = 0.0;
                rad +=
                    (atomLocations[iAtom][2] - dof_coords_electro[nodeId][0]) *
                    (atomLocations[iAtom][2] - dof_coords_electro[nodeId][0]);
                rad +=
                    (atomLocations[iAtom][3] - dof_coords_electro[nodeId][1]) *
                    (atomLocations[iAtom][3] - dof_coords_electro[nodeId][1]);
                rad +=
                    (atomLocations[iAtom][4] - dof_coords_electro[nodeId][2]) *
                    (atomLocations[iAtom][4] - dof_coords_electro[nodeId][2]);
                rad = std::sqrt(rad);
                if (d_dftParams.isPseudopotential)
                  nodalConstraintVal += atomLocations[iAtom][1] / rad;
                else
                  nodalConstraintVal += atomLocations[iAtom][0] / rad;
              }
              d_constraintMatrixElectroHartree.add_line(nodeId);
              d_constraintMatrixElectroHartree.set_inhomogeneity(
                  nodeId, nodalConstraintVal);
              //                          d_constraintMatrixElectroHartree.set_inhomogeneity(nodeId,
              //                          0.0);
            } // non-hanging node check
          }   // Face dof loop
        }     // non-periodic boundary id
      }       // Face loop
    }         // cell locally owned
  d_constraintMatrixElectroHartree.close();

  const dealii::Quadrature<3> &quadratureRuleElectroRhs =
      d_dftMatrixFreeDataElectro->get_quadrature(d_dftElectroRhsQuadIndex);

  const dealii::Quadrature<3> &quadratureRuleElectroAx =
      d_dftMatrixFreeDataElectro->get_quadrature(d_dftElectroAxQuadIndex);

  std::vector<Quadrature<1>> quadratureVector(0);

  unsigned int quadRhsVal = std::cbrt(quadratureRuleElectroRhs.size());
  pcout << " first quad val  = " << quadRhsVal << "\n";

  unsigned int quadAxVal = std::cbrt(quadratureRuleElectroAx.size());
  pcout << " second quad val  = " << quadAxVal << "\n";

  quadratureVector.push_back(QGauss<1>(quadRhsVal));
  quadratureVector.push_back(QGauss<1>(quadAxVal));

  typename MatrixFree<3>::AdditionalData additional_data;
  // comment this if using deal ii version 9
  // additional_data.mpi_communicator = d_mpiCommParent;
  additional_data.tasks_parallel_scheme =
      MatrixFree<3>::AdditionalData::partition_partition;
  additional_data.mapping_update_flags =
      update_values | update_gradients | update_JxW_values;

  std::vector<const dealii::DoFHandler<3> *> matrixFreeDofHandlerVectorInput;
  matrixFreeDofHandlerVectorInput.push_back(d_dofHandlerElectroDFTClass);

  std::vector<const dealii::AffineConstraints<double> *>
      constraintsVectorElectro;

  constraintsVectorElectro.push_back(&d_constraintMatrixElectroHartree);

  // TODO check if passing the quadrature rules this way is correct
  dealii::MatrixFree<3, double> matrixFreeElectro;
  matrixFreeElectro.reinit(
      dealii::MappingQ1<3, 3>(), matrixFreeDofHandlerVectorInput,
      constraintsVectorElectro, quadratureVector, additional_data);
  unsigned int dofHandlerElectroIndex = 0;
  unsigned int quadratureElectroRhsId = 0;
  unsigned int quadratureElectroAxId = 1;

  std::map<dealii::types::global_dof_index, double> dummyAtomMap;
  std::map<dealii::CellId, std::vector<double>> dummySmearedChargeValues;

  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      totalRhoValues;

  dftfe::distributedCPUVec<double> vHartreeElectroNodal;

  dftfe::vectorTools::createDealiiVector<double>(
      matrixFreeElectro.get_vector_partitioner(dofHandlerElectroIndex), 1,
      vHartreeElectroNodal);

  vHartreeElectroNodal = 0.0;

  std::vector<unsigned int> constraintedDofsInverse;
  constraintedDofsInverse =
      matrixFreeElectro.get_constrained_dofs(dofHandlerElectroIndex);
  unsigned int sizeLocalInversePot = constraintedDofsInverse.size();

  MPI_Allreduce(MPI_IN_PLACE, &sizeLocalInversePot, 1,
                dftfe::dataTypes::mpi_type_id(&sizeLocalInversePot), MPI_SUM,
                d_mpiComm_domain);

  pcout << " no of constrained inverse  = " << sizeLocalInversePot << "\n";

  dealii::DoFHandler<3>::active_cell_iterator cell = d_dofHandlerDFTClass
                                                         ->begin_active(),
                                              endc =
                                                  d_dofHandlerDFTClass->end();

  unsigned int iElem = 0;

  const dealii::Quadrature<3> &quadratureRule =
      d_dftMatrixFreeData->get_quadrature(d_dftQuadIndex);
  const unsigned int numQuadPointsPerCell = quadratureRule.size();

  const unsigned int numQuadPointsElectroPerCell =
      quadratureRuleElectroRhs.size();

  const unsigned int nLocalCellsElectro = matrixFreeElectro.n_physical_cells();

  totalRhoValues.resize(numQuadPointsPerCell * nLocalCellsElectro);
  unsigned int spinIndex1 = 0;
  unsigned int spinIndex2 = (d_numSpins == 2) ? 1 : 0;

  for (; cell != endc; ++cell)
    if (cell->is_locally_owned()) {
      //          std::vector<double> cellLevelQuadInput;
      for (unsigned int iQuad = 0; iQuad < numQuadPointsPerCell; iQuad++) {
        totalRhoValues[iElem * numQuadPointsPerCell + iQuad] =
            d_rhoTarget[spinIndex1][iElem * numQuadPointsPerCell + iQuad] +
            d_rhoTarget[spinIndex2][iElem * numQuadPointsPerCell + iQuad];
      }
      //          totalRhoValues[cell->id()] = cellLevelQuadInput;
      iElem++;
    }

  std::shared_ptr<dftfe::basis::FEBasisOperations<
      dftfe::dataTypes::number, double, dftfe::utils::MemorySpace::HOST>>
      basisOpeElectroHost = std::make_shared<dftfe::basis::FEBasisOperations<
          dftfe::dataTypes::number, double, dftfe::utils::MemorySpace::HOST>>(
          d_blasWrapperHost);
  std::vector<dftfe::basis::UpdateFlags> updateFlags;
  updateFlags.resize(2);
  updateFlags[0] = dftfe::basis::update_jxw | dftfe::basis::update_values |
                   dftfe::basis::update_transpose;

  updateFlags[1] = dftfe::basis::update_jxw | dftfe::basis::update_values |
                   dftfe::basis::update_transpose;

  std::vector<unsigned int> quadratureVectorId;
  quadratureVectorId.resize(2);
  quadratureVectorId[0] = quadratureElectroRhsId;
  quadratureVectorId[1] = quadratureElectroAxId;
  basisOpeElectroHost->init(matrixFreeElectro, constraintsVectorElectro,
                            dofHandlerElectroIndex, quadratureVectorId,
                            updateFlags);

  unsigned int numCellsTempSize = 100;
  numCellsTempSize = std::min(numCellsTempSize, nLocalCellsElectro);
  basisOpeElectroHost->reinit(1, numCellsTempSize, quadratureElectroAxId);

  pcout << " solving possion in the pot base \n";

  dftfe::poissonSolverProblem<FEOrder, FEOrderElectro> poissonSolverObj(
      d_mpiComm_domain);
  poissonSolverObj.reinit(basisOpeElectroHost, vHartreeElectroNodal,
                          d_constraintMatrixElectroHartree,
                          dofHandlerElectroIndex, quadratureElectroRhsId,
                          quadratureElectroAxId, dummyAtomMap,
                          dummySmearedChargeValues, 0, totalRhoValues,
                          true,  // isComputeDiagonalA
                          false, // isComputeMeanValueConstraints
                          false, // smearedNuclearCharges
                          true,  // isRhoValues
                          false, // isGradSmearedChargeRhs
                          0,     // smearedChargeGradientComponentId
                          false, // storeSmearedChargeRhs
                          false, // reuseSmearedChargeRhs
                          true); // reinitializeFastConstraints

  dftfe::dealiiLinearSolver dealiiLinearSolverObj(
      d_mpiComm_parent, d_mpiComm_domain, dftfe::dealiiLinearSolver::CG);

  dealiiLinearSolverObj.solve(
      poissonSolverObj, d_dftParams.absLinearSolverTolerance,
      d_dftParams.maxLinearSolverIterations, d_dftParams.verbosity);

  dftfe::dftUtils::constraintMatrixInfo<dftfe::utils::MemorySpace::HOST>
      constraintsMatrixDataInfoElectro;
  constraintsMatrixDataInfoElectro.initialize(
      matrixFreeElectro.get_vector_partitioner(dofHandlerElectroIndex),
      d_constraintMatrixElectroHartree);

  vHartreeElectroNodal.update_ghost_values();
  constraintsMatrixDataInfoElectro.distribute(vHartreeElectroNodal, 1);
  vHartreeElectroNodal.update_ghost_values();

  d_potBaseQuadData.resize(
      d_numSpins,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
          nLocalCellsElectro * numQuadPointsElectroPerCell));

  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      quadratureValueData;
  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      quadratureGradValueData;

  quadratureValueData.resize(nLocalCellsElectro * numQuadPointsElectroPerCell);
  cellElectro = d_dofHandlerElectroDFTClass->begin_active();
  endElectro = d_dofHandlerElectroDFTClass->end();

  d_dftBaseClass->interpolateElectroNodalDataToQuadratureDataGeneral(
      basisOpeElectroHost, dofHandlerElectroIndex, quadratureElectroRhsId,
      vHartreeElectroNodal, quadratureValueData, quadratureGradValueData,
      false // isEvaluateGradData
  );

  dealii::FEValues<3> fe_valuesElectro(d_dofHandlerElectroDFTClass->get_fe(),
                                       quadratureRuleElectroRhs,
                                       dealii::update_quadrature_points);

  std::copy(quadratureValueData.begin(), quadratureValueData.end(),
            d_potBaseQuadData[0].begin());

  unsigned int iElemElectro = 0;
  cellElectro = d_dofHandlerElectroDFTClass->begin_active();
  endElectro = d_dofHandlerElectroDFTClass->end();
  for (; cellElectro != endElectro; ++cellElectro)
    if (cellElectro->is_locally_owned()) {
      fe_valuesElectro.reinit(cellElectro);

      for (unsigned int iQuad = 0; iQuad < numQuadPointsElectroPerCell;
           iQuad++) {
        dealii::Point<3, double> qPointVal =
            fe_valuesElectro.quadrature_point(iQuad);
        for (unsigned int iAtom = 0; iAtom < atomLocations.size(); iAtom++) {
          double rad = 0.0;
          rad += (atomLocations[iAtom][2] - qPointVal[0]) *
                 (atomLocations[iAtom][2] - qPointVal[0]);
          rad += (atomLocations[iAtom][3] - qPointVal[1]) *
                 (atomLocations[iAtom][3] - qPointVal[1]);
          rad += (atomLocations[iAtom][4] - qPointVal[2]) *
                 (atomLocations[iAtom][4] - qPointVal[2]);
          rad = std::sqrt(rad);
          if (d_dftParams.isPseudopotential)
            d_potBaseQuadData[0][iElemElectro * numQuadPointsElectroPerCell +
                                 iQuad] -= atomLocations[iAtom][1] / rad;
          else
            d_potBaseQuadData[0][iElemElectro * numQuadPointsElectroPerCell +
                                 iQuad] -= atomLocations[iAtom][0] / rad;
        }
      }
      iElemElectro++;
    }

  if (d_dftParams.confiningPotential) {
    auto confiningPot = d_dftBaseClass->getConfiningPotential();
    confiningPot.addConfiningPotential(d_potBaseQuadData[0]);
  }

  if (d_numSpins == 2) {
    std::copy(d_potBaseQuadData[0].begin(), d_potBaseQuadData[0].end(),
              d_potBaseQuadData[1].begin());
  }
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTEngine<FEOrder, FEOrderElectro,
                      memorySpace>::setPotBasePoissonNuclear() {
  unsigned int numElectrons = d_dftBaseClass->getNumElectrons();
  // set up the constraints and the matrixFreeObj
  pcout << " numElectrons = " << numElectrons << "\n";

  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      totalRhoValues;

  dftfe::distributedCPUVec<double> vTotalElectroNodal;

  dealii::DoFHandler<3>::active_cell_iterator cell = d_dofHandlerDFTClass
                                                         ->begin_active(),
                                              endc =
                                                  d_dofHandlerDFTClass->end();
  unsigned int iElem = 0;

  const dealii::Quadrature<3> &quadratureRule =
      d_dftMatrixFreeData->get_quadrature(d_dftQuadIndex);
  const unsigned int numQuadPointsPerCell = quadratureRule.size();
  const unsigned int nLocallyOwnedCells =
      d_dftMatrixFreeData->n_physical_cells();
  totalRhoValues.resize(numQuadPointsPerCell * nLocallyOwnedCells);
  unsigned int spinIndex1 = 0;
  unsigned int spinIndex2 = (d_numSpins == 2) ? 1 : 0;

  double sumTotalRho = 0.0;
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned()) {
      //          std::vector<double> cellLevelQuadInput;
      for (unsigned int iQuad = 0; iQuad < numQuadPointsPerCell; iQuad++) {
        totalRhoValues[iElem * numQuadPointsPerCell + iQuad] =
            d_rhoTarget[spinIndex1][iElem * numQuadPointsPerCell + iQuad] +
            d_rhoTarget[spinIndex2][iElem * numQuadPointsPerCell + iQuad];

        sumTotalRho += totalRhoValues[iElem * numQuadPointsPerCell + iQuad];
      }
      //          totalRhoValues[cell->id()] = cellLevelQuadInput;
      iElem++;
    }

  MPI_Allreduce(MPI_IN_PLACE, &sumTotalRho, 1, MPI_DOUBLE, MPI_SUM,
                d_mpiComm_domain);
  pcout << " sum total in solve phi = " << sumTotalRho << "\n";
  pcout << " solving poisson in the pot nuclear \n";

  solvePhiTotalAllElectronNonPeriodic(vTotalElectroNodal, totalRhoValues,
                                      d_mpiComm_parent, d_mpiComm_domain);

  pcout << " vTotalElectroNodal norm solvePhiTotalAllElectronNonPeriodic = "
        << vTotalElectroNodal.l2_norm() << "\n";
  //    vTotalElectroNodal.update_ghost_values();

  const dealii::Quadrature<3> &quadratureRuleElectroRhs =
      d_dftMatrixFreeDataElectro->get_quadrature(d_dftElectroRhsQuadIndex);
  const unsigned int numQuadPointsElectroPerCell =
      quadratureRuleElectroRhs.size();

  const unsigned int nLocalCellsElectro =
      d_dftMatrixFreeDataElectro->n_physical_cells();

  d_potBaseQuadData.resize(
      d_numSpins,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
          nLocalCellsElectro * numQuadPointsElectroPerCell));

  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      quadratureValueData;
  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      quadratureGradValueData;

  quadratureValueData.resize(nLocalCellsElectro * numQuadPointsElectroPerCell);
  dealii::DoFHandler<3>::active_cell_iterator
      cellElectro = d_dofHandlerElectroDFTClass->begin_active(),
      endElectro = d_dofHandlerElectroDFTClass->end();

  d_dftBaseClass->interpolateElectroNodalDataToQuadratureDataGeneral(
      d_basisOperationsElectroHost, d_dftElectroDoFHandlerIndex,
      d_dftElectroRhsQuadIndex, vTotalElectroNodal, quadratureValueData,
      quadratureGradValueData,
      false // isEvaluateGradData
  );

  double sumQuadDat = 0.0;

  for (unsigned int iQuad = 0;
       iQuad < numQuadPointsElectroPerCell * nLocalCellsElectro; iQuad++) {
    sumQuadDat += quadratureValueData.data()[iQuad];
  }

  MPI_Allreduce(MPI_IN_PLACE, &sumQuadDat, 1, MPI_DOUBLE, MPI_SUM,
                d_mpiComm_domain);

  pcout << " sumQuadDat = " << sumQuadDat << "\n";

  dealii::FEValues<3> fe_valuesElectro(d_dofHandlerElectroDFTClass->get_fe(),
                                       quadratureRuleElectroRhs,
                                       dealii::update_quadrature_points);

  std::copy(quadratureValueData.begin(), quadratureValueData.end(),
            d_potBaseQuadData[0].begin());

  if (d_dftParams.confiningPotential) {
    auto confiningPot = d_dftBaseClass->getConfiningPotential();
    confiningPot.addConfiningPotential(d_potBaseQuadData[0]);
  }

  if (d_numSpins == 2) {
    std::copy(d_potBaseQuadData[0].begin(), d_potBaseQuadData[0].end(),
              d_potBaseQuadData[1].begin());
  }
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTEngine<FEOrder, FEOrderElectro, memorySpace>::
    solvePhiTotalAllElectronNonPeriodic(
        dftfe::distributedCPUVec<double> &x,
        const dftfe::utils::MemoryStorage<
            double, dftfe::utils::MemorySpace::HOST> &rhoValues,
        const MPI_Comm &mpiComm_parent, const MPI_Comm &mpiComm_domain) {
  // create the poisson solver problem
  dftfe::poissonSolverProblem<FEOrder, FEOrderElectro> poissonSolverObj(
      mpiComm_domain);

  // create the dealii solver

  dftfe::dealiiLinearSolver CGSolver(mpiComm_parent, mpiComm_domain,
                                     dftfe::dealiiLinearSolver::CG);

  dftfe::vectorTools::createDealiiVector<double>(
      d_dftMatrixFreeDataElectro->get_vector_partitioner(
          d_dftElectroDoFHandlerIndex),
      1, x);

  x = 0.0;

  if (d_dftParams.multipoleBoundaryConditions) {
    d_dftBaseClass->computeMultipoleMoments(
        d_basisOperationsElectroHost, d_dftElectroRhsQuadIndex, rhoValues,
        &(d_dftBaseClass->getBQuadValuesAllAtoms()));
    d_dftBaseClass->updatePRefinedConstraints();
  }

  poissonSolverObj.reinit(d_basisOperationsElectroHost, x,
                          *(d_dftBaseClass->getConstraintsVectorElectro()),
                          d_dftElectroDoFHandlerIndex, d_dftElectroRhsQuadIndex,
                          d_dftElectroAxQuadIndex,
                          d_dftBaseClass->getAtomNodeToChargeMap(),
                          d_dftBaseClass->getBQuadValuesAllAtoms(),
                          d_dftBaseClass->getSmearedChargeQuadratureIdElectro(),
                          rhoValues, // rhoValues,
                          true,      // isComputeDiagonalA
                          false,     // isComputeMeanValueConstraints,
                          d_dftParams.smearedNuclearCharges,
                          true,  // isRhoValues
                          false, // isGradSmearedChargeRhs
                          0,
                          false, // storeSmearedChargeRhs
                          false, // reuseSmearedChargeRhs
                          true   // reinitializeFastConstraints
  );

  // use the CG solver
  CGSolver.solve(poissonSolverObj, d_dftParams.absLinearSolverTolerance,
                 d_dftParams.maxLinearSolverIterations, d_dftParams.verbosity);
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTEngine<FEOrder, FEOrderElectro, memorySpace>::setPotBase() {
  setPotBasePoissonNuclear();
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTEngine<FEOrder, FEOrderElectro, memorySpace>::
    readVxcDataFromFile(
        std::vector<dftfe::distributedCPUVec<double>> &vxcChildNodes) {
  vxcChildNodes.resize(d_numSpins);
  dftfe::vectorTools::createDealiiVector<double>(
      d_matrixFreeDataVxc.get_vector_partitioner(d_dofHandlerVxcIndex), 1,
      vxcChildNodes[0]);
  vxcChildNodes[0] = 0.0;

  if (d_numSpins == 2) {
    vxcChildNodes[1].reinit(vxcChildNodes[0]);
  }

  std::map<dealii::types::global_dof_index, dealii::Point<3, double>>
      dof_coord_child;
  dealii::DoFTools::map_dofs_to_support_points<3, 3>(
      dealii::MappingQ1<3, 3>(), d_dofHandlerTriaVxc, dof_coord_child);
  dealii::types::global_dof_index numberDofsChild =
      d_dofHandlerTriaVxc.n_dofs();

  std::vector<coordinateValues> inputDataFromFile;
  inputDataFromFile.resize(numberDofsChild);

  const std::string filename = d_inverseDFTParams.vxcDataFolder + "/" +
                               d_inverseDFTParams.fileNameReadVxcPostFix;
  std::ifstream vxcInputFile(filename);

  double nodalValue = 0.0;
  double xcoordValue = 0.0;
  double ycoordValue = 0.0;
  double zcoordValue = 0.0;
  double fieldValue0 = 0.0;
  double fieldValue1 = 0.0;

  for (dealii::types::global_dof_index iNode = 0; iNode < numberDofsChild;
       iNode++) {
    vxcInputFile >> nodalValue;
    vxcInputFile >> xcoordValue;
    vxcInputFile >> ycoordValue;
    vxcInputFile >> zcoordValue;
    vxcInputFile >> fieldValue0;
    if (d_numSpins == 2) {
      vxcInputFile >> fieldValue1;
    }
    if (vxcChildNodes[0].in_local_range(nodalValue)) {
      double distBetweenNodes = 0.0;
      distBetweenNodes += (xcoordValue - dof_coord_child[iNode][0]) *
                          (xcoordValue - dof_coord_child[iNode][0]);
      distBetweenNodes += (ycoordValue - dof_coord_child[iNode][1]) *
                          (ycoordValue - dof_coord_child[iNode][1]);
      distBetweenNodes += (zcoordValue - dof_coord_child[iNode][2]) *
                          (zcoordValue - dof_coord_child[iNode][2]);
      distBetweenNodes = std::sqrt(distBetweenNodes);
      if (distBetweenNodes > 1e-3) {
        std::cout << " Errorr while reading data global nodes do not match \n";
      }

      vxcChildNodes[0](iNode) = fieldValue0;
      if (d_numSpins == 2) {
        vxcChildNodes[1](iNode) = fieldValue1;
      }
    }
  }
  vxcInputFile.close();
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTEngine<FEOrder, FEOrderElectro, memorySpace>::readVxcInput() {
  unsigned int totalLocallyOwnedCellsVxc =
      d_matrixFreeDataVxc.n_physical_cells();

  const unsigned int numQuadPointsPerCellInVxc = d_gaussQuadVxc.size();

  double spinFactor = (d_dftParams.spinPolarized == 1) ? 1.0 : 2.0;

  unsigned int locallyOwnedDofs = d_dofHandlerDFTClass->n_locally_owned_dofs();

  d_vxcInitialChildNodes.resize(d_numSpins);
  readVxcDataFromFile(d_vxcInitialChildNodes);

  d_targetPotValuesParentQuadData.resize(d_numSpins);

  unsigned int totalOwnedCellsPsi = d_dftMatrixFreeData->n_physical_cells();

  const dealii::Quadrature<3> &quadratureRulePsi =
      d_dftMatrixFreeData->get_quadrature(d_dftQuadIndex);

  unsigned int numQuadPointsPerPsiCell = quadratureRulePsi.size();

  for (unsigned int spinIndex = 0; spinIndex < d_numSpins; ++spinIndex) {
    d_targetPotValuesParentQuadData[spinIndex].resize(totalOwnedCellsPsi);
    for (unsigned int iCell = 0; iCell < totalOwnedCellsPsi; iCell++) {
      // TODO set the correct values. For now set to a dummy value
      d_targetPotValuesParentQuadData[spinIndex][iCell].resize(
          numQuadPointsPerPsiCell, 0.0);
    }
  }

  /*
  // TODO unComment this to set the adjoint constraints
  dftfe::distributedCPUVec<double> rhoInputTotal;
  dftfe::vectorTools::createDealiiVector<double>(
      d_dftMatrixFreeData->get_vector_partitioner(d_dftDensityDoFHandlerIndex),
      1, rhoInputTotal);
  rhoInputTotal = 0.0;

  unsigned int totalLocallyOwnedCellsPsi =
      d_dftMatrixFreeData->n_physical_cells();

  unsigned int numLocallyOwnedDofsPsi =
      d_dofHandlerDFTClass->n_locally_owned_dofs();
  unsigned int numDofsPerCellPsi = d_dofHandlerDFTClass->get_fe().dofs_per_cell;

  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      rhoValues;
  rhoValues.resize(numQuadPointsPerPsiCell * totalLocallyOwnedCellsPsi);

  typename DoFHandler<3>::active_cell_iterator cellPsiPtr =
      d_dofHandlerDFTClass->begin_active();
  typename DoFHandler<3>::active_cell_iterator endcellPsiPtr =
      d_dofHandlerDFTClass->end();

  unsigned int iElem = 0;
  unsigned int spinIndex1 = 0;
  unsigned int spinIndex2 = 0;
  if (d_numSpins == 2) {
    spinIndex2 = 1;
  }
  for (; cellPsiPtr != endcellPsiPtr; ++cellPsiPtr) {
    if (cellPsiPtr->is_locally_owned()) {
      for (unsigned int iQuad = 0; iQuad < numQuadPointsPerPsiCell; iQuad++) {
        rhoValues[iElem * numQuadPointsPerPsiCell + iQuad] =
            d_rhoTarget[spinIndex1][iElem * numQuadPointsPerPsiCell + iQuad] +
            d_rhoTarget[spinIndex2][iElem * numQuadPointsPerPsiCell + iQuad];
      }
      iElem++;
    }
  }

  d_dftBaseClass->l2ProjectionQuadToNodal(
      d_basisOperationsHost, *d_constraintDFTClass, d_dftDensityDoFHandlerIndex,
      d_dftQuadIndex, rhoValues, rhoInputTotal);
  rhoInputTotal.update_ghost_values();
  d_constraintDFTClass->distribute(rhoInputTotal);
  rhoInputTotal.update_ghost_values();

  setAdjointBoundaryCondition(rhoInputTotal);
  */
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTEngine<FEOrder, FEOrderElectro, memorySpace>::interpolateVxc() {


    d_vxcInitialChildNodes[0].update_ghost_values();
    d_constraintMatrixVxc.distribute(d_vxcInitialChildNodes[0]);
    d_vxcInitialChildNodes[0].update_ghost_values();

    std::vector<dftfe::distributedCPUVec<double>> vxcNodalParentMesh;
    projectVxcToParentMesh(vxcNodalParentMesh,
                           d_vxcInitialChildNodes);

    if (d_inverseDFTParams.writeVtuFile)
    {
        const std::string filename = d_inverseDFTParams.fileNameWriteVxcPostProcess;
        const std::string vtuFilename = filename + "_vtuOutput";

        DataOutBase::VtkFlags flags;
        flags.write_higher_order_cells = true;

        dealii::DataOut<3> data_out_vxc;

        data_out_vxc.set_flags(flags);
        data_out_vxc.attach_dof_handler(*d_dofHandlerDFTClass);

	       std::string outputVecName1 = "Vxc alpha";
        data_out_vxc.add_data_vector(vxcNodalParentMesh[0],outputVecName1);
	if(d_numSpins == 2)
	{
		std::string outputVecName1 = "Vxc beta";
        data_out_vxc.add_data_vector(vxcNodalParentMesh[1],outputVecName1);
	}

        data_out_vxc.build_patches(dealii::MappingQ1<3, 3>(), FEOrder);
        data_out_vxc.write_vtu_with_pvtu_record("./", vtuFilename,
                                                0,d_mpiComm_domain,2, 4);

    }

    if(d_inverseDFTParams.writeToPoints)
    {
        std::vector<std::vector<double>> targetPts;
        // if (d_inverseDFTParams.readPointsFromFile)
        //{
        //
        //}
        // else
        //{
        unsigned int numPointsX = d_inverseDFTParams.numPointsX;
        unsigned int numPointsY = d_inverseDFTParams.numPointsY;
        unsigned int numPointsZ = d_inverseDFTParams.numPointsZ;

        double startingX = d_inverseDFTParams.startX;
        double endingX = d_inverseDFTParams.endX;

        std::vector<double> x_coord(numPointsX, 0.0);
        x_coord[0] = startingX;
        x_coord[numPointsX - 1] = endingX;
        if ((std::abs(startingX - endingX) < 1e-6) && (numPointsX != 1)) {
            AssertThrow(false, ExcMessage(" x coords are too close to interpolate "));
        }
        double dx = (endingX - startingX) / (numPointsX - 1);

        AssertThrow((dx > 0.0) || (numPointsX == 1), ExcMessage(" dx is negative"));
        for (unsigned int iCoord = 1; iCoord < numPointsX - 1; iCoord++) {
            x_coord[iCoord] = x_coord[iCoord - 1] + dx;
        }

        double startingY = d_inverseDFTParams.startY;
        double endingY = d_inverseDFTParams.endY;

        std::vector<double> y_coord(numPointsY, 0.0);
        y_coord[0] = startingY;
        y_coord[numPointsY - 1] = endingY;
        if ((std::abs(startingY - endingY) < 1e-6) && (numPointsY != 1)) {
            AssertThrow(false, ExcMessage(" y coords are too close to interpolate "));
        }
        double dy = (endingY - startingY) / (numPointsY - 1);

        AssertThrow((dy > 0.0) || (numPointsY == 1), ExcMessage(" dy is negative"));
        for (unsigned int iCoord = 1; iCoord < numPointsY - 1; iCoord++) {
            y_coord[iCoord] = y_coord[iCoord - 1] + dy;
        }

        double startingZ = d_inverseDFTParams.startZ;
        double endingZ = d_inverseDFTParams.endZ;

        std::vector<double> z_coord(numPointsZ, 0.0);
        z_coord[0] = startingZ;
        z_coord[numPointsZ - 1] = endingZ;
        if ((std::abs(startingZ - endingZ) < 1e-6) && (numPointsZ != 1)) {
            AssertThrow(false, ExcMessage(" z coords are too close to interpolate "));
        }
        double dz = (endingZ - startingZ) / (numPointsZ - 1);

        AssertThrow((dz > 0.0) || (numPointsZ == 1), ExcMessage(" dz is negative"));
        for (unsigned int iCoord = 1; iCoord < numPointsZ - 1; iCoord++) {
            z_coord[iCoord] = z_coord[iCoord - 1] + dz;
        }

        unsigned int totalNumPoints = numPointsX * numPointsY * numPointsZ;

        // TODO a better domain decomposition would be cubic
        // but I am doing linear for simplicity

        int thisRank, numRank;
        MPI_Comm_rank(d_mpiComm_domain, &thisRank);
        MPI_Comm_size(d_mpiComm_domain, &numRank);

        unsigned int numPointsInProc = totalNumPoints / numRank;
        if (thisRank == numRank - 1) {
            numPointsInProc = numPointsInProc + totalNumPoints % numRank;
        }

        unsigned int startingIndex = (totalNumPoints / numRank) * thisRank;

        targetPts.resize(numPointsInProc, std::vector<double>(3, 0.0));

        for (unsigned int index = startingIndex;
             index < startingIndex + numPointsInProc; index++) {
            unsigned int xIndex = index / (numPointsZ * numPointsY);
            unsigned int yIndex =
                    (index - xIndex * numPointsZ * numPointsY) / (numPointsZ);
            unsigned int zIndex =
                    (index - xIndex * numPointsZ * numPointsY) % (numPointsZ);

            targetPts[index - startingIndex][0] = x_coord[xIndex];
            targetPts[index - startingIndex][1] = y_coord[yIndex];
            targetPts[index - startingIndex][2] = z_coord[zIndex];
        }
        //}

        unsigned int totalOwnedCellsPsi = d_dftMatrixFreeData->n_physical_cells();

        const dealii::Quadrature<3> &quadratureRulePsi =
                d_dftMatrixFreeData->get_quadrature(d_dftQuadIndex);

        unsigned int numQuadPointsPerPsiCell = quadratureRulePsi.size();

        const dealii::FiniteElement<3> &feMesh = d_dofHandlerDFTClass->get_fe();
        std::vector<unsigned int> numberDofsPerCell;
        numberDofsPerCell.resize(totalOwnedCellsPsi);

        std::vector<std::shared_ptr<const dftfe::utils::Cell<3>>> srcCellsMesh(0);

        std::vector<
        std::shared_ptr<dftfe::InterpolateFromCellToLocalPoints<memorySpace>>>
        interpolateLocalMesh(0);

        dealii::DoFHandler<3>::active_cell_iterator cellPsi = d_dofHandlerDFTClass
                ->begin_active(),
                endcPsi =
                d_dofHandlerDFTClass->end();

        // iterate through child cells
        dftfe::size_type iElemIndex = 0;
        for (; cellPsi != endcPsi; cellPsi++) {
            if (cellPsi->is_locally_owned()) {
                numberDofsPerCell[iElemIndex] =
                        d_dofHandlerDFTClass->get_fe().dofs_per_cell;
                auto srcCellPtr =
                        std::make_shared<dftfe::utils::FECell<3>>(cellPsi, feMesh);
                srcCellsMesh.push_back(srcCellPtr);

                interpolateLocalMesh.push_back(
                        std::make_shared<
                        dftfe::InterpolateFromCellToLocalPoints<memorySpace>>(
                                srcCellPtr, numberDofsPerCell[iElemIndex],
                                        d_inverseDFTParams.useMemOptForTransfer));
                iElemIndex++;
            }
        }

        dftfe::InterpolateCellWiseDataToPoints<dftfe::dataTypes::number, memorySpace>
                d_meshVxctoPoints(srcCellsMesh, interpolateLocalMesh, targetPts,
                                  numberDofsPerCell, 4, d_mpiComm_domain);


	dftfe::linearAlgebra::MultiVector<double, dftfe::utils::MemorySpace::HOST>
      dummyPotVec;

	dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      d_dftMatrixFreeData->get_vector_partitioner(
          d_dftDensityDoFHandlerIndex),
      1, dummyPotVec);

        std::vector<dealii::types::global_dof_index> fullFlattenedMapParent;
        dftfe::vectorTools::computeCellLocalIndexSetMap(
                dummyPotVec.getMPIPatternP2P(), *d_dftMatrixFreeData, d_dftDensityDoFHandlerIndex,
                1, fullFlattenedMapParent);

        dftfe::utils::MemoryStorage<dealii::types::global_dof_index,
        dftfe::utils::MemorySpace::HOST>
                fullFlattenedMap;
        fullFlattenedMap.resize(fullFlattenedMapParent.size());
        fullFlattenedMap.copyFrom(fullFlattenedMapParent);

        dftfe::utils::MemoryStorage<dftfe::dataTypes::number,
        dftfe::utils::MemorySpace::HOST>
                outputQuadData;
        d_meshVxctoPoints.interpolateSrcDataToTargetPoints(
                d_blasWrapperHost, vxcNodalParentMesh[0], 1, fullFlattenedMap,
                outputQuadData, true);

        const std::string filename = d_inverseDFTParams.fileNameWriteVxcPostProcess;
        std::vector<std::shared_ptr<dftfe::dftUtils::CompositeData>> data(0);

        MPI_Barrier(d_mpiComm_domain);
        for (unsigned int index = startingIndex;
             index < startingIndex + numPointsInProc; index++) {
            std::vector<double> nodeVals(0);
            nodeVals.push_back(index);
            nodeVals.push_back(targetPts[index - startingIndex][0]);
            nodeVals.push_back(targetPts[index - startingIndex][1]);
            nodeVals.push_back(targetPts[index - startingIndex][2]);

            nodeVals.push_back(outputQuadData.data()[index - startingIndex]);
            data.push_back(std::make_shared<dftfe::dftUtils::NodalData>(nodeVals));
        }

        std::vector<dftfe::dftUtils::CompositeData *> dataRawPtrs(data.size());
        for (unsigned int i = 0; i < data.size(); ++i)
            dataRawPtrs[i] = data[i].get();
        dftfe::dftUtils::MPIWriteOnFile().writeData(dataRawPtrs, filename,
                                                    d_mpiComm_domain);

    }
    MPI_Barrier(d_mpiComm_domain);
}


    template <unsigned int FEOrder, unsigned int FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
    void InverseDFTEngine<FEOrder, FEOrderElectro, memorySpace>::
            projectVxcToParentMesh(std::vector<dftfe::distributedCPUVec<double>> &vxcInitialGuessParentMesh,
                                   std::vector<dftfe::distributedCPUVec<double>> &vxcChildNodes) {

    std::vector<dftfe::utils::MemoryStorage<dftfe::dataTypes::number,
            dftfe::utils::MemorySpace::HOST>> vxcInterpolateToParent;
    vxcInterpolateToParent.resize(d_numSpins);
        vxcInitialGuessParentMesh.resize(d_numSpins);

        unsigned int totalOwnedCellsPsi = d_dftMatrixFreeData->n_physical_cells();

        const dealii::Quadrature<3> &quadratureRulePsi =
                d_dftMatrixFreeData->get_quadrature(d_dftQuadIndex);

        unsigned int numQuadPointsPerPsiCell = quadratureRulePsi.size();

        std::vector <
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
                vxcOutputData;

        vxcOutputData.resize(
                d_numSpins,
                dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
                        totalOwnedCellsPsi * numQuadPointsPerPsiCell));


	dftfe::linearAlgebra::MultiVector<double, dftfe::utils::MemorySpace::HOST>
      dummyPotVec;

  dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      d_matrixFreeDataVxc.get_vector_partitioner(
          d_dofHandlerVxcIndex),
      1, dummyPotVec);

  std::vector<dealii::types::global_dof_index> fullFlattenedMapChild;
  dftfe::vectorTools::computeCellLocalIndexSetMap(
      dummyPotVec.getMPIPatternP2P(), d_matrixFreeDataVxc,
      d_dofHandlerVxcIndex, 1, fullFlattenedMapChild);

 dftfe::utils::MemoryStorage<dealii::types::global_dof_index,
                              dftfe::utils::MemorySpace::HOST>
      fullFlattenedMapChildMemStorage;
 fullFlattenedMapChildMemStorage.resize(fullFlattenedMapChild.size());
 fullFlattenedMapChildMemStorage.copyFrom(fullFlattenedMapChild);



        for (unsigned int spinIndex = 0; spinIndex < d_numSpins; ++spinIndex) {
            dealii::DoFHandler<3>::active_cell_iterator cellPsi = d_dofHandlerDFTClass
                    ->begin_active(),
                    endcPsi =
                    d_dofHandlerDFTClass->end();

            d_inverseDftDoFManagerObjPtr->interpolateMesh2DataToMesh1QuadPoints(
                    d_blasWrapperHost, vxcChildNodes[spinIndex], 1, fullFlattenedMapChildMemStorage,
                    vxcInterpolateToParent[spinIndex], 1,1,0,
                    true);

            unsigned int iElemPsi = 0;
            for (; cellPsi != endcPsi; ++cellPsi)
                if (cellPsi->is_locally_owned()) {


			for (unsigned int iQuad = 0; iQuad < numQuadPointsPerPsiCell;
           ++iQuad) {
				vxcOutputData[spinIndex][(
                            iElemPsi * numQuadPointsPerPsiCell + iQuad)] = d_vxcLDAQuadData[spinIndex][(
                            iElemPsi * numQuadPointsPerPsiCell + iQuad)] * d_inverseDFTParams.factorForLDAVxc +
    vxcInterpolateToParent[spinIndex].data()[iElemPsi * numQuadPointsPerPsiCell + iQuad];
			}
                    iElemPsi++;
                }

            dftfe::vectorTools::createDealiiVector<double>(
                    d_dftMatrixFreeData->get_vector_partitioner(
                            d_dftDensityDoFHandlerIndex),
                    1, vxcInitialGuessParentMesh[spinIndex]);
            vxcInitialGuessParentMesh[spinIndex] = 0.0;

            d_dftBaseClass->l2ProjectionQuadToNodal(
                    d_basisOperationsHost, *d_constraintDFTClass,
                    d_dftDensityDoFHandlerIndex, d_dftQuadIndex,
                    vxcOutputData[spinIndex], vxcInitialGuessParentMesh[spinIndex]);

            vxcInitialGuessParentMesh[spinIndex].update_ghost_values();
            d_constraintDFTClass->distribute(vxcInitialGuessParentMesh[spinIndex]);
            vxcInitialGuessParentMesh[spinIndex].update_ghost_values();
        }
    }

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTEngine<FEOrder, FEOrderElectro, memorySpace>::run() {
  dftfe::dftUtils::printCurrentMemoryUsage(d_mpiComm_domain,
                                           "Before parent cell manager");

  if (d_inverseDFTParams.netCharge != 0) {
    AssertThrow(d_dftParams.multipoleBoundaryConditions == true,
                ExcMessage("DFT-FE error: set MULTIPOLE BOUNDARY CONDITIONS in "
                           "DFT-FE to true "));
  }

  createParentChildDofManager();

  dftfe::dftUtils::printCurrentMemoryUsage(d_mpiComm_domain,
                                           "after parent cell manager");

  const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &rhoInValues = d_dftBaseClass->getDensityInValues();

  const std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &rhoInSpinPolarised = d_dftBaseClass->getDensityInValues();

  const dealii::Quadrature<3> &quadratureRuleParent =
      d_dftMatrixFreeData->get_quadrature(d_dftQuadIndex);
  const unsigned int numQuadraturePointsPerCellParent =
      quadratureRuleParent.size();
  unsigned int totalLocallyOwnedCellsParent =
      d_dftMatrixFreeData->n_physical_cells();

  const unsigned int numTotalQuadraturePointsParent =
      totalLocallyOwnedCellsParent * numQuadraturePointsPerCellParent;

  std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      rhoValuesFeSpin;
  rhoValuesFeSpin.resize(
      d_numSpins,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
          numTotalQuadraturePointsParent));
  if (d_numSpins == 1) {
    rhoValuesFeSpin[0].resize(numTotalQuadraturePointsParent, 0.0);
    typename dealii::DoFHandler<3>::active_cell_iterator
        cell = d_dofHandlerDFTClass->begin_active(),
        endc = d_dofHandlerDFTClass->end();
    unsigned int iElem = 0;
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned()) {
        for (unsigned int iQuad = 0; iQuad < numQuadraturePointsPerCellParent;
             iQuad++) {
          unsigned int index = iElem * numQuadraturePointsPerCellParent + iQuad;
          rhoValuesFeSpin[0][index] = 0.5 * rhoInValues[0][index];
        }
        iElem++;
      }
  } else {
    typename dealii::DoFHandler<3>::active_cell_iterator
        cell = d_dofHandlerDFTClass->begin_active(),
        endc = d_dofHandlerDFTClass->end();
    unsigned int iElem = 0;
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned()) {
        for (unsigned int iQuad = 0; iQuad < numQuadraturePointsPerCellParent;
             iQuad++) {
          unsigned int index = iElem * numQuadraturePointsPerCellParent + iQuad;
          rhoValuesFeSpin[0][index] = rhoInValues[0][index];
          rhoValuesFeSpin[1][index] = rhoInValues[1][index];
        }
        iElem++;
      }
  }

  if (d_inverseDFTParams.readGaussian) {
    setInitialDensityFromGaussian(rhoValuesFeSpin);
  } else {
    setTargetDensity(rhoInValues, rhoInSpinPolarised);
  }

  setInitialPotL2Proj();
  if (d_inverseDFTParams.readVxcData) {
    readVxcInput();
  } 
  

  unsigned int numElectronsWithCharge =
      d_dftBaseClass->getNumElectrons() + d_inverseDFTParams.netCharge;
  d_dftBaseClass->setNumElectrons(numElectronsWithCharge);

  setPotBase();

  InverseDFTSolverFunction<FEOrder, FEOrderElectro, memorySpace>
      inverseDFTSolverFunctionObj(d_mpiComm_parent, d_mpiComm_domain,
                                  d_mpiComm_bandgroup, d_mpiComm_interpool);

  dftfe::dftUtils::printCurrentMemoryUsage(d_mpiComm_domain,
                                           "Created inverse dft solver func");

  unsigned int spinFactor = (d_numSpins == 2) ? 1 : 2;

  std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      weightQuadData;
  weightQuadData.resize(
      d_numSpins,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
          totalLocallyOwnedCellsParent * numQuadraturePointsPerCellParent));

  double tauWeight = d_inverseDFTParams.inverseTauForSmoothening;

  for (unsigned int iSpin = 0; iSpin < d_numSpins; iSpin++) {
    unsigned int sizeOfQuadTotal = d_rhoTarget[iSpin].size();
    for (unsigned int iQuad = 0; iQuad < sizeOfQuadTotal; iQuad++) {
      // weightQuadData[iSpin][iQuad] = 1.0;
      weightQuadData[iSpin][iQuad] =
          1.0 / (std::pow(spinFactor * d_rhoTarget[iSpin][iQuad],
                          d_inverseDFTParams.inverseAlpha1ForWeights) +
                 tauWeight);
      weightQuadData[iSpin][iQuad] +=
          std::pow(spinFactor * d_rhoTarget[iSpin][iQuad],
                   d_inverseDFTParams.inverseAlpha2ForWeights);
    }
  }

  dftfe::KohnShamHamiltonianOperator<memorySpace> *kohnShamClassPtr =
      d_dftBaseClass->getOperatorClass();

  if (d_inverseDFTParams.solvermode == "FUNCTIONAL_TEST") {
    testAdjoint();
    return;
  } else if (d_inverseDFTParams.solvermode == "POST_PROCESS") {
    interpolateVxc();
    return;
  }

  inverseDFTSolverFunctionObj.reinit(
      d_rhoTarget, weightQuadData, d_potBaseQuadData, d_vxcLDAQuadData,
      d_quadCoordinatesParent, *d_dftBaseClass,
      *d_constraintDFTClass,     // assumes that the constraint matrix has
                                 // homogenous BC
      d_constraintMatrixAdjoint, // assumes that the constraint matrix has
                                 // homogenous BC
      d_constraintMatrixVxc, d_blasWrapperHost, d_blasWrapperMemSpace,
      d_basisOperationsAdjointMemSpacePtr, d_basisOperationsAdjointHostPtr,
      d_basisOperationsChildHostPtr, *kohnShamClassPtr,
      d_inverseDftDoFManagerObjPtr, d_kpointWeights, d_numSpins,
      d_numEigenValues, d_adjointMFPsiConstraints,
      //                                       d_adjointMFPsiConstraints,
      d_adjointMFAdjointConstraints, d_dofHandlerVxcIndex, d_quadAdjointIndex,
      d_quadVxcIndex,
      true, //         isComputeDiagonalA
      true, //        isComputeShapeFunction
      d_dftParams, d_inverseDFTParams);

  dftfe::dftUtils::printCurrentMemoryUsage(d_mpiComm_domain,
                                           "after solver func reinit");

  // computing energies
  {
    // compute KE
    pcout << " Kinetic energy at start\n";
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        kineticEnergyDensityValues;
    double kineticEnergy =
        d_dftBaseClass->computeAndPrintKE(kineticEnergyDensityValues);

    // compute electrostatic energy
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        totalDensityValue;
    totalDensityValue.resize(d_rhoTarget[0].size());

    unsigned int spinIndex1 = 0;
    unsigned int spinIndex2 = 1;
    if (d_numSpins == 1)
      spinIndex2 = 0;
    for (unsigned int i = 0; i < d_rhoTarget[0].size(); i++) {
      totalDensityValue[i] =
          d_rhoTarget[spinIndex1][i] + d_rhoTarget[spinIndex2][i];
    }
    pcout << " Electro static energy for rho target\n";
    double totalElectrostaticEnergy =
        inverseDFTSolverFunctionObj.computeElectrostaticEnergy(
            totalDensityValue);

    {
      pcout << " LDA-PW energy at rho target\n";
      xc_func_type funcXLDA, funcCLDA;
      int exceptParamX = xc_func_init(&funcXLDA, XC_LDA_X, XC_UNPOLARIZED);
      int exceptParamC = xc_func_init(&funcCLDA, XC_LDA_C_PW, XC_UNPOLARIZED);
      double xcLDAEnergy = inverseDFTSolverFunctionObj.computeLDAEnergy(
          totalDensityValue, "LDA-PW", funcXLDA, funcCLDA);
    }

    //        {
    //            xc_func_type funcXGGA, funcCGGA ;
    //            int exceptParamX = xc_func_init(funcXGGA, XC_GGA_X_PBE,
    //            XC_UNPOLARIZED); int exceptParamC = xc_func_init(funcCGGA,
    //            XC_GGA_C_PBE, XC_UNPOLARIZED); double xcGGAEnergy =
    //            inverseDFTSolverFunctionObj.computeGGAEnergy(rhoValues,
    //            gradRhoValues, "GGA-PBE", funcXGGA, funcCGGA );
    //        }
    //
    //        {
    //            xc_func_type funcXMGGA, funcCMGGA ;
    //            int exceptParamX = xc_func_init(funcXMGGA, MGGA_X_R2SCAN ,
    //            XC_UNPOLARIZED); int exceptParamC = xc_func_init(funcCMGGA,
    //            MGGA_C_R2SCAN , XC_UNPOLARIZED); double xcMGGAEnergy =
    //            computeMGGAEnergy(rhoValues, gradRhoValues,
    //            kineticEnergyDensityValues, "MGGA-R2SCAN", funcXMGGA,
    //            funcCMGGA );
    //        }
  }

  //    exit(0);

  inverseDFTSolverFunctionObj.setInitialGuess(d_vxcInitialChildNodes,
                                              d_targetPotValuesParentQuadData);

  pcout << " vxc initial guess norm before constructor = "
        << d_vxcInitialChildNodes[0].l2_norm() << "\n";
  BFGSInverseDFTSolver<FEOrder, FEOrderElectro, memorySpace>
      BFGSInverseDFTSolverObj(
          d_numSpins,                                  // numComponents
          d_inverseDFTParams.inverseBFGSTol,           // tol
          d_inverseDFTParams.inverseBFGSLineSearchTol, // lineSearchTol
          d_inverseDFTParams.inverseMaxBFGSIter,       // maxNumIter
          d_inverseDFTParams.inverseBFGSHistory,       // historySize
          d_inverseDFTParams.inverseBFGSLineSearch,    // numLineSearch
          d_mpiComm_parent);

  pcout << " vxc initial guess norm before solve = "
        << d_vxcInitialChildNodes[0].l2_norm() << "\n";
  BFGSInverseDFTSolverObj.solve(
      inverseDFTSolverFunctionObj,
      BFGSInverseDFTSolver<FEOrder, FEOrderElectro, memorySpace>::LSType::CP);
}

template <unsigned int FEOrder, unsigned int FEOrderElectro,
          dftfe::utils::MemorySpace memorySpace>
void InverseDFTEngine<FEOrder, FEOrderElectro, memorySpace>::testAdjoint() {
  unsigned int domainMPIRank =
      dealii::Utilities::MPI::this_mpi_process(d_mpiComm_domain);

  MultiVectorAdjointLinearSolverProblem<memorySpace> multiVectorAdjointProblem(
      d_mpiComm_parent, d_mpiComm_domain);
  dftfe::MultiVectorMinResSolver multiVectorLinearMINRESSolver(
      d_mpiComm_parent, d_mpiComm_domain);

  dftfe::KohnShamHamiltonianOperator<memorySpace> *kohnShamClassPtr =
      d_dftBaseClass->getOperatorClass();

  double TVal = d_dftParams.TVal;
  pcout << " Entering reinit\n";
  multiVectorAdjointProblem.reinit(
      d_blasWrapperMemSpace,
      d_basisOperationsAdjointMemSpacePtr[d_adjointMFPsiConstraints],
      *kohnShamClassPtr, *d_constraintDFTClass, TVal, d_adjointMFPsiConstraints,
      d_quadAdjointIndex, true);

  dftfe::dftUtils::constraintMatrixInfo<memorySpace>
      constraintsMatrixPsiDataInfo;
  const dealii::DoFHandler<3> *dofHandlerAdjoint =
      &d_matrixFreeDataAdjoint.get_dof_handler(d_adjointMFPsiConstraints);

  unsigned int locallyOwnedDofs = dofHandlerAdjoint->n_locally_owned_dofs();

  unsigned int numTotallyOwnedCells =
      d_matrixFreeDataAdjoint.n_physical_cells();

  const dealii::Quadrature<3> &quadratureRhs =
      d_matrixFreeDataAdjoint.get_quadrature(d_quadAdjointIndex);

  const unsigned int numberQuadraturePointsRhs = quadratureRhs.size();

  unsigned int defaultBlockSize = 100;
  dftfe::linearAlgebra::MultiVector<double, memorySpace> boundaryValues,
      multiVectorOutput, psiBlockVecMemSpace;

  unsigned int currentBlockSize = d_numEigenValues;
  const dftfe::utils::MemoryStorage<dftfe::dataTypes::number, memorySpace>
      &eigenVectorsMemSpace = d_dftBaseClass->getEigenVectors();

  const std::vector<std::vector<double>> &eigenValuesHost =
      d_dftBaseClass->getEigenValues();

  dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      d_matrixFreeDataAdjoint.get_vector_partitioner(d_adjointMFPsiConstraints),
      currentBlockSize, psiBlockVecMemSpace);

  d_blasWrapperMemSpace->stridedCopyToBlockConstantStride(
      currentBlockSize, d_numEigenValues, locallyOwnedDofs, 0,
      eigenVectorsMemSpace.begin() + (d_numSpins * 0 + 0) * d_numEigenValues,
      psiBlockVecMemSpace.begin());

  kohnShamClassPtr->reinitkPointSpinIndex(0, 0);

  dftfe::utils::MemoryStorage<double, memorySpace> differenceInDensities;
  differenceInDensities.resize(numberQuadraturePointsRhs * numTotallyOwnedCells,
                               0.0);

  double sumDiffDen = 0.0;
  for (unsigned int iQuad = 0;
       iQuad < numberQuadraturePointsRhs * numTotallyOwnedCells; iQuad++) {
    differenceInDensities.data()[iQuad] = 0.1 * d_rhoTarget[0][iQuad];

    sumDiffDen += differenceInDensities.data()[iQuad] *
                  differenceInDensities.data()[iQuad];
  }

  MPI_Allreduce(MPI_IN_PLACE, &sumDiffDen, 1,
                dftfe::dataTypes::mpi_type_id(&sumDiffDen), MPI_SUM,
                d_mpiComm_domain);

  pcout << " Norm of sumDiffDen = " << sumDiffDen << "\n";

  dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      d_matrixFreeDataAdjoint.get_vector_partitioner(d_adjointMFPsiConstraints),
      currentBlockSize, multiVectorOutput);

  boundaryValues.reinit(multiVectorOutput);

  constraintsMatrixPsiDataInfo.initialize(
      d_matrixFreeDataAdjoint.get_vector_partitioner(d_adjointMFPsiConstraints),
      *d_constraintDFTClass);

  boundaryValues.setValue(0.0);

  multiVectorOutput.setValue(0.0);

  psiBlockVecMemSpace.updateGhostValues();
  constraintsMatrixPsiDataInfo.distribute(psiBlockVecMemSpace);

  unsigned int numElectrons = d_dftBaseClass->getNumElectrons();

  std::vector<double> effectiveOrbitalOccupancy;
  std::vector<std::vector<unsigned int>> degeneracy;
  effectiveOrbitalOccupancy.resize(currentBlockSize);
  degeneracy.resize(currentBlockSize);
  std::vector<double> shiftValues;
  shiftValues.resize(currentBlockSize);

  for (unsigned int iWave = 0; iWave < d_numEigenValues; ++iWave) {
    if (iWave < numElectrons / 2) {
      effectiveOrbitalOccupancy[d_numEigenValues * 0 + iWave] = 1.0;
    } else {
      effectiveOrbitalOccupancy[d_numEigenValues * 0 + iWave] = 0.0;
    }
    shiftValues[iWave] = eigenValuesHost[0][d_numEigenValues * 0 + iWave];

    degeneracy[iWave].push_back(iWave);
  }

  const double fermiEnergy = d_dftBaseClass->getFermiEnergy();
  multiVectorAdjointProblem.updateInputPsi(
      psiBlockVecMemSpace, effectiveOrbitalOccupancy, differenceInDensities,
      degeneracy, fermiEnergy, shiftValues, currentBlockSize);

  multiVectorLinearMINRESSolver.solve(
      multiVectorAdjointProblem, d_blasWrapperMemSpace, multiVectorOutput,
      boundaryValues, locallyOwnedDofs, currentBlockSize, 1e-9, 5000, 4,
      true); // distributeFlag

  std::vector<double> l2NormVec(currentBlockSize, 0.0);

  // multiVectorOutput.l2Norm(&l2NormVec[0]);

  pcout << " multiVectorOutput = \n";
  for (unsigned int iB = 0; iB < currentBlockSize; iB++) {
    pcout << " iB = " << iB << " norm = " << l2NormVec[iB] << "\n";
  }

  exit(0);
}

template class InverseDFTEngine<2, 2, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTEngine<2, 3, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTEngine<2, 4, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTEngine<3, 3, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTEngine<3, 4, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTEngine<3, 5, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTEngine<3, 6, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTEngine<4, 4, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTEngine<4, 5, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTEngine<4, 6, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTEngine<4, 7, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTEngine<5, 5, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTEngine<5, 6, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTEngine<5, 7, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTEngine<5, 8, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTEngine<6, 6, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTEngine<6, 7, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTEngine<6, 8, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTEngine<6, 9, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTEngine<7, 7, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTEngine<7, 8, dftfe::utils::MemorySpace::HOST>;
template class InverseDFTEngine<7, 9, dftfe::utils::MemorySpace::HOST>;
#ifdef DFTFE_WITH_DEVICE
template class InverseDFTEngine<2, 2, dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTEngine<2, 3, dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTEngine<2, 4, dftfe::utils::MemorySpace::DEVICE>;

template class InverseDFTEngine<3, 3, dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTEngine<3, 4, dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTEngine<3, 5, dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTEngine<3, 6, dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTEngine<4, 4, dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTEngine<4, 5, dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTEngine<4, 6, dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTEngine<4, 7, dftfe::utils::MemorySpace::DEVICE>;

template class InverseDFTEngine<5, 5, dftfe::utils::MemorySpace::DEVICE>;

template class InverseDFTEngine<5, 6, dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTEngine<5, 7, dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTEngine<5, 8, dftfe::utils::MemorySpace::DEVICE>;

template class InverseDFTEngine<6, 6, dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTEngine<6, 7, dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTEngine<6, 8, dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTEngine<6, 9, dftfe::utils::MemorySpace::DEVICE>;

template class InverseDFTEngine<7, 7, dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTEngine<7, 8, dftfe::utils::MemorySpace::DEVICE>;
template class InverseDFTEngine<7, 9, dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace invDFT
