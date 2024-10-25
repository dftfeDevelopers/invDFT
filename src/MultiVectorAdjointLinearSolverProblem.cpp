//
// Created by VISHAL SUBRAMANIAN on 4/30/24.
//

#include "MultiVectorAdjointLinearSolverProblem.h"

#include <DeviceAPICalls.h>
#include <DeviceBlasWrapper.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceKernelLauncherConstants.h>
#include <deviceKernelsGeneric.h>

namespace invDFT {

namespace {

#ifdef DFTFE_WITH_DEVICE
template <typename ValueType1, typename ValueType2>
__global__ void rMatrixDeviceKernel(const dftfe::size_type numLocalCells,
                                    const dftfe::size_type numDofsPerElem,
                                    const dftfe::size_type numQuadPoints,
                                    const ValueType1 *shapeFunc,
                                    const ValueType1 *shapeFuncTranspose,
                                    const ValueType2 *inputJxW,
                                    ValueType2 *rMatrix) {
  const dftfe::size_type globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
  const dftfe::size_type numberEntries =
      numLocalCells * numDofsPerElem * numDofsPerElem;

  for (dftfe::size_type index = globalThreadId; index < numberEntries;
       index += blockDim.x * gridDim.x) {
    dftfe::size_type iElem = index / (numDofsPerElem * numDofsPerElem);
    dftfe::size_type nodeIndex = index % (numDofsPerElem * numDofsPerElem);
    dftfe::size_type iNode = nodeIndex / (numDofsPerElem);
    dftfe::size_type jNode = nodeIndex % (numDofsPerElem);

    dftfe::size_type elemRIndex = iElem * numDofsPerElem * numDofsPerElem;
    dftfe::size_type nodeRIndex = iNode * numDofsPerElem + jNode;

    dftfe::size_type iNodeQuadIndex = iNode * numQuadPoints;
    dftfe::size_type jNodeQuadIndex = jNode * numQuadPoints;

    dftfe::size_type elemQuadIndex = iElem * numQuadPoints;
    for (dftfe::size_type iQuad = 0; iQuad < numQuadPoints; iQuad++) {
      dftfe::utils::copyValue(
          rMatrix + elemRIndex + nodeRIndex,
          dftfe::utils::add(
              rMatrix[elemRIndex + nodeRIndex],
              dftfe::utils::mult(
                  shapeFuncTranspose[iNodeQuadIndex + iQuad],
                  dftfe::utils::mult(shapeFuncTranspose[jNodeQuadIndex + iQuad],
                                     inputJxW[elemQuadIndex + iQuad]))));
    }
  }
}

template <typename ValueType1, typename ValueType2>
void rMatrixMemSpaceKernel(
    const dftfe::size_type numLocalCells, const dftfe::size_type numDofsPerElem,
    const dftfe::size_type numQuadPoints,
    const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &BLASWrapperPtr,
    const dftfe::utils::MemoryStorage<
        ValueType1, dftfe::utils::MemorySpace::DEVICE> &shapeFunc,
    const dftfe::utils::MemoryStorage<
        ValueType2, dftfe::utils::MemorySpace::DEVICE> &shapeFuncTranspose,
    const dftfe::utils::MemoryStorage<
        ValueType1, dftfe::utils::MemorySpace::DEVICE> &inputJxW,
    dftfe::utils::MemoryStorage<ValueType1, dftfe::utils::MemorySpace::DEVICE>
        &rMatrix) {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
  rMatrixDeviceKernel<<<(numLocalCells * numDofsPerElem * numDofsPerElem) /
                                dftfe::utils::DEVICE_BLOCK_SIZE +
                            1,
                        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
      numLocalCells, numDofsPerElem, numQuadPoints,
      dftfe::utils::makeDataTypeDeviceCompatible(shapeFunc.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(shapeFuncTranspose.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(inputJxW.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(rMatrix.begin()));
#elif DFTFE_WITH_DEVICE_LANG_HIP
  hipLaunchKernelGGL(
      rMatrixDeviceKernel,
      (numLocalCells * numDofsPerElem * numDofsPerElem) /
              dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
      dftfe::utils::DEVICE_BLOCK_SIZE, 0, 0, numLocalCells, numDofsPerElem,
      numQuadPoints,
      dftfe::utils::makeDataTypeDeviceCompatible(shapeFunc.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(shapeFuncTranspose.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(inputJxW.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(rMatrix.begin()));
#endif
}

template <typename ValueType>
__global__ void muMatrixDeviceKernel(
    const dftfe::size_type numLocalCells, const dftfe::size_type numVec,
    const dftfe::size_type numQuadPoints, const dftfe::size_type blockSize,
    const ValueType *orbitalOccupancy, const unsigned int *vecList,
    const ValueType *cellLevelQuadValues, const ValueType *inputJxW,
    ValueType *muMatrixCellWise) {
  const dftfe::size_type globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
  const dftfe::size_type numberEntries = numLocalCells * numVec;

  for (dftfe::size_type index = globalThreadId; index < numberEntries;
       index += blockDim.x * gridDim.x) {
    dftfe::size_type iElem = index / (numVec);
    dftfe::size_type vecIndex = index % numVec;
    dftfe::size_type vecId = vecList[2 * vecIndex];
    dftfe::size_type degenerateId = vecList[2 * vecIndex + 1];

    dftfe::size_type elemQuadIndex = iElem * numQuadPoints * blockSize;
    dftfe::size_type elemInputQuadIndex = iElem * numQuadPoints;
    for (dftfe::size_type iQuad = 0; iQuad < numQuadPoints; iQuad++) {
      dftfe::utils::copyValue(
          muMatrixCellWise + iElem * numVec + vecIndex,
          dftfe::utils::add(
              muMatrixCellWise[iElem * numVec + vecIndex],
              dftfe::utils::mult(
                  dftfe::utils::mult(2.0, orbitalOccupancy[vecId]),
                  dftfe::utils::mult(
                      cellLevelQuadValues[elemQuadIndex + vecId +
                                          iQuad * blockSize],
                      dftfe::utils::mult(
                          cellLevelQuadValues[elemQuadIndex + degenerateId +
                                              iQuad * blockSize],
                          inputJxW[elemInputQuadIndex + iQuad])))));
    }
  }
}

template <typename ValueType>
void muMatrixMemSpaceKernel(
    const dftfe::size_type numLocalCells, const dftfe::size_type numVec,
    const dftfe::size_type numQuadPoints, const dftfe::size_type blockSize,
    const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &BLASWrapperPtr,
    const dftfe::utils::MemoryStorage<
        ValueType, dftfe::utils::MemorySpace::DEVICE> &orbitalOccupancy,
    const dftfe::utils::MemoryStorage<
        unsigned int, dftfe::utils::MemorySpace::DEVICE> &vecList,
    const dftfe::utils::MemoryStorage<
        ValueType, dftfe::utils::MemorySpace::DEVICE> &cellLevelQuadValues,
    const dftfe::utils::MemoryStorage<
        ValueType, dftfe::utils::MemorySpace::DEVICE> &inputJxW,
    dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
        &muMatrixCellWise) {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
  muMatrixDeviceKernel<<<
      (numLocalCells * numVec) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
      dftfe::utils::DEVICE_BLOCK_SIZE>>>(
      numLocalCells, numVec, numQuadPoints, blockSize,
      dftfe::utils::makeDataTypeDeviceCompatible(orbitalOccupancy.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(vecList.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(cellLevelQuadValues.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(inputJxW.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(muMatrixCellWise.begin()));
#elif DFTFE_WITH_DEVICE_LANG_HIP
  hipLaunchKernelGGL(
      muMatrixDeviceKernel,
      (numLocalCells * numVec) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
      dftfe::utils::DEVICE_BLOCK_SIZE, 0, 0, numLocalCells, numVec,
      numQuadPoints, blockSize,
      dftfe::utils::makeDataTypeDeviceCompatible(orbitalOccupancy.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(vecList.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(cellLevelQuadValues.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(inputJxW.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(muMatrixCellWise.begin()));
#endif
}
#endif

template <typename ValueType1, typename ValueType2>
void rMatrixMemSpaceKernel(
    const dftfe::size_type numLocalCells, const dftfe::size_type numDofsPerElem,
    const dftfe::size_type numQuadPoints,
    const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
        &BLASWrapperPtr,
    const dftfe::utils::MemoryStorage<
        ValueType1, dftfe::utils::MemorySpace::HOST> &shapeFunc,
    const dftfe::utils::MemoryStorage<
        ValueType1, dftfe::utils::MemorySpace::HOST> &shapeFuncTranspose,
    const dftfe::utils::MemoryStorage<
        ValueType2, dftfe::utils::MemorySpace::HOST> &inputJxW,
    dftfe::utils::MemoryStorage<ValueType2, dftfe::utils::MemorySpace::HOST>
        &rMatrix) {
  AssertThrow(inputJxW.size() == (numQuadPoints * numLocalCells),
              dealii::ExcMessage(
                  "In inputJxW the inputJxW should have only one component"
                  "u(r) = w(r)*(rho_target(r) - rho_KS(r))"));

  std::fill(rMatrix.begin(), rMatrix.end(), 0.0);
  const unsigned int inc = 1;

  std::vector<double> cellLevelJxW, cellLevelShapeFunction, cellLevelRhsInput;
  cellLevelJxW.resize(numQuadPoints);

  std::vector<double> shapeFuncIJ(numDofsPerElem * numQuadPoints, 0.0);
  std::vector<double> cellLevelR(numDofsPerElem * numLocalCells, 0.0);

  double beta = 0.0, alpha = 1.0;
  char transA = 'N', transB = 'N';
  for (unsigned int iNode = 0; iNode < numDofsPerElem; iNode++) {
    for (unsigned int iQuad = 0; iQuad < numQuadPoints; iQuad++) {
      for (unsigned int jNode = 0; jNode < numDofsPerElem; jNode++) {
        shapeFuncIJ[iQuad * numDofsPerElem + jNode] =
            shapeFuncTranspose[iNode * numQuadPoints + iQuad] *
            shapeFunc[jNode + iQuad * numDofsPerElem];
      }
    }

    if (numLocalCells == 0) {
      std::cout << " Error in numLocalCells is zero !!!!!!\n";
    }
    if (inputJxW.size() != numQuadPoints * numLocalCells) {
      std::cout << " inputJxW error in compute r mat\n";
    }
    BLASWrapperPtr->xgemm(transA, transB, numDofsPerElem, numLocalCells,
                          numQuadPoints, &alpha, &shapeFuncIJ[0],
                          numDofsPerElem, &inputJxW[0], numQuadPoints, &beta,
                          &cellLevelR[0], numDofsPerElem);
    for (unsigned int elemId = 0; elemId < numLocalCells; elemId++) {
      dftfe::dcopy_(&numDofsPerElem, &cellLevelR[elemId * numDofsPerElem], &inc,
                    &rMatrix[elemId * numDofsPerElem * numDofsPerElem +
                             iNode * numDofsPerElem],
                    &inc);
    }
  }
}

template <typename ValueType>
void muMatrixMemSpaceKernel(
    const dftfe::size_type numLocalCells, const dftfe::size_type numVec,
    const dftfe::size_type numQuadPoints, const dftfe::size_type blockSize,
    const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
        &BLASWrapperPtr,
    const dftfe::utils::MemoryStorage<
        ValueType, dftfe::utils::MemorySpace::HOST> &orbitalOccupancy,
    const dftfe::utils::MemoryStorage<unsigned int,
                                      dftfe::utils::MemorySpace::HOST> &vecList,
    const dftfe::utils::MemoryStorage<
        ValueType, dftfe::utils::MemorySpace::HOST> &cellLevelQuadValues,
    const dftfe::utils::MemoryStorage<
        ValueType, dftfe::utils::MemorySpace::HOST> &inputJxW,
    dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::HOST>
        &muMatrixCellWise) {
  for (unsigned int index = 0; index < numLocalCells * numVec; index++) {
    dftfe::size_type iElem = index / (numVec);
    dftfe::size_type vecIndex = index % numVec;
    dftfe::size_type vecId = vecList[2 * vecIndex];
    dftfe::size_type degenerateId = vecList[2 * vecIndex + 1];

    dftfe::size_type elemQuadIndex = iElem * numQuadPoints * blockSize;
    dftfe::size_type elemInputQuadIndex = iElem * numQuadPoints;
    for (dftfe::size_type iQuad = 0; iQuad < numQuadPoints; iQuad++) {
      muMatrixCellWise[iElem * numVec + vecIndex] +=
          2.0 * orbitalOccupancy[vecId] *
          cellLevelQuadValues[elemQuadIndex + vecId + iQuad * blockSize] *
          cellLevelQuadValues[elemQuadIndex + degenerateId +
                              iQuad * blockSize] *
          inputJxW[elemInputQuadIndex + iQuad];
    }
  }
}

template <typename ValueType>
__global__ void performHadamardProductKernel(const dftfe::size_type contiguousBlockSize,
                                    const dftfe::size_type nonConiguousBlockSize,
                                    const dftfe::size_type numDegenerateVec,
                                    const unsigned int *vectorList,
                                    const ValueType *vec1QuadValues,
                                    const ValueType *vec2QuadValues,
                                    ValueType *vecOutputQuadValues) {
  const dftfe::size_type globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
  const dftfe::size_type numberEntries = numDegenerateVec*nonConiguousBlockSize;

  for (dftfe::size_type index = globalThreadId; index < numberEntries;
       index += blockDim.x * gridDim.x) {
	  dftfe::size_type iNode = index/numDegenerateVec;
	  dftfe::size_type vecIndex = index - iNode*numDegenerateVec;
	  dftfe::size_type vec1Id = vectorList[2*vecIndex];
	  dftfe::size_type vec2Id = vectorList[2*vecIndex+1];

	  dftfe::utils::copyValue(vecOutputQuadValues + numDegenerateVec*iNode + vecIndex,
			  dftfe::utils::mult(vec1QuadValues[iNode*contiguousBlockSize + vec1Id],
				  vec2QuadValues[iNode*contiguousBlockSize + vec2Id]));
    }
  }


template <typename ValueType>
void performHadamardProduct(const unsigned int contiguousBlockSize,
                            const unsigned int nonConiguousBlockSize,
                            const unsigned int numDegenerateVec,
                            const dftfe::utils::MemoryStorage<unsigned int,dftfe::utils::MemorySpace::DEVICE> & vectorList,
                       const dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::DEVICE>& vec1QuadValues,
                       const dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::DEVICE>& vec2QuadValues,
                       dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::DEVICE>& vecOutputQuadValues)
         {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
  performHadamardProductKernel<<<(numDegenerateVec * nonConiguousBlockSize) /
                                dftfe::utils::DEVICE_BLOCK_SIZE +
                            1,
                        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
      contiguousBlockSize, nonConiguousBlockSize, numDegenerateVec,
      dftfe::utils::makeDataTypeDeviceCompatible(vectorList.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(vec1QuadValues.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(vec2QuadValues.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(vecOutputQuadValues.begin()));
#elif DFTFE_WITH_DEVICE_LANG_HIP
  hipLaunchKernelGGL(
      performHadamardProductKernel,
      (numDegenerateVec * nonConiguousBlockSize) /
              dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
      dftfe::utils::DEVICE_BLOCK_SIZE, 0, 0, contiguousBlockSize, nonConiguousBlockSize,
      numDegenerateVec,
      dftfe::utils::makeDataTypeDeviceCompatible(vectorList.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(vec1QuadValues.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(vec2QuadValues.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(vecOutputQuadValues.begin()));
#endif
}

template <typename ValueType>
void performHadamardProduct(const unsigned int contiguousBlockSize,
		            const unsigned int nonConiguousBlockSize,
			    const unsigned int numDegenerateVec,
			    const dftfe::utils::MemoryStorage<unsigned int,dftfe::utils::MemorySpace::HOST> & vectorList,
                       const dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::HOST>& vec1QuadValues,
                       const dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::HOST>& vec2QuadValues,
                       dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::HOST>& vecOutputQuadValues)
{

	for( unsigned int iNode = 0; iNode < nonConiguousBlockSize ; iNode++)
	{
		for (unsigned int iVec = 0 ; iVec < numDegenerateVec; iVec++)
		{
			unsigned int vec1Id = vectorList.data()[2*iVec];
			unsigned int vec2Id = vectorList.data()[2*iVec + 1];
			vecOutputQuadValues.data()[iNode*numDegenerateVec + iVec] = vec1QuadValues.data()[iNode*contiguousBlockSize + vec1Id]*vec2QuadValues.data()[iNode*contiguousBlockSize + vec2Id];
		}
	}

}

template <typename ValueType>
__global__ void removeNullSpaceAtomicAddKernel(const dftfe::size_type contiguousBlockSize,
                                    const dftfe::size_type nonConiguousBlockSize,
                                    const dftfe::size_type numDegenerateVec,
                                    const unsigned int *vectorList,
                                    const ValueType *nullVectors,
                                    const ValueType *dotProduct,
                                    ValueType *outputVec) {
  const dftfe::size_type globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
  const dftfe::size_type numberEntries = numDegenerateVec*nonConiguousBlockSize;

  for (dftfe::size_type index = globalThreadId; index < numberEntries;
       index += blockDim.x * gridDim.x) {
          dftfe::size_type iNode = index/numDegenerateVec;
          dftfe::size_type vecIndex = index - iNode*numDegenerateVec;
          dftfe::size_type vec1Id = vectorList[2*vecIndex];
          dftfe::size_type vec2Id = vectorList[2*vecIndex+1];

	  atomicAdd(outputVec + vec1Id + iNode*contiguousBlockSize,
		dftfe::utils::mult(nullVectors[iNode*contiguousBlockSize + vec2Id], dotProduct[vecIndex]));
    }
  }

template <typename ValueType>
  void removeNullSpace(const unsigned int contiguousBlockSize,
                  const unsigned int nonConiguousBlockSize,
                            const unsigned int numDegenerateVec,
                            const dftfe::utils::MemoryStorage<unsigned int,dftfe::utils::MemorySpace::DEVICE>& vectorList,
                            const dftfe::linearAlgebra::MultiVector<ValueType,dftfe::utils::MemorySpace::DEVICE>& nullVectors,
                           const dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::DEVICE>& dotProduct,
                          dftfe::linearAlgebra::MultiVector<ValueType,dftfe::utils::MemorySpace::DEVICE>&  outputVec)
{
	#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
  removeNullSpaceAtomicAddKernel<<<(numDegenerateVec * nonConiguousBlockSize) /
                                dftfe::utils::DEVICE_BLOCK_SIZE +
                            1,
                        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
      contiguousBlockSize, nonConiguousBlockSize, numDegenerateVec,
      dftfe::utils::makeDataTypeDeviceCompatible(vectorList.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(nullVectors.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(dotProduct.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(outputVec.begin()));
#elif DFTFE_WITH_DEVICE_LANG_HIP
  hipLaunchKernelGGL(
      removeNullSpaceAtomicAddKernel,
      (numDegenerateVec * nonConiguousBlockSize) /
              dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
      dftfe::utils::DEVICE_BLOCK_SIZE, 0, 0, contiguousBlockSize, nonConiguousBlockSize,
      numDegenerateVec,
      dftfe::utils::makeDataTypeDeviceCompatible(vectorList.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(nullVectors.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(dotProduct.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(outputVec.begin()));
#endif

}



template <typename ValueType>
  void removeNullSpace(const unsigned int contiguousBlockSize,
		  const unsigned int nonConiguousBlockSize,
                            const unsigned int numDegenerateVec,
			    const dftfe::utils::MemoryStorage<unsigned int,dftfe::utils::MemorySpace::HOST> & vectorList,
			    const dftfe::linearAlgebra::MultiVector<ValueType,dftfe::utils::MemorySpace::HOST>& nullVectors,
			   const dftfe::utils::MemoryStorage<ValueType,dftfe::utils::MemorySpace::HOST>& dotProduct,
			  dftfe::linearAlgebra::MultiVector<ValueType,dftfe::utils::MemorySpace::HOST>&  outputVec)
{
	for( unsigned int iNode = 0; iNode < nonConiguousBlockSize ; iNode++)
        {
                for (unsigned int iVec = 0 ; iVec < numDegenerateVec; iVec++)
                {
                        unsigned int vec1Id = vectorList.data()[2*iVec];
                        unsigned int vec2Id = vectorList.data()[2*iVec + 1];
                        outputVec.data()[iNode*contiguousBlockSize+ vec1Id] +=  dotProduct.data()[iVec]*nullVectors.data()[iNode*contiguousBlockSize + vec2Id];
                }
        }


}


} // namespace

// constructor
template <dftfe::utils::MemorySpace memorySpace>
MultiVectorAdjointLinearSolverProblem<memorySpace>::
    MultiVectorAdjointLinearSolverProblem(const MPI_Comm &mpi_comm_parent,
                                          const MPI_Comm &mpi_comm_domain)
    : mpi_communicator(mpi_comm_domain),
      n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_domain)),
      this_mpi_process(
          dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain)),
      pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0)) {
  d_isComputeDiagonalA = true;
  d_constraintMatrixPtr = NULL;
  d_blockedXPtr = NULL;
  d_matrixFreeQuadratureComponentRhs = -1;
  d_matrixFreeVectorComponent = -1;
  d_blockSize = 0;
  d_cellBlockSize = 100; // TODO set this based on rum time.
}

template <dftfe::utils::MemorySpace memorySpace>
void MultiVectorAdjointLinearSolverProblem<memorySpace>::reinit(
    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
        BLASWrapperPtr,
    std::shared_ptr<dftfe::basis::FEBasisOperations<dftfe::dataTypes::number,
                                                    double, memorySpace>>
        basisOperationsPtr,
    dftfe::KohnShamHamiltonianOperator<memorySpace> &ksHamiltonianObj,
    const dealii::AffineConstraints<double> &constraintMatrix,
    const double TVal,
    const unsigned int matrixFreeVectorComponent,
    const unsigned int matrixFreeQuadratureComponentRhs,
    const bool isComputeDiagonalA) {
  int this_process;
  MPI_Comm_rank(mpi_communicator, &this_process);
  MPI_Barrier(mpi_communicator);

  d_BLASWrapperPtr = BLASWrapperPtr;
  d_basisOperationsPtr = basisOperationsPtr;
  d_matrixFreeDataPtr = &(basisOperationsPtr->matrixFreeData());
  d_constraintMatrixPtr = &constraintMatrix;
  d_matrixFreeVectorComponent = matrixFreeVectorComponent;
  d_matrixFreeQuadratureComponentRhs = matrixFreeQuadratureComponentRhs;

  d_TVal = TVal;
  d_numCells = d_basisOperationsPtr->nCells();

  d_cellBlockSize = std::min(d_cellBlockSize, d_numCells);

  d_basisOperationsPtr->reinit(1, d_cellBlockSize,
                               d_matrixFreeQuadratureComponentRhs,
                               true,  // TODO should this be set to true
                               true); // TODO should this be set to true

  d_locallyOwnedSize = d_basisOperationsPtr->nOwnedDofs();
  d_numberDofsPerElement = d_basisOperationsPtr->nDofsPerCell();

  d_numQuadsPerCell = d_basisOperationsPtr->nQuadsPerCell();

  d_ksOperatorPtr = &ksHamiltonianObj;

  // std::cout<<" local size in adjoint = "<<d_locallyOwnedSize<<"\n";

  if (isComputeDiagonalA) {
    computeDiagonalA();
    d_isComputeDiagonalA = true;
  }

  d_constraintsInfo.initialize(
      d_matrixFreeDataPtr->get_vector_partitioner(matrixFreeVectorComponent),
      constraintMatrix);

  d_onesMemSpace.resize(d_locallyOwnedSize);
  d_onesMemSpace.setValue(1.0);

  d_onesQuadMemSpace.resize(d_numCells * d_numQuadsPerCell);
  d_onesQuadMemSpace.setValue(1.0);

  d_basisOperationsPtr->computeCellStiffnessMatrix(
      d_matrixFreeQuadratureComponentRhs, 1, true, false);
  d_basisOperationsPtr->computeCellMassMatrix(
      d_matrixFreeQuadratureComponentRhs, 1, true, false);

  d_basisOperationsPtr->computeInverseSqrtMassVector(true, false);
}

template <dftfe::utils::MemorySpace memorySpace>
void MultiVectorAdjointLinearSolverProblem<memorySpace>::computeDiagonalA() {
  d_basisOperationsPtr->computeStiffnessVector(true, true);
  d_basisOperationsPtr->computeInverseSqrtMassVector();

  dftfe::utils::MemoryStorage<dftfe::global_size_type,
                              dftfe::utils::MemorySpace::HOST>
      nodeIds;
  nodeIds.resize(d_locallyOwnedSize);
  for (dftfe::size_type i = 0; i < d_locallyOwnedSize; i++) {
    nodeIds.data()[i] = i;
  }

  dftfe::utils::MemoryStorage<dftfe::global_size_type, memorySpace>
      mapNodeIdToProcId;
  mapNodeIdToProcId.resize(d_locallyOwnedSize);
  mapNodeIdToProcId.copyFrom(nodeIds);

  auto sqrtMassMat = d_basisOperationsPtr->sqrtMassVectorBasisData();
  auto inverseStiffVec =
      d_basisOperationsPtr->inverseStiffnessVectorBasisData();
  auto inverseSqrtStiffVec =
      d_basisOperationsPtr->inverseSqrtStiffnessVectorBasisData();

  d_basisOperationsPtr->createMultiVector(1, d_diagonalA);
  d_diagonalA.setValue(1.0);
  d_BLASWrapperPtr->stridedBlockScaleCopy(
      1, d_locallyOwnedSize, 1.0 / 0.5, inverseStiffVec.data(),
      d_diagonalA.data(), d_diagonalA.data(), mapNodeIdToProcId.data());

  d_BLASWrapperPtr->stridedBlockScaleCopy(
      1, d_locallyOwnedSize, 1.0, sqrtMassMat.data(), d_diagonalA.data(),
      d_diagonalA.data(), mapNodeIdToProcId.data());

  d_BLASWrapperPtr->stridedBlockScaleCopy(
      1, d_locallyOwnedSize, 1.0, sqrtMassMat.data(), d_diagonalA.data(),
      d_diagonalA.data(), mapNodeIdToProcId.data());

  d_basisOperationsPtr->createMultiVector(1, d_diagonalSqrtA);
  d_diagonalSqrtA.setValue(1.0);
  d_BLASWrapperPtr->stridedBlockScaleCopy(
      1, d_locallyOwnedSize, std::sqrt(1.0 / 0.5), inverseSqrtStiffVec.data(),
      d_diagonalSqrtA.data(), d_diagonalSqrtA.data(), mapNodeIdToProcId.data());

  d_BLASWrapperPtr->stridedBlockScaleCopy(
      1, d_locallyOwnedSize, 1.0, sqrtMassMat.data(), d_diagonalSqrtA.data(),
      d_diagonalSqrtA.data(), mapNodeIdToProcId.data());
}

template <dftfe::utils::MemorySpace memorySpace>
void MultiVectorAdjointLinearSolverProblem<memorySpace>::precondition_Jacobi(
    dftfe::linearAlgebra::MultiVector<dftfe::dataTypes::number, memorySpace>
        &dst,
    const dftfe::linearAlgebra::MultiVector<dftfe::dataTypes::number,
                                            memorySpace> &src,
    const double omega) const {

  d_BLASWrapperPtr->stridedBlockScaleCopy(
      d_blockSize, d_locallyOwnedSize, 1.0, d_diagonalA.data(), src.data(),
      dst.data(), d_mapNodeIdToProcId.data());
}

template <dftfe::utils::MemorySpace memorySpace>
void MultiVectorAdjointLinearSolverProblem<memorySpace>::
    precondition_JacobiSqrt(
        dftfe::linearAlgebra::MultiVector<dftfe::dataTypes::number, memorySpace>
            &dst,
        const dftfe::linearAlgebra::MultiVector<dftfe::dataTypes::number,
                                                memorySpace> &src,
        const double omega) const {

  d_BLASWrapperPtr->stridedBlockScaleCopy(
      d_blockSize, d_locallyOwnedSize, 1.0, d_diagonalSqrtA.data(), src.data(),
      dst.data(), d_mapNodeIdToProcId.data());
}

template <dftfe::utils::MemorySpace memorySpace>
dftfe::linearAlgebra::MultiVector<dftfe::dataTypes::number, memorySpace> &
MultiVectorAdjointLinearSolverProblem<memorySpace>::computeRhs(
    dftfe::linearAlgebra::MultiVector<dftfe::dataTypes::number, memorySpace>
        &NDBCVec,
    dftfe::linearAlgebra::MultiVector<dftfe::dataTypes::number, memorySpace>
        &outputVec,
    unsigned int blockSizeInput) {
  //    dealii::TimerOutput computing_timer(mpi_communicator,
  //                                        pcout,
  //                                        dealii::TimerOutput::summary,
  //                                        dealii::TimerOutput::wall_times);

  d_basisOperationsPtr->reinit(blockSizeInput, d_cellBlockSize,
                               d_matrixFreeQuadratureComponentRhs,
                               true,  // TODO should this be set to true
                               true); // TODO should this be set to true

  if (d_blockSize != blockSizeInput) {
    d_blockSize = blockSizeInput;
    dftfe::utils::MemoryStorage<dftfe::global_size_type,
                                dftfe::utils::MemorySpace::HOST>
        nodeIds, quadIds;
    nodeIds.resize(d_locallyOwnedSize);
    for (dftfe::size_type i = 0; i < d_locallyOwnedSize; i++) {
      nodeIds.data()[i] = i * d_blockSize;
    }
    d_mapNodeIdToProcId.resize(d_locallyOwnedSize);
    d_mapNodeIdToProcId.copyFrom(nodeIds);

    quadIds.resize(d_numCells * d_numQuadsPerCell);
    for (dftfe::size_type i = 0; i < d_numCells * d_numQuadsPerCell; i++) {
      quadIds.data()[i] = i * d_blockSize;
    }
    d_mapQuadIdToProcId.resize(d_numCells * d_numQuadsPerCell);
    d_mapQuadIdToProcId.copyFrom(quadIds);

    d_basisOperationsPtr->createMultiVector(d_blockSize, d_rhsMemSpace);

    tempOutputDotProdMemSpace.resize(d_blockSize);
    oneBlockSizeMemSpace.resize(d_blockSize);
    oneBlockSizeMemSpace.setValue(1.0);

          vec1QuadValues.resize(d_cellBlockSize * d_blockSize*d_numQuadsPerCell);
          vec2QuadValues.resize(d_cellBlockSize * d_blockSize*d_numQuadsPerCell);

	  unsigned int numDegenerateVec = d_vectorListMemSpace.size()/2;
          vecOutputQuadValues.resize(numDegenerateVec * d_cellBlockSize *d_numQuadsPerCell);
  

    //vec1QuadValues.resize(d_blockSize * d_numCells * d_numQuadsPerCell);
    //vec2QuadValues.resize(d_blockSize * d_numCells * d_numQuadsPerCell);
    //vecOutputQuadValues.resize(d_blockSize * d_numCells * d_numQuadsPerCell);
  
  }
  d_blockedXPtr = &outputVec;

  // psiTemp = M^{1/2} psi

  //    computing_timer.enter_subsection("Rhs init MemSpace MPI");
  dftfe::linearAlgebra::MultiVector<double, memorySpace> psiTempMemSpace;
  psiTempMemSpace.reinit(*d_psiMemSpace);

  d_BLASWrapperPtr->axpby(d_locallyOwnedSize * d_blockSize, 1.0,
                          d_psiMemSpace->begin(), 0.0, psiTempMemSpace.begin());
  psiTempMemSpace.updateGhostValues();
  d_constraintsInfo.distribute(psiTempMemSpace);

  dftfe::linearAlgebra::MultiVector<double, memorySpace> psiTempMemSpace2;
  psiTempMemSpace2.reinit(*d_psiMemSpace);

  psiTempMemSpace2.setValue(0.0);

  //    computing_timer.leave_subsection("Rhs init MemSpace MPI");

  //    computing_timer.enter_subsection("M^(-1/2) MemSpace MPI");

  auto sqrtMassMat = d_basisOperationsPtr->sqrtMassVectorBasisData();

  d_BLASWrapperPtr->stridedBlockScaleCopy(
      d_blockSize, d_locallyOwnedSize, 1.0, sqrtMassMat.data(),
      psiTempMemSpace.data(), psiTempMemSpace.data(),
      d_mapNodeIdToProcId.data());

  psiTempMemSpace.updateGhostValues();
  //    computing_timer.leave_subsection("M^(-1/2) MemSpace MPI");

  //    computing_timer.enter_subsection("computeR MemSpace MPI");
  computeRMatrix(d_inputJxWMemSpace);
  //    computing_timer.leave_subsection("computeR MemSpace MPI");
  //    computing_timer.enter_subsection("computeMu MemSpace MPI");
  computeMuMatrix(d_inputJxWMemSpace, d_effectiveOrbitalOccupancyMemSpace, *d_psiMemSpace);

  //    computing_timer.leave_subsection("computeMu MemSpace MPI");

  //    computing_timer.enter_subsection("Mu*Psi MemSpace MPI");

  d_rhsMemSpace.setValue(0.0);
  // Calculating the rhs from the quad points
  // multiVectorInput is stored on the quad points

  const unsigned int inc = 1;
  const double beta = 0.0, alpha = 1.0, alpha_minus_two = -2.0,
               alpha_minus_one = -1.0;

  // rhs = Psi*Mu. Since blas/lapack assume a column-major format whereas the
  // Psi is stored in a row major format, we do Mu^T*\Psi^T = Mu*\Psi^T
  // (because Mu is symmetric)

  d_BLASWrapperPtr->xgemm(
      'N', 'N', d_blockSize, d_locallyOwnedSize, d_blockSize, &alpha_minus_two,
      d_MuMatrixMemSpace.data(), d_blockSize, psiTempMemSpace.data(),
      d_blockSize, &beta, d_rhsMemSpace.data(), d_blockSize);

  //    computing_timer.leave_subsection("Mu*Psi MemSpace MPI");

  //
  // y = M^{-1/2} * R * M^{-1/2} * PsiTemp
  // 1. Do PsiTemp = M^{-1/2}*PsiTemp
  // 2. Do PsiTemp2 = R*PsiTemp
  // 3. PsiTemp2 = M^{-1/2}*PsiTemp2
  //

  //    computing_timer.enter_subsection("psi*M(-1/2) MemSpace MPI");

  auto invSqrtMassMat = d_basisOperationsPtr->inverseSqrtMassVectorBasisData();
  d_BLASWrapperPtr->stridedBlockScaleCopy(
      d_blockSize, d_locallyOwnedSize, 1.0, invSqrtMassMat.data(),
      psiTempMemSpace.data(), psiTempMemSpace.data(),
      d_mapNodeIdToProcId.data());

  psiTempMemSpace.updateGhostValues();
  d_constraintsInfo.distribute(psiTempMemSpace);
  //    computing_timer.leave_subsection("psi*M(-1/2) MemSpace MPI");

  //    computing_timer.enter_subsection("R times psi MemSpace MPI");

  // 2. Do PsiTemp2 = R*PsiTemp
  d_cellWaveFunctionMatrixMemSpace.setValue(0.0);
  std::pair<unsigned int, unsigned int> cellRange =
      std::make_pair(0, d_numCells);

  d_basisOperationsPtr->reinit(d_blockSize, d_cellBlockSize,
                               d_matrixFreeQuadratureComponentRhs,
                               true,   // TODO should this be set to true
                               false); // TODO should this be set to true
  d_basisOperationsPtr->extractToCellNodalDataKernel(
      psiTempMemSpace, d_cellWaveFunctionMatrixMemSpace.data(), cellRange);

  const dftfe::dataTypes::number scalarCoeffAlpha =
                                     dftfe::dataTypes::number(1.0),
                                 scalarCoeffBeta =
                                     dftfe::dataTypes::number(0.0);
  const unsigned int strideA = d_numberDofsPerElement * d_blockSize;
  const unsigned int strideB = d_numberDofsPerElement * d_numberDofsPerElement;
  const unsigned int strideC = d_numberDofsPerElement * d_blockSize;

  d_BLASWrapperPtr->xgemmStridedBatched(
      'N', 'N', d_blockSize, d_numberDofsPerElement, d_numberDofsPerElement,
      &scalarCoeffAlpha, d_cellWaveFunctionMatrixMemSpace.begin(), d_blockSize,
      strideA, d_RMatrixMemSpace.begin(), d_numberDofsPerElement, strideB,
      &scalarCoeffBeta, d_cellRMatrixTimesWaveMatrixMemSpace.begin(),
      d_blockSize, strideC, d_numCells);

  d_basisOperationsPtr->reinit(d_blockSize, d_cellBlockSize,
                               d_matrixFreeQuadratureComponentRhs,
                               true,   // TODO should this be set to true
                               false); // TODO should this be set to true

  d_basisOperationsPtr->accumulateFromCellNodalData(
      d_cellRMatrixTimesWaveMatrixMemSpace.begin(), psiTempMemSpace2);
  d_constraintsInfo.distribute_slave_to_master(psiTempMemSpace2);
  psiTempMemSpace2.accumulateAddLocallyOwned();

  // 3. PsiTemp2 = M^{-1/2}*PsiTemp2
  //    computing_timer.leave_subsection("R times psi MemSpace MPI");

  //    computing_timer.enter_subsection("psiTemp M^(-1/2) MemSpace MPI");

  psiTempMemSpace2.updateGhostValues();
  d_constraintsInfo.distribute(psiTempMemSpace2);

  d_BLASWrapperPtr->stridedBlockScaleCopy(
      d_blockSize, d_locallyOwnedSize, 1.0, invSqrtMassMat.data(),
      psiTempMemSpace2.data(), psiTempMemSpace2.data(),
      d_mapNodeIdToProcId.data());

  psiTempMemSpace2.updateGhostValues();
  d_constraintsInfo.distribute(psiTempMemSpace2);

  d_BLASWrapperPtr->stridedBlockScaleAndAddColumnWise(
      d_blockSize, d_locallyOwnedSize, psiTempMemSpace2.data(),
      d_4xeffectiveOrbitalOccupancyMemSpace.data(), d_rhsMemSpace.data());

  d_constraintsInfo.set_zero(d_rhsMemSpace);

  //    computing_timer.leave_subsection("psiTemp M^(-1/2) MemSpace MPI");

  return d_rhsMemSpace;
}

// TODO PLease call d_kohnShamClassPtr->reinitkPointSpinIndex() before
// calling this functions.

template <dftfe::utils::MemorySpace memorySpace>
void MultiVectorAdjointLinearSolverProblem<memorySpace>::updateInputPsi(
    dftfe::linearAlgebra::MultiVector<dftfe::dataTypes::number,
                                      memorySpace>
        &psiInputVecMemSpace, // need to call distribute
    std::vector<double>
        &effectiveOrbitalOccupancy, // incorporates spin information
    dftfe::utils::MemoryStorage<double, memorySpace> &differenceInDensity,
    std::vector<std::vector<unsigned int>> &degeneracy,
    double fermiEnergy, std::vector<double> &eigenValues, unsigned int blockSize) {
  pcout << " updating psi inside adjoint\n";

  d_fermiEnergy = fermiEnergy;
  d_psiMemSpace = &psiInputVecMemSpace;
  d_psiMemSpace->updateGhostValues();
  d_constraintsInfo.distribute(*d_psiMemSpace);

  d_RMatrixMemSpace.resize(d_numCells * d_numberDofsPerElement *
                           d_numberDofsPerElement);
  d_RMatrixMemSpace.setValue(0.0);

  d_MuMatrixMemSpace.resize(blockSize * blockSize);
  d_MuMatrixMemSpace.setValue(0.0);

  std::vector<double> effectiveOrbitalOccupancyHost;
  effectiveOrbitalOccupancyHost = effectiveOrbitalOccupancy;
  d_effectiveOrbitalOccupancyMemSpace.resize(blockSize);
  d_effectiveOrbitalOccupancyMemSpace.copyFrom(effectiveOrbitalOccupancyHost);

  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      effectiveOrbitalOccupancyHost_4x;
  effectiveOrbitalOccupancyHost_4x.resize(blockSize);

  for (unsigned int i = 0; i < blockSize; i++) {
    effectiveOrbitalOccupancyHost_4x[i] = 4.0 * effectiveOrbitalOccupancy[i];
  
    pcout<<" orb occ ["<<i<<"] = "<<effectiveOrbitalOccupancy[i]<<"\n";

  }

  d_4xeffectiveOrbitalOccupancyMemSpace.resize(blockSize);
  d_4xeffectiveOrbitalOccupancyMemSpace.copyFrom(
      effectiveOrbitalOccupancyHost_4x);

  d_degenerateState = degeneracy;
  d_eigenValues = eigenValues;

  d_vectorList.resize(0);
  for (unsigned int iVec = 0; iVec < blockSize; iVec++) {
    unsigned int totalNumDegenerateStates = d_degenerateState[iVec].size();
    for (unsigned int jVec = 0; jVec < totalNumDegenerateStates; jVec++) {
      d_vectorList.push_back(iVec);
      d_vectorList.push_back(d_degenerateState[iVec][jVec]);
    }
  }

  d_MuMatrixMemSpaceCellWise.resize((d_vectorList.size() / 2) * d_numCells,
                                    0.0);
  d_MuMatrixHostCellWise.resize((d_vectorList.size() / 2) * d_numCells, 0.0);

  d_MuMatrixHost.resize(blockSize * blockSize);
  std::fill(d_MuMatrixHost.begin(), d_MuMatrixHost.end(), 0.0);
  d_vectorListMemSpace.resize(d_vectorList.size());
  d_vectorListMemSpace.copyFrom(d_vectorList);
  if (blockSize != d_blockSize) {
    // If the number of vectors in the size is different, then the Map has
    // to be re-initialised. The d_blockSize is set to -1 in the
    // constructor, so that this if condition is satisfied the first time
    // the code is called.

    d_cellWaveFunctionMatrixMemSpace.resize(
        d_numCells * d_numberDofsPerElement * blockSize, 0.0);

    d_cellRMatrixTimesWaveMatrixMemSpace.resize(
        d_numCells * d_numberDofsPerElement * blockSize, 0.0);
    d_cellWaveFunctionQuadMatrixMemSpace.resize(d_numCells * d_numQuadsPerCell *
                                                blockSize);
    d_cellWaveFunctionQuadMatrixMemSpace.setValue(0.0);
  }

  d_negEigenValuesMemSpace.resize(blockSize);

  for (signed int iBlock = 0; iBlock < blockSize; iBlock++) {
    eigenValues[iBlock] = -1.0 * eigenValues[iBlock];
  }
  d_negEigenValuesMemSpace.copyFrom(eigenValues);

  auto cellJxW = d_basisOperationsPtr->JxW();
  d_inputJxWMemSpace.resize(d_numQuadsPerCell * d_numCells);
  d_BLASWrapperPtr->hadamardProduct(d_numCells * d_numQuadsPerCell,
                                    differenceInDensity.data(), cellJxW.data(),
                                    d_inputJxWMemSpace.data());
}

template <dftfe::utils::MemorySpace memorySpace>
void MultiVectorAdjointLinearSolverProblem<memorySpace>::computeMuMatrix(
    dftfe::utils::MemoryStorage<double, memorySpace> &inputJxwMemSpace,
    dftfe::utils::MemoryStorage<double, memorySpace> &effectiveOrbitalOcc,
    dftfe::linearAlgebra::MultiVector<dftfe::dataTypes::number, memorySpace>
        &psiVecMemSpace) {

  const unsigned int inc = 1;
  const double beta = 0.0, alpha = 1.0;
  char transposeMat = 'T';
  char doNotTransposeMat = 'N';

  d_MuMatrixMemSpace.setValue(0.0);
  d_MuMatrixHost.setValue(0.0);

  d_cellWaveFunctionMatrixMemSpace.setValue(0.0);

  d_basisOperationsPtr->reinit(d_blockSize, d_numCells,
                               d_matrixFreeQuadratureComponentRhs,
                               true,   // TODO should this be set to true
                               false); // TODO should this be set to true
                                       //

  std::pair<unsigned int, unsigned int> cellRange =
      std::make_pair(0, d_numCells);
  d_basisOperationsPtr->extractToCellNodalDataKernel(
      psiVecMemSpace, d_cellWaveFunctionMatrixMemSpace.data(), cellRange);

  const dftfe::dataTypes::number scalarCoeffAlpha =
                                     dftfe::dataTypes::number(1.0),
                                 scalarCoeffBeta =
                                     dftfe::dataTypes::number(0.0);
  const unsigned int strideA = d_numberDofsPerElement * d_blockSize;
  const unsigned int strideB = 0;
  const unsigned int strideC = d_numQuadsPerCell * d_blockSize;

  auto shapeFunctionData = d_basisOperationsPtr->shapeFunctionData(false);
  d_BLASWrapperPtr->xgemmStridedBatched(
      'N', 'N', d_blockSize, d_numQuadsPerCell, d_numberDofsPerElement, &alpha,
      d_cellWaveFunctionMatrixMemSpace.begin(), d_blockSize, strideA,
      shapeFunctionData.begin(), d_numberDofsPerElement, strideB, &beta,
      d_cellWaveFunctionQuadMatrixMemSpace.begin(), d_blockSize, strideC,
      d_numCells);

  unsigned int numVec = d_vectorList.size() / 2;
  d_MuMatrixMemSpaceCellWise.setValue(0.0);

  muMatrixMemSpaceKernel(d_numCells, numVec, d_numQuadsPerCell, d_blockSize,
                         d_BLASWrapperPtr, effectiveOrbitalOcc,
                         d_vectorListMemSpace,
                         d_cellWaveFunctionQuadMatrixMemSpace, inputJxwMemSpace,
                         d_MuMatrixMemSpaceCellWise);

  d_MuMatrixHostCellWise.copyFrom(d_MuMatrixMemSpaceCellWise);

  for (unsigned int iVecList = 0; iVecList < numVec; iVecList++) {
    unsigned int iVec = d_vectorList[2 * iVecList];
    unsigned int degenerateVecId = d_vectorList[2 * iVecList + 1];
    for (unsigned int iCell = 0; iCell < d_numCells; iCell++) {
      d_MuMatrixHost[iVec * d_blockSize + degenerateVecId] +=
          d_MuMatrixHostCellWise[iVecList + iCell * numVec];
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &d_MuMatrixHost[0], d_blockSize * d_blockSize,
                MPI_DOUBLE, MPI_SUM, mpi_communicator);


  for (unsigned int iVecList = 0; iVecList < numVec; iVecList++) {
    unsigned int iVec = d_vectorList[2 * iVecList];
    unsigned int degenerateVecId = d_vectorList[2 * iVecList + 1];
    pcout<<" Mu["<<iVec<<"]["<<degenerateVecId<<"]"<<d_MuMatrixHost[iVec * d_blockSize + degenerateVecId]<<"\n"; 
  }


  d_MuMatrixMemSpace.copyFrom(d_MuMatrixHost);
}

template <dftfe::utils::MemorySpace memorySpace>
void MultiVectorAdjointLinearSolverProblem<memorySpace>::computeRMatrix(
    dftfe::utils::MemoryStorage<double, memorySpace> &inputJxwMemSpace) {

  auto shapeFunctionDataTranspose =
      d_basisOperationsPtr->shapeFunctionData(true);
  auto shapeFunctionData = d_basisOperationsPtr->shapeFunctionData(false);

  rMatrixMemSpaceKernel(d_numCells, d_numberDofsPerElement, d_numQuadsPerCell,
                        d_BLASWrapperPtr, shapeFunctionData,
                        shapeFunctionDataTranspose, d_inputJxWMemSpace,
                        d_RMatrixMemSpace);
}

template <dftfe::utils::MemorySpace memorySpace>
void MultiVectorAdjointLinearSolverProblem<memorySpace>::distributeX() {

	unsigned int numDegenerateVec = d_vectorListMemSpace.size()/2;
	  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      dotProductHost(numDegenerateVec, 0.0);

  auto invSqrtMassMat = d_basisOperationsPtr->inverseSqrtMassVectorBasisData();

  std::vector<double> l2NormVec(d_blockSize, 0.0);

  d_BLASWrapperPtr->stridedBlockScaleCopy(
      d_blockSize, d_locallyOwnedSize, 1.0, invSqrtMassMat.data(),
      d_blockedXPtr->data(), d_blockedXPtr->data(), d_mapNodeIdToProcId.data());

  d_blockedXPtr->updateGhostValues();

  d_constraintsInfo.distribute(*d_blockedXPtr);

  multiVectorDotProdQuadWise(*d_blockedXPtr, *d_psiMemSpace, dotProductHost);

  dftfe::utils::MemoryStorage<dftfe::dataTypes::number, memorySpace>
      dotProductMemSpace(numDegenerateVec, 0.0);

  // Compute eta
  
  double sum_dFi_dmu = 0.0;
  double dFi_dmu =0.0;
  computeMuMatrix(d_inputJxWMemSpace, oneBlockSizeMemSpace, *d_psiMemSpace);
  
  double etaRightSide = 0.0;
  for (unsigned int iBlock = 0 ; iBlock <d_blockSize ; iBlock++)
  {
	  dFi_dmu  = (-dftfe::dftUtils::getPartialOccupancyDer(d_eigenValues[iBlock], d_fermiEnergy,  dftfe::C_kb, d_TVal));
  sum_dFi_dmu += dFi_dmu;
  etaRightSide += dFi_dmu*d_MuMatrixHost[iBlock*d_blockSize + iBlock];

  }

  double etaVal = 0.0;
if(std::abs(etaRightSide) < 1e-16)
{
	etaVal = 0.0; // keeping it to zero if the numerator is comparable to machine precision
}
else
{
	etaVal = etaRightSide/sum_dFi_dmu;
}

  pcout<<" etaVal = "<<etaVal<< " etaRightSide = "<<etaRightSide<<" sum_dFi_dmu = "<<sum_dFi_dmu<<"\n";


  for (unsigned int i = 0; i < numDegenerateVec; i++) {
	  unsigned int vec1 = d_vectorList[2*i];
	  unsigned int vec2 = d_vectorList[2*i + 1];
	  double DFi_epsi = dftfe::dftUtils::getPartialOccupancyDer(d_eigenValues[vec1], d_fermiEnergy,  dftfe::C_kb, d_TVal);
	  double correction = -DFi_epsi*d_MuMatrixHost[vec1*d_blockSize + vec2];
	  if(vec1 == vec2)
	  {
		  correction += DFi_epsi*etaVal;
	  }

	  pcout<<" iVec = "<<vec1<<" jVec = "<<vec2<<" dotProd = "<<dotProductHost[i]<<" corr = "<<correction<<"\n";
    dotProductHost[i] = -1.0 * dotProductHost[i] + correction;
    //dotProductHost[i] = -1.0 * dotProductHost[i];
   }

/*
      for (unsigned int i = 0; i < numDegenerateVec; i++) {
        dotProductHost[i] = -1.0 * dotProductHost[i]; 	      
      }
*/

  dotProductMemSpace.copyFrom(dotProductHost);

//  d_BLASWrapperPtr->stridedBlockScaleAndAddColumnWise(
//      d_blockSize, d_locallyOwnedSize, d_psiMemSpace->data(),
//      dotProductMemSpace.data(), d_blockedXPtr->data());

  removeNullSpace(d_blockSize, d_locallyOwnedSize, numDegenerateVec, d_vectorListMemSpace, *d_psiMemSpace, dotProductMemSpace, *d_blockedXPtr);

  d_blockedXPtr->updateGhostValues();

  /*
   *  For testing purpose
   */

  multiVectorDotProdQuadWise(*d_blockedXPtr, *d_psiMemSpace, dotProductHost);

   for (unsigned int i = 0; i < numDegenerateVec; i++) {
          unsigned int vec1 = d_vectorList[2*i];
          unsigned int vec2 = d_vectorList[2*i + 1];
          pcout<<" iVec = "<<vec1<<" jVec = "<<vec2<<" dotProd = "<<dotProductHost[i]<<"\n";
   }
   /*
   *  End testing purpose
   */
 

  MPI_Barrier(mpi_communicator);
/*	
	unsigned int numDegenerateVec = d_vectorListMemSpace.size()/2;
  dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      dotProductHost(numDegenerateVec, 0.0);

  auto invSqrtMassMat = d_basisOperationsPtr->inverseSqrtMassVectorBasisData();


  d_BLASWrapperPtr->stridedBlockScaleCopy(
      d_blockSize, d_locallyOwnedSize, 1.0, invSqrtMassMat.data(),
      d_blockedXPtr->data(), d_blockedXPtr->data(), d_mapNodeIdToProcId.data());

  d_blockedXPtr->updateGhostValues();

  d_constraintsInfo.distribute(*d_blockedXPtr);

  multiVectorDotProdQuadWise(*d_blockedXPtr, *d_psiMemSpace, dotProductHost);

  dftfe::utils::MemoryStorage<dftfe::dataTypes::number, memorySpace>
      dotProductMemSpace(numDegenerateVec, 0.0);

  for (unsigned int i = 0; i < numDegenerateVec; i++) {
    dotProductHost[i] = -1.0 * dotProductHost[i];
  }

  dotProductMemSpace.copyFrom(dotProductHost);

  removeNullSpace(d_blockSize, d_locallyOwnedSize, numDegenerateVec, d_vectorListMemSpace, *d_psiMemSpace, dotProductMemSpace, *d_blockedXPtr);

  d_blockedXPtr->updateGhostValues();
*/
 
}

template <dftfe::utils::MemorySpace memorySpace>
void MultiVectorAdjointLinearSolverProblem<memorySpace>::
    multiVectorDotProdQuadWise(
        dftfe::linearAlgebra::MultiVector<dftfe::dataTypes::number, memorySpace>
            &vec1,
        dftfe::linearAlgebra::MultiVector<dftfe::dataTypes::number, memorySpace>
            &vec2,
        dftfe::utils::MemoryStorage<dftfe::dataTypes::number,
                                    dftfe::utils::MemorySpace::HOST>
            &dotProductOutputHost) {

	    unsigned int numDegenerateVec = d_vectorListMemSpace.size()/2; 

	    //for(unsigned int iVec = 0; iVec < numDegenerateVec; iVec++)
		    //std::cout<<" iVec = "<<iVec<<" i = "<<d_vectorListMemSpace.data()[2*iVec]<<" j = "<<d_vectorListMemSpace.data()[2*iVec+1]<<"\n";
	    //std::cout<<" numDegenerateVec = "<<numDegenerateVec<<"\n";


	    std::cout<<std::flush;
	    MPI_Barrier(mpi_communicator);
 d_basisOperationsPtr->reinit(d_blockSize,
                               d_cellBlockSize,
                               d_matrixFreeQuadratureComponentRhs,
                               true,   // TODO should this be set to true
                               false); // TODO should this be set to true
                                       //

 tempOutputDotProdMemSpace.resize(numDegenerateVec);
 if ( vecOutputQuadValues.size() < numDegenerateVec * d_cellBlockSize *d_numQuadsPerCell)
 {
	 vecOutputQuadValues.resize(numDegenerateVec * d_cellBlockSize *d_numQuadsPerCell);
 }
  
 auto jxwVec = d_basisOperationsPtr->JxW();

 tempOutputDotProdMemSpace.resize(numDegenerateVec);
  tempOutputDotProdMemSpace.setValue(0.0);

  unsigned int one = 1;
  double oneDouble = 1.0;
  double zeroDouble = 0.0;


  for(unsigned int iCell = 0 ; iCell < d_numCells; iCell += d_cellBlockSize)
  {
          unsigned int cellEnd = iCell + d_cellBlockSize;
          cellEnd = std::min(cellEnd, d_numCells);
          unsigned int numCellTemp = cellEnd - iCell;
          
          d_basisOperationsPtr->interpolateKernel(vec1, vec1QuadValues.data(), nullptr , std::pair<unsigned int, unsigned int>(iCell, cellEnd));
          d_basisOperationsPtr->interpolateKernel(vec2, vec2QuadValues.data(), nullptr , std::pair<unsigned int, unsigned int>(iCell, cellEnd));
  
	  performHadamardProduct(d_blockSize,
                       numCellTemp * d_numQuadsPerCell,
                       numDegenerateVec,
                       d_vectorListMemSpace,
                       vec1QuadValues,
                       vec2QuadValues,
                       vecOutputQuadValues);


	  d_BLASWrapperPtr->xgemm(
      'N', 'T', one, numDegenerateVec, numCellTemp * d_numQuadsPerCell, &oneDouble,
      jxwVec.data() + iCell*d_numQuadsPerCell, one, vecOutputQuadValues.data(), numDegenerateVec,
      &oneDouble, tempOutputDotProdMemSpace.data(), one);
  
  }
 
  //d_BLASWrapperPtr->stridedBlockScaleCopy(
  //    d_blockSize, d_numCells * d_numQuadsPerCell, 1.0, jxwVec.data(),
  //    vec1QuadValues.data(), vec1QuadValues.data(), d_mapQuadIdToProcId.data());

  
  //performHadamardProduct(d_blockSize,
  //                     d_numCells * d_numQuadsPerCell,
  //                     numDegenerateVec,
  //                     d_vectorListMemSpace,
  //                     vec1QuadValues,
  //                     vec2QuadValues,
  //                     vecOutputQuadValues);

  //d_BLASWrapperPtr->hadamardProduct(
  //    d_blockSize * d_numCells * d_numQuadsPerCell, vec1QuadValues.data(),
  //    vec2QuadValues.data(), vecOutputQuadValues.data());

  //unsigned int one = 1;
  //double oneDouble = 1.0;
  //double zeroDouble = 0.0;
  //d_BLASWrapperPtr->xgemm(
  //    'N', 'T', one, numDegenerateVec, d_numCells * d_numQuadsPerCell, &oneDouble,
  //    d_onesQuadMemSpace.data(), one, vecOutputQuadValues.data(), numDegenerateVec,
  //    &zeroDouble, tempOutputDotProdMemSpace.data(), one);


 // d_BLASWrapperPtr->xgemm(
 //     'N', 'T', one, numDegenerateVec, d_numCells * d_numQuadsPerCell, &oneDouble,
 //     jxwVec.data(), one, vecOutputQuadValues.data(), numDegenerateVec,
 //     &zeroDouble, tempOutputDotProdMemSpace.data(), one);

  dotProductOutputHost.resize(numDegenerateVec);
  dotProductOutputHost.copyFrom(tempOutputDotProdMemSpace);

  MPI_Allreduce(MPI_IN_PLACE, &dotProductOutputHost[0], numDegenerateVec , MPI_DOUBLE,
                MPI_SUM, mpi_communicator);

 MPI_Barrier(mpi_communicator);
	    /*
	    d_cellBlockSize = d_numCells;

  d_basisOperationsPtr->reinit(d_blockSize,
                               d_cellBlockSize,
                               d_matrixFreeQuadratureComponentRhs,
                               true,   // TODO should this be set to true
                               false); // TODO should this be set to true
                                       //

  auto jxwVec = d_basisOperationsPtr->JxW();

  unsigned int numDegenerateVec = d_vectorListMemSpace.size()/2;

  vec1QuadValues.resize(d_cellBlockSize * d_blockSize*d_numQuadsPerCell);
  vec2QuadValues.resize(d_cellBlockSize * d_blockSize*d_numQuadsPerCell);
  vecOutputQuadValues.resize(numDegenerateVec * d_cellBlockSize *d_numQuadsPerCell);

 // 
 // if ( vec1QuadValues.size() < d_cellBlockSize * d_blockSize*d_numQuadsPerCell);
 // {
//	  vec1QuadValues.resize(d_cellBlockSize * d_blockSize*d_numQuadsPerCell);
//  }
//
//  if ( vec2QuadValues.size() < d_cellBlockSize * d_blockSize*d_numQuadsPerCell);
//  {
//          vec2QuadValues.resize(d_cellBlockSize * d_blockSize*d_numQuadsPerCell);
//  }
//
//  if (vecOutputQuadValues.size() < numDegenerateVec * d_cellBlockSize *d_numQuadsPerCell)
//  {
//	  vecOutputQuadValues.resize(numDegenerateVec * d_cellBlockSize *d_numQuadsPerCell);
//  }

  vecOutputQuadValues.setValue(0.0);
   unsigned int one = 1;
  double oneDouble = 1.0;
  double zeroDouble = 0.0;

  tempOutputDotProdMemSpace.resize(numDegenerateVec);
  tempOutputDotProdMemSpace.setValue(0.0);
  for(unsigned int iCell = 0 ; iCell < d_numCells; iCell += d_cellBlockSize)
  {
	  unsigned int cellEnd = iCell + d_cellBlockSize;
	  cellEnd = std::min(cellEnd, d_numCells);
	  unsigned int numCellTemp = cellEnd - iCell;
	  
	  d_basisOperationsPtr->interpolateKernel(vec1, vec1QuadValues.data(), nullptr , std::pair<unsigned int, unsigned int>(iCell, cellEnd));
	  d_basisOperationsPtr->interpolateKernel(vec2, vec2QuadValues.data(), nullptr , std::pair<unsigned int, unsigned int>(iCell, cellEnd));
	  //d_BLASWrapperPtr->stridedBlockScaleCopy(
      //d_blockSize, numCellTemp * d_numQuadsPerCell, 1.0, jxwVec.data() ,
      //vec1QuadValues.data(), vec1QuadValues.data(), d_mapQuadIdToProcId.data() + iCell*d_numQuadsPerCell);

       performHadamardProduct(d_blockSize,
		       numCellTemp * d_numQuadsPerCell, 
		       numDegenerateVec,
		       d_vectorListMemSpace,
		       vec1QuadValues, 
		       vec2QuadValues,
		       vecOutputQuadValues); 	  


//        d_BLASWrapperPtr->xgemm(
//      'N', 'T', one, numDegenerateVec, numCellTemp * d_numQuadsPerCell, &oneDouble,
//      d_onesQuadMemSpace.data(), one, vecOutputQuadValues.data(), numDegenerateVec,
//      &oneDouble, tempOutputDotProdMemSpace.data(), one);
 

        d_BLASWrapperPtr->xgemm(
      'N', 'T', one, numDegenerateVec, numCellTemp * d_numQuadsPerCell, &oneDouble,
      jxwVec.data() + iCell*d_numQuadsPerCell, one, vecOutputQuadValues.data(), numDegenerateVec,
      &oneDouble, tempOutputDotProdMemSpace.data(), one);
      }


  dotProductOutputHost.resize(numDegenerateVec);
  dotProductOutputHost.copyFrom(tempOutputDotProdMemSpace);

  MPI_Allreduce(MPI_IN_PLACE, &dotProductOutputHost[0], numDegenerateVec, MPI_DOUBLE,
                MPI_SUM, mpi_communicator);


   d_cellBlockSize = 100;
   d_cellBlockSize = std::min(d_cellBlockSize,d_numCells);
*/
}

template <dftfe::utils::MemorySpace memorySpace>
void MultiVectorAdjointLinearSolverProblem<memorySpace>::vmult(
    dftfe::linearAlgebra::MultiVector<dftfe::dataTypes::number, memorySpace>
        &Ax,
    dftfe::linearAlgebra::MultiVector<dftfe::dataTypes::number, memorySpace> &x,
    unsigned int blockSize) {
  Ax.setValue(0.0);

  d_ksOperatorPtr->HX(x,
                      1.0, // scalarHX,
                      0.0, // scalarY,
                      0.0, // scalarX
                      Ax,
                      false); // onlyHPrimePartForFirstOrderDensityMatResponse

  d_constraintsInfo.set_zero(x);
  d_constraintsInfo.set_zero(Ax);

  d_BLASWrapperPtr->stridedBlockScaleAndAddColumnWise(
      d_blockSize, d_locallyOwnedSize, x.data(),
      d_negEigenValuesMemSpace.data(), Ax.data());
}

template <dftfe::utils::MemorySpace memorySpace>
MultiVectorAdjointLinearSolverProblem<
    memorySpace>::~MultiVectorAdjointLinearSolverProblem() {}

template class MultiVectorAdjointLinearSolverProblem<
    dftfe::utils::MemorySpace::HOST>;

#ifdef DFTFE_WITH_DEVICE
template class MultiVectorAdjointLinearSolverProblem<
    dftfe::utils::MemorySpace::DEVICE>;
#endif

} // end of namespace invDFT
