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
//
// @authors Bikash Kanungo, Vishal Subramanian
//

#ifndef DFTFE_BFGSINVERSEDFTSOLVER_H
#define DFTFE_BFGSINVERSEDFTSOLVER_H

#include <headers.h>
#include <InverseDFTSolverFunction.h>

namespace invDFT
{
  template <unsigned int FEOrder, unsigned int FEOrderElectro, dftfe::utils::MemorySpace memorySpace>
  class BFGSInverseDFTSolver
  {
  public:
    enum class LSType
    {
      SECANT_LOSS,
      SECANT_FORCE_NORM,
      CP
    };

  public:
    BFGSInverseDFTSolver(int             numComponents,
                         double          tol,
                         double          lineSearchTol,
                         unsigned int    maxNumIter,
                         int             historySize,
                         int             numLineSearch,
                         const MPI_Comm &mpi_comm_parent);


    void
    solve(InverseDFTSolverFunction<FEOrder,FEOrderElectro,memorySpace> &      iDFTSolverFunction,
          const BFGSInverseDFTSolver::LSType lsType);

    void
    inverseJacobianTimesVec(const dftfe::distributedCPUVec<double> &g,
                            dftfe::distributedCPUVec<double> &      z,
                            const unsigned int               component,
                            InverseDFTSolverFunction<FEOrder,FEOrderElectro,memorySpace> &iDFTSolverFunction);

    void
    fnormCP(const std::vector<dftfe::distributedCPUVec<double>> &x,
            const std::vector<dftfe::distributedCPUVec<double>> &p,
            const std::vector<double> &                   alpha,
            std::vector<double> &                         fnorms,
            InverseDFTSolverFunction<FEOrder,FEOrderElectro,memorySpace> &                 iDFTSolverFunction);

    void
    solveLineSearchCP(std::vector<std::vector<double>> &lambda,
                      std::vector<std::vector<double>> &f,
                      const int                         maxIter,
                      const double                      tolerance,
                      InverseDFTSolverFunction<FEOrder,FEOrderElectro,memorySpace> &     iDFTSolverFunction);

    void
    solveLineSearchSecantLoss(std::vector<std::vector<double>> &lambda,
                              std::vector<std::vector<double>> &f,
                              const int                         maxIter,
                              const double                      tolerance,
                              InverseDFTSolverFunction<FEOrder,FEOrderElectro,memorySpace> &iDFTSolverFunction);

    void
    solveLineSearchSecantForceNorm(
      std::vector<std::vector<double>> &lambda,
      std::vector<std::vector<double>> &f,
      const int                         maxIter,
      const double                      tolerance,
      InverseDFTSolverFunction<FEOrder,FEOrderElectro,memorySpace> &     iDFTSolverFunction);

    std::vector<dftfe::distributedCPUVec<double>>
    getSolution() const;

  private:
    int                        d_numComponents;
    int                        d_maxNumIter;
    int                        d_historySize;
    int                        d_numLineSearch;
    int                        d_debugLevel;
    double                     d_tol;
    double                     d_lineSearchTol;
    dealii::ConditionalOStream pcout;

    std::vector<int>
      d_k; // stores the iteration for each component (i.e., each spin index)
    std::vector<dftfe::distributedCPUVec<double>>
      d_x; // stores the potential for each component
    std::vector<dftfe::distributedCPUVec<double>>
      d_g; // stores the force vector for each component
    std::vector<dftfe::distributedCPUVec<double>>
      d_p; // stores negative inverse Jacobian times the force vector for each
           // component
    // store the difference between successive force vectors for each component,
    // i.e., for a given component, y_k = g_{k+1} - g_k, where g is force vector
    // and k is the iteration index
    std::vector<std::list<dftfe::distributedCPUVec<double>>> d_y;
    // store the difference between successive potential vectors for each
    // component i.e., for each component, s_k = x_{k+1} - x_k
    std::vector<std::list<dftfe::distributedCPUVec<double>>> d_s;
    // store \rho_k = 1.0/(dot_product(y_k,s_k)) for each component
    std::vector<std::list<double>> d_rho;
  };
} // namespace invDFT
#endif // DFTFE_BFGSINVERSEDFTSOLVER_H
