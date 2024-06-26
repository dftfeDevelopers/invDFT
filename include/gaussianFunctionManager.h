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
// @author Bikash Kanungo
//


#ifndef DFTFE_GAUSSIANFUNCTIONMANAGER_H
#define DFTFE_GAUSSIANFUNCTIONMANAGER_H


#include "headers.h"
#include <dftUtils.h>

#include <string>
#include <vector>
#include <set>

namespace invDFT
{
  class gaussianFunctionManager
  {
    //
    // types
    //
  public:
    struct contractedGaussian
    {
      int     L;
      double *alpha;
      double *c;
      char    lquantum;
    };

    struct basis
    {
      int           L;     // Number of Gaussians used for contraction
      const double *alpha; // exponents of the L Gaussians used for contraction
      const double *c;     // Coefficients of for each of the
      int           l;     // l quantum number
      int           m;     // m quantum number
      double *normConsts;  // Normalization constant for each primitive Gaussian
      const double *origin;  // Origin or position of the atom about which the
                                // Gaussian is defined
      double basisNormConst; // Normalization constant for the resulting
                                // contracted Gaussian
    };


    //
    // methods
    //
  public:
    /**
     * @brief Constructor
     *
     * @param densityMatFilenames Vector of filenames containing the
     * density matrix for each spin. In the spin unpolarized case,
     * it should be of size 1 and the density matrix should
     * correspond to rho_up (=rho_down) and not the total density
     *
     * @param smatrixFilename Name of the file containing the overlap
     * for the gaussian basis. Ideally, it should not be needed, but we
     * provide it for sanity checks.
     *
     * @param atomicCoordsFilename File containing the coordinates of the atoms
     * and their gaussian basis file names. The format is as follows:
     * atom-symbol1 x1 y1 z1 gaussian-basis-file-name1
     * atom-symbol2 x2 y2 z2 gaussian-basis-file-name2
     * ....
     * ....
     * atom-symbolN xN yN zN gaussian-basis-file-nameN
     * @param unit Character to specify unit for the coordinates. Acceptable values:
     * 1. 'a' or 'A' for Angstrom
     * 2. 'b' or 'B' for bohr (atomic units)
     */
    gaussianFunctionManager(const std::vector<std::string> densityMatFilenames,
                            const std::string              atomicCoordsFilename,
                            const char                     unit,
                            const MPI_Comm &               mpi_comm_parent,
                            const MPI_Comm &               mpi_comm_domain);

    /**
     * @brief Initializes the gaussian basis based on a given set of
     * quadrature points and weights. This function must be called after
     * the constructor or anytime the user wants to update the quadrature
     * points and weights. This function internally performs the
     * following tasks:
     * 1. Evaluates the basis functions and their derivatives
     *	on the supplied quadrature
     * 2. Evaluates the overlap matrix based on the supplied quadrature
     * 3. Normalizes the basis functions based on the supplied quadrature
     *    Optionally, one can supply an external overlap matrix to check if
     *    the internal evaluation of the overlap matrix (based on the
     *    supplied quadrature) is consistent or not. Providing this external
     *    overlap matrix (from, say, gaussian basis based codes like QChem,
     *    PySCF, NWChem, etc.) is highly recommended.
     */
    void
    evaluateForQuad(const double *     points,
                    const double *     weights,
                    const unsigned int nPoints,
                    const bool         evalBasis,
                    const bool         evalBasisDerivatives,
                    const bool         evalBasisDoubleDerivatives,
                    const bool         evalSMat,
                    const bool         normalizeBasis,
                    const unsigned int quadIndex,
                    std::string        smatrixExternalFilename = "");


    /**
     * @brief Destructor
     */
    ~gaussianFunctionManager();

    /**
     * @brief Get the value of the density at a list of points
     *
     * @param points Pointer to the points at which the density is to be
     * computed. It is assumed to be of size 3*(number of points).
     * In other words, the j-th coordinate of the i-th point should be
     * given be points[i*3 + j]
     *
     * @param N The number of points
     * @param spinIndex index of the spin (0 or 1)
     * @param[out] rho pointer to density at the points
     * @note: rho must be appropriately pre-allocated to number of points(N)
     */
    void
    getRhoValue(const double *     points,
                const unsigned int N,
                const unsigned int spinIndex,
                double *           rho) const;

    /**
     * @brief Get the value of the density using a quadIndex. This
     * function should be used only after evaluateForQuad() is called, so
     * that the relevant basis data on the quadrature points that are part
     * of the quadIndex are pre-computed.
     *
     * @param quadIndex index of the quadrature grid on which to revaluate
     *  the density
     * @param spinIndex index of the spin (0 or 1)
     * @param[out] rho pointer to density at the quadrature points that are
     *  part of the quadIndex.
     * @note: rho must be appropriately pre-allocated to the number of points
     * belonging to quadIndex
     */
    void
    getRhoValue(const unsigned int quadIndex,
                const unsigned int spinIndex,
                double *           rho) const;

    /**
     * @brief get the gradient of the density at a list of points
     *
     * @param points Pointer to the points at which the density is to be
     * computed. It is assumed to be of size 3*(number of points).
     * In other words, the j-th coordinate of the i-th point should be
     * given be points[i*3 + j]
     *
     * @param N The number of points
     * @param spinIndex index of the spin (0 or 1)
     * @param[out] rhoGrad Pointer to gradient of the density at the
 points
     * rhoGrad is resized inside this function to appropriate size.
     * The derivative with respect to j-th coordinate for the i-th point
 is
     * given by rhoGrad[i*3 + j]
     */
    void
    getRhoGradient(const double *x, const int
                                        spinIndex,  std::vector<double> &returnValue);

    //    	  /**
    //    	   * @brief Get the value of the gradient of the density using a quadIndex.
    //    	   * This function should be used only after evaluateForQuad() is
    //     called,
    //    	   * so that the relevant basis data on the quadrature points that are
    //     part
    //    	   * of the quadIndex are pre-computed.
    //    	   *
    //    	   * @param quadIndex index of the quadrature grid on which to revaluate
    //    	   *  the density
    //    	   * @param spinIndex index of the spin (0 or 1)
    //    	   * @param[out] rhoGrad Pointer to gradient of the density at the
    //    	   *  quadrature points that are part of the quadIndex.
    //    	   *  rhoGrad is resized inside this function to appropriate size.
    //    	   *  The derivative with respect to j-th coordinate for the i-th point
    //    	   *  is given by  rhoGrad[i*3 + j]
    //    	   */
    //    	  void
    //    		getRhoGradient(const unsigned int quadIndex,
    //    			const unsigned int spinIndex,
    //    			double * rhoGrad) const;
    //
    //	  /**
    //	   * @brief get the laplacian of the density at a list of points
    //	   *
    //	   * @param points Pointer to the points at which the density is to be
    //	   * computed. It is assumed to be of size 3*(number of points).
    //	   * In other words, the j-th coordinate of the i-th point should be
    //	   * given be points[i*3 + j]
    //	   *
    //	   * @param N The number of points
    //	   * @param spinIndex index of the spin (0 or 1)
    //	   * @param[out] rhoLap Pointer to laplacian of density at the points.
    //	   *  rhoLap is resized appropriately inside the function.
    //	   */
    //	  void
    //		getRhoLaplacian(const double * points, const unsigned int N,
    //			const unsigned int spinIndex, double * rhoLap) const;
    //
    //	  /**
    //	   * @brief Get the laplacian of the density using a quadIndex.
    //	   * This function should be used only after evaluateForQuad() is
    // called,
    //	   * so that the relevant basis data on the quadrature points that are
    // part
    //	   * of the quadIndex are pre-computed.
    //	   *
    //	   * @param quadIndex index of the quadrature grid on which to revaluate
    //	   *  the density
    //	   * @param spinIndex index of the spin (0 or 1)
    //	   * @param[out] rhoLap Pointer to the laplacian of the density at the
    //	   *  quadrature points that are part of the quadIndex.
    //	   *  rhoLap is resized inside this function to appropriate size.
    //	   */
    //	  void
    //		getRhoLaplacian(const unsigned int quadIndex,
    //			const unsigned int spinIndex,
    //			double * rhoLap) const;
    //
    //	  /**
    //	   * @brief get the double derivatives of the density at a list of points
    //	   *
    //	   * @param points Pointer to the points at which the density is to be
    //	   * computed. It is assumed to be of size 3*(number of points).
    //	   * In other words, the j-th coordinate of the i-th point should be
    //	   * given be points[i*3 + j]
    //	   *
    //	   * @param N The number of points
    //	   * @param spinIndex index of the spin (0 or 1)
    //	   * @param[out] rhoDD Pointer to double derivative of the density at
    // the
    //	   * points. rhoDD is resized appropriately inside the function.
    //	   * The derivative with respect to (j,k) pair of coordinates for the
    //	   * i-th point is given by rhoDD[i*9 + j*3 + k]
    //	   */
    //	  void
    //		getRhoDoubleDerivative(const double * points, const unsigned int N,
    //			const unsigned int spinIndex, double * rhoDD) const;
    //
    //	  /**
    //	   * @brief Get the double derivative of the density using a quadIndex.
    //	   * This function should be used only after evaluateForQuad() is
    // called,
    //	   * so that the relevant basis data on the quadrature points that are
    // part
    //	   * of the quadIndex are pre-computed.
    //	   *
    //	   * @param quadIndex index of the quadrature grid on which to revaluate
    //	   *  the density
    //	   * @param spinIndex index of the spin (0 or 1)
    //	   * @param[out] rhoDD Pointer to double derivative of the density at
    // the
    //	   *  quadrature points that are part of the quadIndex.
    //	   *  rhoDD is resized inside this function to appropriate size.
    //	   *  The derivative with respect to (j,k) pair of coordinates for the
    //       *  i-th point is given by rhoDD[i*9 + j*3 + k]
    //	   */
    //	  void
    //		getRhoDoubleDerivative(const unsigned int quadIndex,
    //			const unsigned int spinIndex,
    //			double * rhoDD) const;

    /**
     * @brief get the value of a basis function at a given point
     *
     * @param basisId Id of the basis function
     * @param point Point at which the density is to be computed
     *
     * @return value of the basis function at the point
     */
    double
    getBasisFunctionValue(const int basisId, const double *point) const;

    /**
     * @brief get the gradient of a basis function at a given point
     *
     * @param basisId Id of the basis function
     * @param point Point at which the density is to be computed
     *
     * @return vector containing the gradient of the basis function at the point
     */
    std::vector<double>
    getBasisFunctionGradient(const int basisId, const double *point) const;

    /**
     * @brief get the double derivatives of the basis function at a point
     *
     * @param point Point at which the basis function is to be computed
     *
     * @return double derivatives of the basis function at the point
     */
    std::vector<double>
    getBasisFunctionDoubleDerivatives(const int     basisId,
                                      const double *point) const;

    /**
     * @brief get the laplacian of a basis function at a given point
     *
     * @param basisId Id of the basis function
     * @param point Point at which the density is to be computed
     *
     * @return laplacian of the basis function at the point
     */
    double
    getBasisFunctionLaplacian(const int basisId, const double *point) const;

    /**
     * @brief get the number of basis functions
     *
     * @return Number of basis functions
     */
    unsigned int
    getNumberBasisFunctions() const;

    const std::vector<double> &
    getSMat(const unsigned int quadIndex) const;

    const std::vector<double> &
    getDensityMat(const int spinIndex) const;

  private:
    //
    // store atomic coordinates
    //
    std::vector<std::vector<double>> d_atomicCoords;

    //
    // store the density matrix
    //
    std::vector<std::vector<double>> d_densityMats;


    //
    // store basis file names for each atom
    //
    std::vector<std::string> d_basisFileNames;

    //
    // store the unique basis file names
    //
    std::set<std::string> d_uniqueBasisFileNames;

    //
    // store the contracted gaussians for each atom type
    //
    std::vector<std::vector<gaussianFunctionManager::contractedGaussian *>>
      d_contractedGaussians;

    //
    // store basis function paramters
    //
    std::vector<gaussianFunctionManager::basis *> d_basisFunctions;


    unsigned int d_numSpins;

    dealii::ConditionalOStream pcout;
    MPI_Comm                   d_mpiComm_domain, d_mpiComm_parent;

    std::vector<double> d_basisCutoff;
    // IDs of basis whose compact support intersects with the
    // domain (defined by list of quad points supplied) belonging to this
    // processor
    std::map<unsigned int, std::vector<unsigned int>>
                                                d_basisIdsWithCompactSupportInProc;
    std::map<unsigned int, unsigned int>        d_numQuadPoints;
    std::map<unsigned int, std::vector<double>> d_basisVals;
    std::map<unsigned int, std::vector<double>> d_basisDerivatives;
    std::map<unsigned int, std::vector<double>> d_basisDoubleDerivatives;
    std::map<unsigned int, std::vector<double>> d_SMat;
  };
} // namespace invDFT
#endif // DFTFE_GAUSSIANFUNCTIONMANAGER_H
