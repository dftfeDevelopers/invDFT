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

#ifndef INVDFT_SLATERFUNCTIONMANAGER_H
#define INVDFT_SLATERFUNCTIONMANAGER_H


#if defined(HAVE_CONFIG_H)
#include "dft_config.h"
#endif // HAVE_CONFIG_H

#include <string>
#include <vector>
#include <set>

namespace invDFT {


    /**
     * @brief Class to read and store the Slater basis parameters to construct the input density
     *
     */
    class SlaterFunctionManager {

        //
        // types
        //
    public:

        struct slaterBasis {
            double alpha; // exponent of the basis
            int n; // principal quantum number
            int l; // azimuthal(angular) quantum number
            int m; // magnetic quantum number
            double normConst; // normalization constant
        };

        struct basis {
            const slaterBasis * sb; // pointer to the Slater basis
            const double * origin; // Origin or position of the atom about which the Slater is defined
            double basisNormConst; // normalization constant
        };
        //
        // methods
        //
    public:

        /**
         * @brief Constructor
         */
        SlaterFunctionManager(const std::string densityMatFilename,
                              const std::string smatrixFilename,
                              const std::string atomicCoordsFilename);


        /**
         * @brief Destructor
         *
         */
        ~SlaterFunctionManager();

        /**
         * @brief get the value of the density at a point
         *
         * @param point Point at which the density is to be computed
         *
         * @return value of density at the point
         */
        double getRhoValue(const double * point);
        std::vector<double> getRhoGradient(const double * point);
        double getRhoLaplacian(const double * point);

        /**
         * @brief get the gradient of the density at a point
         *
         * @param point Point at which the gradient is to be computed
         *
         * @return value of gradient density at the point
         */
        //std::vector<double> getRhoGradient(const double * point);

        /**
         * @brief get the value of a basis function at a given point
         *
         * @param basisId Id of the basis function
         * @param point Point at which the density is to be computed
         *
         * @return value of the basis function at the point
         */
        double getBasisFunctionValue(const int basisId, const double * point);
        std::vector<double> getBasisFunctionGradient(const int basisId, const double * point);
        double getBasisFunctionLaplacian(const int basisId, const double * point);

        /**
         * @brief get the gradient of a basis function at a given point
         *
         * @param basisId Id of the basis function
         * @param point Point at which the gradient is to be computed
         *
         * @return gradient of the basis function at the point
         */
        //std::vector<double>
        //	getBasisFunctionGradientValue(const int basisId, const double * point);

        /**
         * @brief get the number of basis functions
         *
         * @return Number of basis functions
         */
        int getNumberBasisFunctions();

        std::vector<std::vector<double> >
        getEvaluatedSMat();

//        std::vector<double>
//        getProjectedMO(QuadratureValuesContainer<DoubleVector>  MOInput,
//                       const int meshId);
//
//        QuadratureValuesContainer<DoubleVector>
//        computeMOSlaterFunction(const QuadratureValuesContainer<DoubleVector> & mo,
//                                const int slaterFunctionId,
//                                const int meshId);
//
//        QuadratureValuesContainer<DoubleVector>
//        computeMOFromDensityCoeffs(const std::vector<double> & densityCoeffs,
//                                   const int meshId);
    private:

        //
        // store atomic coordinates
        //
        std::vector<std::vector<double> > d_atomicCoords;

        //
        // store the density matrix
        //
        std::vector<std::vector<double> > d_densityMat;

        std::vector<std::vector<double> > d_SMat;

        //
        // store basis file names for each atom
        //
        std::vector<std::string> d_basisFileNames;

        //
        //store the unique basis file names
        //
        std::set<std::string> d_uniqueBasisFileNames;

        std::vector<std::vector<SlaterFunctionManager::slaterBasis* > > d_slaterBasisFunctions;

        //
        // store basis function paramters
        //
        std::vector<SlaterFunctionManager::basis*> d_basisFunctions;

        std::vector<double> d_SMatInvFlattened;
    };

}


#endif //INVDFT_SLATERFUNCTIONMANAGER_H
