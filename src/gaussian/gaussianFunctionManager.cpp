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
#include "gaussianFunctionManager.h"
#include "constants.h"

#include <boost/math/special_functions/factorials.hpp>
#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>

#include <cerrno>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
//
//
//
/**
 * For the definition of the associated Legendre polynomials i.e. Plm() and
 * their derivatives (as used for evaluating the real form of spherical
 * harmonics and their derivatives) refer:
 * @article{bosch2000computation,
 *  	   title={On the computation of derivatives of Legendre functions},
 *     	   author={Bosch, W},
 *         journal={Physics and Chemistry of the Earth, Part A: Solid Earth and
 * Geodesy}, volume={25}, number={9-11}, pages={655--659}, year={2000},
 *         publisher={Elsevier}
 *        }
 */

namespace invDFT {
#define DIST_TOL 1e-8

//
// TODO read them from dft params
//
#define DFTFE_ZERO_RADIUS_TOL 1e-15
#define DFTFE_POLAR_ANGLE_TOL 1e-12
#define DFTFE_GAUSSIAN_ZERO_TOL 1e-15
#define DFTFE_SMAT_DIFF_TOL 1e-4
#define DFTFE_FINITE_DIFF_H 1e-3
namespace {
template <typename T, std::size_t N> std::size_t length_of(T const (&)[N]) {
  return N;
}

bool isNumber(double &i, std::string s) {
  char *end;
  double d;
  errno = 0;
  d = strtod(s.c_str(), &end);
  if ((errno == ERANGE && d == std::numeric_limits<double>::max())) {
    return false;
  }

  if ((errno == ERANGE && d == std::numeric_limits<double>::min())) {
    return false;
  }

  if (s.size() == 0 || *end != 0) {
    return false;
  }

  i = d;
  return true;
}

bool isInteger(int &i, std::string s, const int base = 10) {
  char *end;
  long l;
  errno = 0;
  l = strtol(s.c_str(), &end, base);
  if ((errno == ERANGE && l == std::numeric_limits<long>::max()) ||
      l > std::numeric_limits<int>::max()) {
    return false;
  }

  if ((errno == ERANGE && l == std::numeric_limits<long>::min()) ||
      l < std::numeric_limits<int>::min()) {
    return false;
  }

  if (s.size() == 0 || *end != 0) {
    return false;
  }

  i = l;
  return true;
}

std::string trim(const std::string &str,
                 const std::string &whitespace = " \t") {
  std::size_t strBegin = str.find_first_not_of(whitespace);
  if (strBegin == std::string::npos)
    return ""; // no content
  std::size_t strEnd = str.find_last_not_of(whitespace);
  std::size_t strRange = strEnd - strBegin + 1;
  return str.substr(strBegin, strRange);
}

double getDistance(const double *x, const double *y) {
  double r = 0.0;
  for (unsigned int i = 0; i < 3; ++i)
    r += pow(x[i] - y[i], 2.0);
  return sqrt(r);
}

int factorial(int n) {
  if (n == 0)
    return 1;
  else
    return n * factorial(n - 1);
}

double doubleFactorial(int n) {
  if (n == 0 || n == -1)
    return 1.0;
  return n * doubleFactorial(n - 2);
}

double dotProduct(const std::vector<double> &x, const std::vector<double> &y) {
  double returnValue = 0.0;
  for (unsigned int i = 0; i < x.size(); ++i)
    returnValue += x[i] * y[i];

  return returnValue;
}

void convertCartesianToSpherical(const std::vector<double> &x, double &r,
                                 double &theta, double &phi) {
  r = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
  if (r == 0) {
    theta = 0.0;
    phi = 0.0;
  }

  else {
    theta = acos(x[2] / r);
    //
    // check if theta = 0 or PI (i.e, whether the point is on the Z-axis)
    // If yes, assign phi = 0.0.
    // NOTE: In case theta = 0 or PI, phi is undetermined. The actual
    // value of phi doesn't matter in computing the enriched function
    // value or its gradient. We assign phi = 0.0 here just as a dummy
    // value
    //
    if (fabs(theta - 0.0) >= DFTFE_POLAR_ANGLE_TOL &&
        fabs(theta - M_PI) >= DFTFE_POLAR_ANGLE_TOL) {
      phi = atan2(x[1], x[0]);
    } else {
      phi = 0.0;
    }
  }
}

std::vector<double> getFDCoeffs(const int numFDPoints) {
  std::vector<double> returnValue(numFDPoints);
  if (numFDPoints == 3) {
    returnValue[0] = -1.0 / 2.0;
    returnValue[1] = 0.0;
    returnValue[2] = 1.0 / 2.0;
  }

  else if (numFDPoints == 5) {
    returnValue[0] = 1.0 / 12.0;
    returnValue[1] = -2.0 / 3.0;
    returnValue[2] = 0.0;
    returnValue[3] = 2.0 / 3.0;
    returnValue[4] = -1.0 / 12.0;
  }

  else if (numFDPoints == 7) {
    returnValue[0] = -1.0 / 60.0;
    returnValue[1] = 3.0 / 20.0;
    returnValue[2] = -3.0 / 4.0;
    returnValue[3] = 0.0;
    returnValue[4] = 3.0 / 4.0;
    returnValue[5] = -3.0 / 20.0;
    returnValue[6] = 1.0 / 60.0;
  }

  else if (numFDPoints == 9) {
    returnValue[0] = 1.0 / 280.0;
    returnValue[1] = -4.0 / 105.0;
    returnValue[2] = 1.0 / 5.0;
    returnValue[3] = -4.0 / 5.0;
    returnValue[4] = 0.0;
    returnValue[5] = 4.0 / 5.0;
    returnValue[6] = -1.0 / 5.0;
    returnValue[7] = 4.0 / 105.0;
    returnValue[8] = -1.0 / 280.0;
  }

  else if (numFDPoints == 11) {
    returnValue[0] = -2.0 / 2520.0;
    returnValue[1] = 25.0 / 2520.0;
    returnValue[2] = -150.0 / 2520.0;
    returnValue[3] = 600.0 / 2520.0;
    returnValue[4] = -2100.0 / 2520.0;
    returnValue[5] = 0.0 / 2520.0;
    returnValue[6] = 2100.0 / 2520.0;
    returnValue[7] = -600.0 / 2520.0;
    returnValue[8] = 150.0 / 2520.0;
    returnValue[9] = -25.0 / 2520.0;
    returnValue[10] = 2.0 / 2520.0;
  }

  else if (numFDPoints == 13) {
    returnValue[0] = 5.0 / 27720.0;
    returnValue[1] = -72.0 / 27720.0;
    returnValue[2] = 495.0 / 27720.0;
    returnValue[3] = -2200.0 / 27720.0;
    returnValue[4] = 7425.0 / 27720.0;
    returnValue[5] = -23760 / 27720.0;
    returnValue[6] = 0.0 / 27720.0;
    returnValue[7] = 23760.0 / 27720.0;
    returnValue[8] = -7425.0 / 27720.0;
    returnValue[9] = 2200.0 / 27720.0;
    returnValue[10] = -495.0 / 27720.0;
    returnValue[11] = 72.0 / 27720.0;
    returnValue[12] = -5.0 / 27720.0;
  }

  else {
    const std::string message("Invalid number of FD points. Please enter "
                              "number of FD points as 3, 5, 7, 9, 11 or 13.");
    AssertThrow(false, dealii::ExcMessage(message));
  }

  return returnValue;
}

std::vector<double> getNormConsts(const double *alpha, const int l,
                                  const int L) {
  std::vector<double> returnValue(L);
  for (unsigned int i = 0; i < L; ++i) {
    const double term1 = doubleFactorial(2 * l + 1) * sqrt(M_PI);
    const double term2 = pow(2.0, 2 * l + 3.5) * pow(alpha[i], l + 1.5);
    const double overlapIntegral = term1 / term2;
    returnValue[i] = 1.0 / sqrt(overlapIntegral);
  }

  return returnValue;
}

inline double Dm(const int m) {
  if (m == 0)
    return 1.0 / sqrt(2 * M_PI);
  else
    return 1.0 / sqrt(M_PI);
}

double Blm(const int l, const int m) {
  if (m == 0)
    return sqrt((2.0 * l + 1) / 2.0);
  else
    return Blm(l, m - 1) / sqrt((l - m + 1.0) * (l + m));
}

inline double Clm(const int l, const int m) {
  Assert(m >= 0 && m <= l,
         dealii::ExcMessage(
             "m value in Clm() function inside "
             "gaussianFunctionManager.cc must be a positive integer"));
  /*return sqrt(((2.0 * l + 1) * boost::math::factorial<double>(l - m)) /
              (2.0 * boost::math::factorial<double>(l + m)));
*/
  return Blm(l, abs(m));
}

double Qm(const int m, const double phi) {
  double returnValue = 0.0;
  if (m > 0)
    returnValue = cos(m * phi);
  if (m == 0)
    returnValue = 1.0;
  if (m < 0)
    returnValue = sin(std::abs(m) * phi);

  return returnValue;
}

double dQmDPhi(const int m, const double phi) {
  if (m > 0)
    return -m * sin(m * phi);
  else if (m == 0)
    return 0.0;
  else
    return std::abs(m) * cos(std::abs(m) * phi);
}

double Plm(const int l, const int m, const double x) {
  if (std::abs(m) > l)
    return 0.0;
  else
    //
    // NOTE: Multiplies by {-1}^m to remove the
    // implicit Condon-Shortley factor in the associated legendre
    // polynomial implementation of boost
    // This is done to be consistent with the QChem's implementation
    return pow(-1.0, m) * boost::math::legendre_p(l, m, x);
}

double dPlmDTheta(const int l, const int m, const double theta) {
  const double cosTheta = cos(theta);
  if (std::abs(m) > l)
    return 0.0;

  else if (l == 0)
    return 0.0;

  else if (m < 0) {
    const int modM = std::abs(m);
    const double factor = pow(-1, m) *
                          boost::math::factorial<double>(l - modM) /
                          boost::math::factorial<double>(l + modM);
    return factor * dPlmDTheta(l, modM, theta);
  }

  else if (m == 0) {
    return -1.0 * Plm(l, 1, cosTheta);
  }

  else if (m == l)
    return l * Plm(l, l - 1, cosTheta);

  else {
    const double term1 = (l + m) * (l - m + 1) * Plm(l, m - 1, cosTheta);
    const double term2 = Plm(l, m + 1, cosTheta);
    return 0.5 * (term1 - term2);
  }
}

double d2PlmDTheta2(const int l, const int m, const double theta) {
  const double cosTheta = cos(theta);
  if (std::abs(m) > l)
    return 0.0;

  else if (l == 0)
    return 0.0;

  else if (m < 0) {
    const int modM = std::abs(m);
    const double factor = pow(-1, m) *
                          boost::math::factorial<double>(l - modM) /
                          boost::math::factorial<double>(l + modM);
    return factor * d2PlmDTheta2(l, modM, theta);
  }

  else if (m == 0)
    return -1.0 * dPlmDTheta(l, 1, theta);

  else if (m == l)
    return l * dPlmDTheta(l, l - 1, theta);

  else {
    double term1 = (l + m) * (l - m + 1) * dPlmDTheta(l, m - 1, theta);
    double term2 = dPlmDTheta(l, m + 1, theta);
    return 0.5 * (term1 - term2);
  }
}

double gfRadialPart(const double r, const int l, const double alpha) {
  return pow(r, l) * exp(-alpha * r * r);
}

double gfRadialPartDerivative(const double r, const double alpha, const int l,
                              const int derOrder) {
  if (derOrder == 0 && l >= 0)
    return pow(r, l) * exp(-alpha * r * r);
  else if (derOrder == 0 && l < 0)
    return 0.0;
  else
    return (l * gfRadialPartDerivative(r, alpha, l - 1, derOrder - 1) -
            2 * alpha * gfRadialPartDerivative(r, alpha, l + 1, derOrder - 1));
}

double getLimitingValueLaplacian(const int l, const int m, const double theta) {
  double returnValue = 0.0;
  if (std::fabs(theta - 0.0) < DFTFE_POLAR_ANGLE_TOL) {
    if (m == 0)
      returnValue = -0.5 * l * (l + 1);
    if (m == 2)
      returnValue = 0.25 * (l - 1) * l * (l + 1) * (l + 2);
  }

  if (std::fabs(theta - M_PI) < DFTFE_POLAR_ANGLE_TOL) {
    if (m == 0)
      returnValue = -0.5 * l * (l + 1) * pow(-1.0, l);
    if (m == 2)
      returnValue = 0.25 * (l - 1) * l * (l + 1) * (l + 2) * pow(-1.0, l);
    ;
  }

  return returnValue;
}

double evaluateBasisValue(const invDFT::gaussianFunctionManager::basis *b,
                          const double *x) {
  const int L = b->L;
  const double *R = b->origin;
  const int l = b->l;
  const int m = b->m;

  std::vector<double> dx(3, 0.0);
  for (unsigned int i = 0; i < 3; ++i)
    dx[i] = x[i] - R[i];

  double r, theta, phi;
  convertCartesianToSpherical(dx, r, theta, phi);

  double returnValue = 0.0;
  for (unsigned int i = 0; i < L; ++i) {
    const double alphaVal = b->alpha[i];
    const double cVal = b->c[i];
    const double norm = b->normConsts[i];
    returnValue += cVal * norm * gfRadialPart(r, l, alphaVal);
  }

  const int modM = std::abs(m);
  const double C = Clm(l, modM) * Dm(m);
  const double cosTheta = cos(theta);
  const double P = Plm(l, modM, cosTheta);
  const double Q = Qm(m, phi);

  returnValue *= C * P * Q;
  returnValue *= b->basisNormConst;
  return returnValue;
}

std::vector<double> getSphericalGaussianGradient(const std::vector<double> &x,
                                                 const int l, const int m,
                                                 const double alpha) {
  double r, theta, phi;
  convertCartesianToSpherical(x, r, theta, phi);
  std::vector<double> returnValue(3);
  const int modM = std::abs(m);
  const double C = Clm(l, modM) * Dm(m);
  if (r < DFTFE_ZERO_RADIUS_TOL) {
    if (l == 1) {
      if (m == -1) {
        returnValue[0] = 0.0;
        returnValue[1] = C;
        returnValue[2] = 0.0;
      }

      if (m == 0) {
        returnValue[0] = 0.0;
        returnValue[1] = 0.0;
        returnValue[2] = C;
      }

      if (m == 1) {
        returnValue[0] = C;
        returnValue[1] = 0.0;
        returnValue[2] = 0.0;
      }
    }

    else {
      returnValue[0] = 0.0;
      returnValue[1] = 0.0;
      returnValue[2] = 0.0;
    }
  }

  else if (std::fabs(theta - 0.0) < DFTFE_POLAR_ANGLE_TOL) {
    const double R = gfRadialPart(r, l, alpha);
    const double dRDr = gfRadialPartDerivative(r, alpha, l, 1);
    const double cosTheta = cos(theta);
    const double P = Plm(l, modM, cosTheta);
    if (m == 0) {
      returnValue[0] = 0.0;
      returnValue[1] = 0.0;
      returnValue[2] = C * dRDr * P * cosTheta;
    }

    else if (m == 1) {
      returnValue[0] = C * (R / r) * l * (l + 1) / 2.0;
      returnValue[1] = 0.0;
      returnValue[2] = 0.0;
    }

    else if (m == -1) {
      returnValue[0] = 0.0;
      returnValue[1] = C * (R / r) * l * (l + 1) / 2.0;
      returnValue[2] = 0.0;
    }

    else {
      returnValue[0] = 0.0;
      returnValue[1] = 0.0;
      returnValue[2] = 0.0;
    }
  }

  else if (std::fabs(theta - M_PI) < DFTFE_POLAR_ANGLE_TOL) {
    const double R = gfRadialPart(r, l, alpha);
    const double dRDr = gfRadialPartDerivative(r, alpha, l, 1);
    const double cosTheta = cos(theta);
    const double P = Plm(l, modM, cosTheta);
    if (m == 0) {
      returnValue[0] = 0.0;
      returnValue[1] = 0.0;
      returnValue[2] = C * dRDr * P * cosTheta;
    }

    else if (m == 1) {
      returnValue[0] = C * (R / r) * l * (l + 1) / 2.0 * pow(-1, l + 1);
      returnValue[1] = 0.0;
      returnValue[2] = 0.0;
    }

    else if (m == -1) {
      returnValue[0] = 0.0;
      returnValue[1] = C * (R / r) * l * (l + 1) / 2.0 * pow(-1, l + 1);
      returnValue[2] = 0.0;
    }

    else {
      returnValue[0] = 0.0;
      returnValue[1] = 0.0;
      returnValue[2] = 0.0;
    }
  }

  else {
    const double R = gfRadialPart(r, l, alpha);
    const double dRDr = gfRadialPartDerivative(r, alpha, l, 1);
    const double cosTheta = cos(theta);
    const double P = Plm(l, modM, cosTheta);
    const double dPDTheta = dPlmDTheta(l, modM, theta);
    const double Q = Qm(m, phi);
    const double dQDPhi = dQmDPhi(m, phi);
    double jacobianInverse[3][3];
    jacobianInverse[0][0] = sin(theta) * cos(phi);
    jacobianInverse[0][1] = cos(theta) * cos(phi) / r;
    jacobianInverse[0][2] = -1.0 * sin(phi) / (r * sin(theta));
    jacobianInverse[1][0] = sin(theta) * sin(phi);
    jacobianInverse[1][1] = cos(theta) * sin(phi) / r;
    jacobianInverse[1][2] = cos(phi) / (r * sin(theta));
    jacobianInverse[2][0] = cos(theta);
    jacobianInverse[2][1] = -1.0 * sin(theta) / r;
    jacobianInverse[2][2] = 0.0;

    double partialDerivatives[3];
    partialDerivatives[0] = dRDr * P * Q;
    partialDerivatives[1] = R * dPDTheta * Q;
    partialDerivatives[2] = R * P * dQDPhi;
    for (unsigned int i = 0; i < 3; ++i) {
      returnValue[i] = C * (jacobianInverse[i][0] * partialDerivatives[0] +
                            jacobianInverse[i][1] * partialDerivatives[1] +
                            jacobianInverse[i][2] * partialDerivatives[2]);
    }
  }

  return returnValue;
}

double getSphericalGaussianLaplacian(const std::vector<double> &x, const int l,
                                     const int m, const double alpha) {
  double r, theta, phi;
  convertCartesianToSpherical(x, r, theta, phi);
  double returnValue = 0.0;
  if (r < DFTFE_ZERO_RADIUS_TOL) {
    const int modM = std::abs(m);
    const double C = Clm(l, modM) * Dm(m);
    if (l == 0)
      returnValue = -C * 6.0 * alpha;
    else
      returnValue = 0.0;
  }

  else {
    const int modM = std::abs(m);
    const double C = Clm(l, modM) * Dm(m);
    const double cosTheta = cos(theta);
    const double sinTheta = sin(theta);
    const double R = gfRadialPart(r, l, alpha);
    const double dRdr = gfRadialPartDerivative(r, alpha, l, 1);
    const double d2Rdr2 = gfRadialPartDerivative(r, alpha, l, 2);
    const double P = Plm(l, modM, cosTheta);
    const double Q = Qm(m, phi);
    const double term1 = C * P * Q * (2.0 * dRdr / r + d2Rdr2);

    if (std::fabs(theta - 0.0) < DFTFE_POLAR_ANGLE_TOL ||
        std::fabs(theta - M_PI) < DFTFE_POLAR_ANGLE_TOL) {
      const double limitingVal = getLimitingValueLaplacian(l, modM, theta);
      const double term2 = C * (R / (r * r)) * Q * (limitingVal + limitingVal);
      const double term3 = -C * m * m * (R / (r * r)) * Q * (limitingVal / 2.0);
      returnValue = (term1 + term2 + term3);
    }

    else {
      const double a = dPlmDTheta(l, modM, theta);
      const double b = d2PlmDTheta2(l, modM, theta);
      const double term2 =
          C * (R / (r * r)) * Q * ((cosTheta / sinTheta) * a + b);
      const double term3 =
          -C * m * m * (R / (r * r)) * Q * P / (sinTheta * sinTheta);
      returnValue = term1 + term2 + term3;
    }
  }

  return returnValue;
}

double getBasisHigherOrderDerivativeFD(
    const double *point, const invDFT::gaussianFunctionManager::basis *basis,
    std::vector<int> &indices, const int numFDPoints, const double h) {
  int numIndices = indices.size();
  double returnValue = 0.0;
  if (numIndices == 0) {
    returnValue = evaluateBasisValue(basis, point);
    return returnValue;
  }

  else {
    int currentIndex = indices[numIndices - 1];
    std::vector<int> indicesNext(numIndices - 1);
    for (unsigned int i = 0; i < numIndices - 1; ++i)
      indicesNext[i] = indices[i];

    std::vector<double> coeffs = getFDCoeffs(numFDPoints);
    for (unsigned int iPoint = 0; iPoint < numFDPoints; ++iPoint) {
      std::vector<double> FDPoint(3);
      FDPoint[0] = point[0];
      FDPoint[1] = point[1];
      FDPoint[2] = point[2];
      int shiftIndex = (int)(iPoint - numFDPoints / 2);
      FDPoint[currentIndex] += shiftIndex * h;
      const double factor = coeffs[iPoint] / h;
      returnValue +=
          factor * getBasisHigherOrderDerivativeFD(&FDPoint[0], basis,
                                                   indicesNext, numFDPoints, h);
    }

    return returnValue;
  }
}

void readAtomicCoordsAndBasisFileNames(
    std::vector<std::vector<double>> &atomicCoords,
    std::vector<std::string> &basisFileNames, std::string fileName) {
  std::ifstream readFile;
  readFile.open(fileName.c_str());
  AssertThrow(readFile.is_open(),
              dealii::ExcMessage("Could not find file " + fileName));

  std::string readLine;
  while (std::getline(readFile, readLine)) {
    AssertThrow(!readLine.empty(),
                dealii::ExcMessage("Empty invalid found while reading atomic "
                                   " coordinates in file " +
                                   fileName));
    std::istringstream lineString(readLine);
    std::string word;
    unsigned int count = 0;
    std::vector<double> coord(0);
    while (lineString >> word) {
      std::string wordTrimmed = trim(word);
      AssertThrow(
          !wordTrimmed.empty(),
          dealii::ExcMessage("Empty column entry found while reading atomic "
                             "coordinates in file" +
                             fileName));
      if (count == 0) {
        count++;
        continue;
      }
      if (count >= 1 && count <= 3) {
        double val;
        if (isNumber(val, wordTrimmed))
          coord.push_back(val);
        else {
          AssertThrow(false, dealii::ExcMessage("Coordinate entry in file " +
                                                fileName + " is not a number"));
        }
      }

      else if (count == 4)
        basisFileNames.push_back(wordTrimmed);

      else {
        AssertThrow(false, dealii::ExcMessage(
                               "Invalid number of column entries found while "
                               "reading atomic coordinates in file " +
                               fileName +
                               ". Number of columns must be 5, i.e, "
                               "atom-symbol x y z gaussian-basis-filename"));
      }

      count++;
    }

    atomicCoords.push_back(coord);
  }

  readFile.close();
}

void readGaussianFiles(
    std::vector<invDFT::gaussianFunctionManager::contractedGaussian *>
        &atomicContractedGaussians,
    const std::string fileName) {
  std::ifstream readFile;
  readFile.open(fileName.c_str());
  AssertThrow(readFile.is_open(),
              dealii::ExcMessage("Could not find file " + fileName));

  std::string readLine;

  // ignore the first line
  std::getline(readFile, readLine);

  while (std::getline(readFile, readLine)) {
    std::istringstream lineString(readLine);
    std::string word;
    unsigned int count = 0;
    while (lineString >> word) {
      if (count >= 2) {
        count++;
        continue;
      }
      if (!word.empty())
        count++;
      //
      // check if it's a valid string
      // i.e., it contains one of the following string:
      // "S", "SP", "SPD", SPDF" ...
      //
      std::size_t pos = word.find_first_not_of("SPDFGHIspdfghi");
      if (pos == std::string::npos) {
        std::string lChars = trim(word);
        const int numLChars = lChars.size();
        std::string strNContracted;
        lineString >> strNContracted;

        if (!strNContracted.empty())
          count++;

        int nContracted;
        if (isInteger(nContracted, strNContracted)) {
          double alpha[nContracted];
          double c[nContracted][numLChars];
          for (unsigned int i = 0; i < nContracted; ++i) {
            if (std::getline(readFile, readLine)) {
              AssertThrow(
                  !readLine.empty(),
                  dealii::ExcMessage("Empty line found in the Gaussian basis "
                                     " file " +
                                     fileName));

              std::istringstream lineContracted(readLine);
              lineContracted >> word;
              AssertThrow(
                  isNumber(alpha[i], word),
                  dealii::ExcMessage("Exponent in the Gaussian basis file " +
                                     fileName + " is not a number"));
              for (unsigned int j = 0; j < numLChars; ++j) {
                lineContracted >> word;
                AssertThrow(isNumber(c[i][j], word),
                            dealii::ExcMessage(
                                "Coefficient in the Gaussian basis file " +
                                fileName + " is not a number"));
              }
            } else {
              std::string msg = "Undefined row for the contracted basis "
                                " in the Gaussian basis file " +
                                fileName;
            }
          }

          for (unsigned int j = 0; j < numLChars; ++j) {
            invDFT::gaussianFunctionManager::contractedGaussian *cg =
                new invDFT::gaussianFunctionManager::contractedGaussian;
            cg->L = nContracted;
            cg->alpha = new double[nContracted];
            cg->c = new double[nContracted];
            cg->lquantum = lChars.at(j);
            for (unsigned int i = 0; i < nContracted; ++i) {
              cg->alpha[i] = alpha[i];
              cg->c[i] = c[i][j];
            }
            atomicContractedGaussians.push_back(cg);
          }
        }

        else {
          std::string msg =
              "In Gaussian basis file " + fileName +
              ", the number of contracted Gaussians is not an integer.";
          AssertThrow(false, dealii::ExcMessage(msg));
        }
      }

      else {
        std::string msg = "Undefined L character(s) for the contracted "
                          " basis read in file " +
                          fileName;
        AssertThrow(false, dealii::ExcMessage(msg));
      }
    }
  }

  readFile.close();
}

void readMatrix(std::vector<std::vector<double>> &densityMat,
                const std::string fileName) {
  std::ifstream readFile;
  readFile.open(fileName.c_str());
  AssertThrow(readFile.is_open(),
              dealii::ExcMessage("Could not find file " + fileName));
  std::string readLine;
  while (getline(readFile, readLine)) {
    std::string msg = "Empty line file found in the density matrix "
                      "file: " +
                      fileName;
    AssertThrow(!readLine.empty(), dealii::ExcMessage(msg));
    std::vector<double> rowVals(0);
    std::istringstream lineString(readLine);
    std::string word;
    while (lineString >> word) {
      std::string msg = "Value read in density matrix is not a number";
      double val;
      AssertThrow(isNumber(val, word), dealii::ExcMessage(msg));
      rowVals.push_back(val);
    }
    densityMat.push_back(rowVals);
  }

  readFile.close();
}

void convertCoordinatesFromAngsToBohr(
    std::vector<std::vector<double>> &coords) {
  const unsigned int N = coords.size();
  const unsigned int dim = coords[0].size();
  for (unsigned int i = 0; i < N; ++i) {
    for (unsigned int j = 0; j < dim; ++j)
      coords[i][j] *= dftfe::C_AngToBohr;
  }
}

void checkAtomicCoords(const std::vector<std::vector<double>> &coords1,
                       const std::vector<std::vector<double>> &coords2,
                       const double tol) {
  const int numAtoms = coords1.size();
  AssertThrow(
      coords1.size() == coords2.size(),
      dealii::ExcMessage("The number atoms provided to the "
                         "gaussianFunctionManager is different from those "
                         "read from the parameters file"));
  for (unsigned int i = 0; i < numAtoms; ++i) {
    double r = 0.0;
    for (unsigned int j = 0; j < 3; ++j)
      r += pow(coords1[i][j] - coords2[i][j], 2.0);
    r = sqrt(r);
    std::string message("Mismatch in coordinates provided to "
                        "gaussianFunctionManager and parameters file");
    AssertThrow(r < tol, dealii::ExcMessage(message));
  }
}

void storeContractedGaussians(
    std::vector<
        std::vector<invDFT::gaussianFunctionManager::contractedGaussian *>>
        &contractedGaussians,
    const std::set<std::string> &uniqueBasisFileNames) {
  const int numUniqueBasisFiles = uniqueBasisFileNames.size();
  contractedGaussians.resize(numUniqueBasisFiles);
  unsigned int i = 0;
  for (std::set<std::string>::const_iterator iter =
           uniqueBasisFileNames.begin();
       iter != uniqueBasisFileNames.end(); iter++) {
    std::vector<invDFT::gaussianFunctionManager::contractedGaussian *>
        &atomicContractedGaussians = contractedGaussians[i];
    std::string fileName = *iter;
    readGaussianFiles(atomicContractedGaussians, fileName);
    i++;
  }
}

void storeBasisFunctions(
    std::vector<invDFT::gaussianFunctionManager::basis *> &basisFunctions,
    const std::vector<
        std::vector<invDFT::gaussianFunctionManager::contractedGaussian *>>
        &contractedGaussians,
    const std::vector<std::vector<double>> &atomicCoords,
    const std::vector<std::string> &basisFileNames,
    const std::set<std::string> &uniqueBasisFileNames) {
  const int numAtoms = basisFileNames.size();
  for (unsigned int iAtom = 0; iAtom < numAtoms; ++iAtom) {
    const std::string basisFileName = basisFileNames[iAtom];
    std::set<std::string>::const_iterator iter =
        uniqueBasisFileNames.find(basisFileName);
    unsigned int index = std::distance(uniqueBasisFileNames.begin(), iter);
    const std::vector<gaussianFunctionManager::contractedGaussian *>
        &atomicContractedGaussians = contractedGaussians[index];
    unsigned int numContractedGaussians = atomicContractedGaussians.size();
    for (unsigned int j = 0; j < numContractedGaussians; ++j) {
      const invDFT::gaussianFunctionManager::contractedGaussian *cg =
          atomicContractedGaussians[j];
      int polyOrder;
      char lquantum = cg->lquantum;

      if (lquantum == 'S' || lquantum == 's')
        polyOrder = 0;

      else if (lquantum == 'P' || lquantum == 'p')
        polyOrder = 1;

      else if (lquantum == 'D' || lquantum == 'd')
        polyOrder = 2;

      else if (lquantum == 'F' || lquantum == 'f')
        polyOrder = 3;

      else if (lquantum == 'G' || lquantum == 'g')
        polyOrder = 4;

      else if (lquantum == 'H' || lquantum == 'h')
        polyOrder = 5;

      else if (lquantum == 'I' || lquantum == 'i')
        polyOrder = 6;

      else {
        std::string message("Invalid L character detected while reading in "
                            "gaussianFunctionManager");
        AssertThrow(false, dealii::ExcMessage(message));
      }

      const int nContracted = cg->L;

      //
      // @note QChem even in the spherical form uses cartesian form for the s
      // and p orbitals. The ordering for cartesian orbitals are lexicographic -
      // i.e., for p orbitals it's ordered as x,y,z. While in the spherical form
      // if one uses -l to +l ordering for the m quantum numbers for l=1 (p
      // orbital), then it translates to ordering the p orbitals as y,z,x. To be
      // consistent with QChem's ordering for p orbital, we do it in an ad-hoc
      // manner by ordering the m quantum numbers as 1,-1,0 for l=1 (p orbital).
      //
      if (polyOrder == 1) {
        int qChem_p[] = {1, -1, 0};
        for (unsigned int iM = 0; iM < 3; ++iM) {
          invDFT::gaussianFunctionManager::basis *b =
              new invDFT::gaussianFunctionManager::basis;
          b->L = nContracted;
          b->alpha = cg->alpha;
          b->c = cg->c;
          b->l = polyOrder;
          b->m = qChem_p[iM];

          b->normConsts = new double[nContracted];
          std::vector<double> normConstsTmp =
              getNormConsts(b->alpha, b->l, nContracted);
          for (unsigned int iContracted = 0; iContracted < nContracted;
               ++iContracted)
            b->normConsts[iContracted] = normConstsTmp[iContracted];

          b->origin = &atomicCoords[iAtom][0];
          // Set the basis normalization factor to 1.0
          b->basisNormConst = 1.0;
          basisFunctions.push_back(b);
        }
      }

      else {
        for (int m = -polyOrder; m <= polyOrder; ++m) {
          invDFT::gaussianFunctionManager::basis *b =
              new invDFT::gaussianFunctionManager::basis;
          b->L = nContracted;
          b->alpha = cg->alpha;
          b->c = cg->c;
          b->l = polyOrder;
          b->m = m;

          b->normConsts = new double[nContracted];
          std::vector<double> normConstsTmp =
              getNormConsts(b->alpha, b->l, nContracted);
          for (unsigned int iContracted = 0; iContracted < nContracted;
               ++iContracted)
            b->normConsts[iContracted] = normConstsTmp[iContracted];

          b->origin = &atomicCoords[iAtom][0];
          // Set the basis normalization factor to 1.0
          b->basisNormConst = 1.0;
          basisFunctions.push_back(b);
        }
      }
    }
  }
}

void getBasisCutoff(
    const std::vector<invDFT::gaussianFunctionManager::basis *> &basisFunctions,
    std::vector<double> &cutoff, const double tol) {
  const unsigned int numBasis = basisFunctions.size();
  cutoff.resize(numBasis, 0.0);
  for (unsigned int i = 0; i < numBasis; ++i) {
    invDFT::gaussianFunctionManager::basis *b = basisFunctions[i];
    int L = b->L;
    double rcut = 0.0;
    for (unsigned int j = 0; j < L; ++j) {
      const double alpha = b->alpha[j];
      const double c = std::fabs(b->c[j]);
      const double norm = b->normConsts[j];
      double r = -(std::log(tol / (c * norm))) / alpha;
      r = std::sqrt(std::abs(r));
      if (r > rcut)
        rcut = r;
    }
    cutoff[i] = rcut;
  }
}

double
getClosestDistanceToBasis(const double *points, const unsigned int nPoints,
                          const invDFT::gaussianFunctionManager::basis *b) {
  double rmin = 1e15;
  const double *R = b->origin;
  for (unsigned int i = 0; i < nPoints; ++i) {
    double r = 0.0;
    for (unsigned j = 0; j < 3; ++j) {
      r += std::pow(points[i * 3 + j] - R[j], 2.0);
    }

    r = std::sqrt(r);
    if (r < rmin)
      rmin = r;
  }

  return rmin;
}

void printData(
    const std::string filename,
    const std::vector<invDFT::gaussianFunctionManager::basis *> &basisFunctions,
    const std::vector<std::vector<double>> &SMat,
    const std::vector<std::vector<double>> &SMatEvaluated,
    const std::vector<std::vector<std::vector<double>>> &densityMats) {
  std::ofstream outfile;
  outfile.open(filename);
  outfile << std::setprecision(16) << std::endl;

  const int numBasis = basisFunctions.size();
  for (unsigned int i = 0; i < numBasis; ++i) {
    outfile << "Info for basis: " << i << std::endl;
    const int L = basisFunctions[i]->L;
    outfile << "(l,m): (" << basisFunctions[i]->l << "," << basisFunctions[i]->m
            << ")\t"
            << "NormConsts: ";
    for (unsigned int j = 0; j < L; ++j) {
      outfile << basisFunctions[i]->normConsts[j] << "\t";
    }

    outfile << "Coeffs: ";
    for (unsigned int j = 0; j < L; ++j) {
      outfile << basisFunctions[i]->c[j] << "\t";
    }

    outfile << std::endl;
  }

  outfile.close();
}

} // namespace

//
// Constructor
//
gaussianFunctionManager::gaussianFunctionManager(
    const std::vector<std::string> densityMatFilenames,
    const std::string atomicCoordsFilename, const char unit,
    const MPI_Comm &mpi_comm_parent, const MPI_Comm &mpi_comm_domain)
    : d_atomicCoords(0), d_basisFileNames(0), d_densityMats(0),
      d_basisFunctions(0), d_contractedGaussians(0),
      d_numSpins(densityMatFilenames.size()),
      pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0)),
      d_mpiComm_domain(mpi_comm_domain), d_mpiComm_parent(mpi_comm_parent) {
  readAtomicCoordsAndBasisFileNames(d_atomicCoords, d_basisFileNames,
                                    atomicCoordsFilename);
  std::vector<std::vector<std::vector<double>>> densityMatsTmp(
      d_numSpins, std::vector<std::vector<double>>(0, std::vector<double>(0)));

  if (d_numSpins == 1) {
    readMatrix(densityMatsTmp[0], densityMatFilenames[0]);
  } else {
    readMatrix(densityMatsTmp[0], densityMatFilenames[0]);
    readMatrix(densityMatsTmp[1], densityMatFilenames[1]);
  }

  // Convert from angstrom to bohr
  const int numAtoms = d_atomicCoords.size();
  if (unit == 'a' || unit == 'A') {
    convertCoordinatesFromAngsToBohr(d_atomicCoords);
  }

  for (unsigned int i = 0; i < d_basisFileNames.size(); ++i)
    d_uniqueBasisFileNames.insert(d_basisFileNames[i]);

  storeContractedGaussians(d_contractedGaussians, d_uniqueBasisFileNames);
  storeBasisFunctions(d_basisFunctions, d_contractedGaussians, d_atomicCoords,
                      d_basisFileNames, d_uniqueBasisFileNames);

  pcout << "Number of Gaussian basis: " << d_basisFunctions.size() << std::endl;
  pcout << "Density Matrix size: " << densityMatsTmp[0].size() << std::endl;
  AssertThrow(
      d_basisFunctions.size() == densityMatsTmp[0].size(),
      dealii::ExcMessage("Mismatch in number of gaussian basis and the size of "
                         " density matrix provided"));

  const int numBasis = d_basisFunctions.size();
  d_densityMats.resize(d_numSpins,
                       std::vector<double>(numBasis * numBasis, 0.0));
  for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin) {
    for (unsigned int i = 0; i < numBasis; ++i) {
      for (unsigned int j = 0; j < numBasis; ++j) {
        d_densityMats[iSpin][i * numBasis + j] = densityMatsTmp[iSpin][i][j];
      }
    }
  }

  d_basisCutoff.resize(numBasis, 0.0);
  getBasisCutoff(d_basisFunctions, d_basisCutoff, DFTFE_GAUSSIAN_ZERO_TOL);

  for (unsigned int iBasis = 0; iBasis < numBasis; iBasis++) {
    pcout << " i = " << iBasis << " cutoff = " << d_basisCutoff[iBasis] << "\n";
  }
}

//
// Destructor
//
gaussianFunctionManager::~gaussianFunctionManager() {
  unsigned int numAtoms = d_contractedGaussians.size();
  for (unsigned int i = 0; i < numAtoms; ++i) {
    std::vector<invDFT::gaussianFunctionManager::contractedGaussian *>
        &atomicContractedGaussians = d_contractedGaussians[i];
    const int numAtomicContractedGaussians = atomicContractedGaussians.size();
    for (unsigned int j = 0; j < numAtomicContractedGaussians; ++j) {
      invDFT::gaussianFunctionManager::contractedGaussian *cg =
          atomicContractedGaussians[j];
      delete cg->alpha;
      delete cg->c;
      delete cg;
    }
  }

  unsigned int numBasis = d_basisFunctions.size();
  for (unsigned int i = 0; i < numBasis; ++i) {
    invDFT::gaussianFunctionManager::basis *b = d_basisFunctions[i];
    delete b->normConsts;
    delete b;
  }
}

void gaussianFunctionManager::evaluateForQuad(
    const double *points, const double *weights, const unsigned int nPoints,
    const bool evalBasis, const bool evalBasisDerivatives,
    const bool evalBasisDoubleDerivatives, const bool evalSMat,
    const bool normalizeBasis, const unsigned int quadIndex,
    std::string smatrixExternalFilename /*= ""*/) {
  d_numQuadPoints[quadIndex] = nPoints;

  d_basisIdsWithCompactSupportInProc[quadIndex] = std::vector<unsigned int>(0);
  std::vector<unsigned int> &basisIdsWithCompactSupportInProc =
      d_basisIdsWithCompactSupportInProc[quadIndex];
  const unsigned int numBasis = d_basisFunctions.size();
  for (unsigned i = 0; i < numBasis; ++i) {
    const double r =
        getClosestDistanceToBasis(points, nPoints, d_basisFunctions[i]);
    if (r < d_basisCutoff[i])
      basisIdsWithCompactSupportInProc.push_back(i);
  }

  const unsigned int numBasisWithCompactSupport =
      basisIdsWithCompactSupportInProc.size();

  if (evalBasis) {
    d_basisVals[quadIndex] =
        std::vector<double>(numBasisWithCompactSupport * nPoints, 0.0);
    std::vector<double> &basisVals = d_basisVals[quadIndex];
    for (unsigned int i = 0; i < numBasisWithCompactSupport; ++i) {
      const int basisId = basisIdsWithCompactSupportInProc[i];
      for (unsigned int j = 0; j < nPoints; ++j) {
        basisVals[i * nPoints + j] =
            getBasisFunctionValue(basisId, &points[j * 3]);
      }
    }
  }

  if (evalBasisDerivatives) {
    d_basisDerivatives[quadIndex] =
        std::vector<double>(numBasisWithCompactSupport * nPoints * 3, 0.0);
    std::vector<double> &basisDerivatives = d_basisDerivatives[quadIndex];
    for (unsigned int i = 0; i < numBasisWithCompactSupport; ++i) {
      const int basisId = basisIdsWithCompactSupportInProc[i];
      for (unsigned int j = 0; j < nPoints; ++j) {
        std::vector<double> grad =
            getBasisFunctionGradient(basisId, &points[j * 3]);
        for (unsigned int k = 0; k < 3; ++k)
          basisDerivatives[i * nPoints * 3 + j * 3 + k] = grad[k];
      }
    }
  }

  if (evalBasisDoubleDerivatives) {
    d_basisDoubleDerivatives[quadIndex] =
        std::vector<double>(numBasisWithCompactSupport * nPoints * 9, 0.0);
    std::vector<double> &basisDoubleDerivatives =
        d_basisDoubleDerivatives[quadIndex];
    for (unsigned int i = 0; i < numBasisWithCompactSupport; ++i) {
      const int basisId = basisIdsWithCompactSupportInProc[i];
      for (unsigned int j = 0; j < nPoints; ++j) {
        std::vector<double> dd =
            getBasisFunctionDoubleDerivatives(basisId, &points[j * 3]);
        for (unsigned int k = 0; k < 3; ++k) {
          for (unsigned int l = 0; l < 3; ++l) {
            basisDoubleDerivatives[i * nPoints * 9 + j * 9 + k * 3 + l] =
                dd[k * 3 + l];
          }
        }
      }
    }
  }

  if (evalSMat) {
    AssertThrow(
        evalBasis,
        dealii::ExcMessage(
            "Cannot evaluate the gaussian "
            "overlap (S) matrix without evaluating the basis value. Set "
            "evalBasis flag to true while calling "
            "gaussianFunctionManager.evaluateForQuad()"));
    d_SMat[quadIndex] = std::vector<double>(numBasis * numBasis, 0.0);
    std::vector<double> &SMat = d_SMat[quadIndex];
    for (unsigned int i = 0; i < numBasisWithCompactSupport; ++i) {
      const int basisIdI = basisIdsWithCompactSupportInProc[i];
      for (unsigned int j = i; j < numBasisWithCompactSupport; ++j) {
        const int basisIdJ = basisIdsWithCompactSupportInProc[j];
        double overlap = 0.0;
        for (unsigned int k = 0; k < nPoints; ++k) {
          overlap += d_basisVals[quadIndex][i * nPoints + k] *
                     d_basisVals[quadIndex][j * nPoints + k] * weights[k];
        }

        SMat[basisIdI * numBasis + basisIdJ] = overlap;
        SMat[basisIdJ * numBasis + basisIdI] = overlap;
      }
    }

    MPI_Allreduce(MPI_IN_PLACE, &SMat[0], numBasis * numBasis, MPI_DOUBLE,
                  MPI_SUM, d_mpiComm_domain);
  }

  if (normalizeBasis) {
    AssertThrow(
        evalBasis && evalSMat,
        dealii::ExcMessage(
            "Cannot evaluate the "
            "normalize the gaussian basis without evaluating the basis value "
            "and the overlap (S) matrix . Set evalBasis and evalSMat flags "
            "to true while calling "
            "gaussianFunctionManager.evaluateForQuad()"));
    for (unsigned int i = 0; i < numBasis; ++i) {
      const double factor = 1.0 / sqrt(d_SMat[quadIndex][i * numBasis + i]);
      d_basisFunctions[i]->basisNormConst = factor;
    }

    // ******

    int rank;
    MPI_Comm_rank(d_mpiComm_domain, &rank);
    if (rank == 0) {
      for (unsigned int i = 0; i < numBasis; ++i) {
        std::cout << " i = " << i
                  << " norm const = " << d_basisFunctions[i]->basisNormConst
                  << "\n";
      }
    }

    ////////

    // scale the S matrix
    for (unsigned int i = 0; i < numBasis; ++i) {
      for (unsigned int j = 0; j < numBasis; ++j) {
        d_SMat[quadIndex][i * numBasis + j] *=
            d_basisFunctions[i]->basisNormConst *
            d_basisFunctions[j]->basisNormConst;
      }
    }

    // scale the basis values
    for (unsigned int i = 0; i < numBasisWithCompactSupport; ++i) {
      const int basisId = basisIdsWithCompactSupportInProc[i];
      const double factor = d_basisFunctions[basisId]->basisNormConst;
      for (unsigned int iPoint = 0; iPoint < nPoints; ++iPoint) {
        d_basisVals[quadIndex][i * nPoints + iPoint] *= factor;
      }
    }

    // scale the basis derivaive valuess
    if (evalBasisDerivatives) {
      for (unsigned int i = 0; i < numBasisWithCompactSupport; ++i) {
        const int basisId = basisIdsWithCompactSupportInProc[i];
        const double factor = d_basisFunctions[basisId]->basisNormConst;
        for (unsigned int iPoint = 0; iPoint < nPoints; ++iPoint) {
          for (unsigned int iDim = 0; iDim < 3; ++iDim) {
            d_basisDerivatives[quadIndex]
                              [i * nPoints * 3 + iPoint * 3 + iDim] *= factor;
          }
        }
      }
    }

    // scale the basis double derivaive valuess
    if (evalBasisDoubleDerivatives) {
      for (unsigned int i = 0; i < numBasisWithCompactSupport; ++i) {
        const int basisId = basisIdsWithCompactSupportInProc[i];
        const double factor = d_basisFunctions[basisId]->basisNormConst;
        for (unsigned int iPoint = 0; iPoint < nPoints; ++iPoint) {
          for (unsigned int iDim = 0; iDim < 3; ++iDim) {
            for (unsigned int jDim = 0; jDim < 3; ++jDim) {
              d_basisDoubleDerivatives[quadIndex][i * nPoints * 9 + iPoint * 9 +
                                                  iDim * 3 + jDim] *= factor;
            }
          }
        }
      }
    }
  }

  if (evalSMat) {
    if (dealii::Utilities::MPI::this_mpi_process(d_mpiComm_parent) == 0) {
      std::ofstream outfile;
      outfile.open("GaussianSMatrixEvaluated");
      outfile << std::setprecision(16) << std::endl;
      for (unsigned int i = 0; i < numBasis; ++i) {
        for (unsigned int j = 0; j < numBasis; ++j) {
          outfile << d_SMat[quadIndex][i * numBasis + j] << " ";
        }
        outfile << std::endl;
      }

      outfile.close();
    }
  }

  if (smatrixExternalFilename != "") {
    std::vector<std::vector<double>> SMatExternal(0);
    readMatrix(SMatExternal, smatrixExternalFilename);
    AssertThrow(SMatExternal.size() == numBasis,
                dealii::ExcMessage(
                    "Mismatch in "
                    "number of basis and the size of the external overlap (S) "
                    "matrix supplied to "
                    "gaussianFunctionManager.evaluateForQuad()"));
    for (unsigned int i = 0; i < numBasis; ++i) {
      for (unsigned int j = 0; j < numBasis; ++j) {
        const double diff =
            std::abs(d_SMat[quadIndex][i * numBasis + j] - SMatExternal[i][j]);
        //                if (diff > DFTFE_SMAT_DIFF_TOL)
        //                  pcout
        //                    << "Mismatch in evaluated and external overlap (S)
        //                    matrix "
        //                       " for pairs "
        //                    << i << ", " << j << ". Evaluated val: "
        //                    << d_SMat[quadIndex][i * numBasis + j]
        //                    << " External val: " << SMatExternal[i][j] <<
        //                    std::endl;
      }
    }

    double rho = 0.0;
    double rhoEvaluated = 0.0;
    for (unsigned int iSpin = 0; iSpin < d_numSpins; ++iSpin) {
      for (unsigned int i = 0; i < numBasis; ++i) {
        for (unsigned int j = 0; j < numBasis; ++j) {
          rhoEvaluated += d_SMat[quadIndex][i * numBasis + j] *
                          d_densityMats[iSpin][i * numBasis + j];
          rho += SMatExternal[i][j] * d_densityMats[iSpin][i * numBasis + j];
        }
      }

      pcout << "Integral Rho Gaussian for iSpin " << iSpin << " : "
            << rhoEvaluated << std::endl;
      pcout << "Integral Rho Gaussian for iSpin " << iSpin
            << " from external overlap (S) matrix: " << rho << std::endl;
    }
  }
}

//
// get density value
//
void gaussianFunctionManager::getRhoValue(const unsigned int quadIndex,
                                          const unsigned int spinIndex,
                                          double *rho) const {
  unsigned int numBasis = d_basisFunctions.size();
  auto it = d_basisVals.find(quadIndex);
  AssertThrow(
      d_basisVals.find(quadIndex) != d_basisVals.end(),
      dealii::ExcMessage("Gaussian basis values are not computed for the given "
                         "quadIndex"));

  auto itBasisIds = d_basisIdsWithCompactSupportInProc.find(quadIndex);
  const std::vector<unsigned int> &basisIdsWithCompactSupportInProc =
      itBasisIds->second;
  const unsigned int numBasisWithCompactSupport =
      basisIdsWithCompactSupportInProc.size();

  auto itNPoints = d_numQuadPoints.find(quadIndex);
  unsigned int nPoints = itNPoints->second;

  const std::vector<double> &basisVals = it->second;
  for (unsigned int iPoint = 0; iPoint < nPoints; ++iPoint) {
    double val = 0.0;
    for (unsigned int i = 0; i < numBasisWithCompactSupport; ++i) {
      unsigned int basisIdI = basisIdsWithCompactSupportInProc[i];
      const double basisValI = basisVals[i * nPoints + iPoint];
      val += d_densityMats[spinIndex][basisIdI * numBasis + basisIdI] *
             basisValI * basisValI;
      for (unsigned int j = i + 1; j < numBasisWithCompactSupport; ++j) {
        unsigned int basisIdJ = basisIdsWithCompactSupportInProc[j];
        const double basisValJ = basisVals[j * nPoints + iPoint];
        val += 2.0 * d_densityMats[spinIndex][basisIdI * numBasis + basisIdJ] *
               basisValI * basisValJ;
      }
    }

    rho[iPoint] = val;
  }
}

//
// get density value
//
void gaussianFunctionManager::getRhoValue(const double *points,
                                          const unsigned int N,
                                          const unsigned int spinIndex,
                                          double *rho) const {
  const int numBasis = d_basisFunctions.size();
  for (unsigned int iPoint = 0; iPoint < N; ++iPoint) {
    const double *point = &points[iPoint * 3];
    double val = 0.0;
    for (unsigned int i = 0; i < numBasis; ++i) {
      val += 2.0 * d_densityMats[spinIndex][i * numBasis + i] *
             evaluateBasisValue(d_basisFunctions[i], point) *
             evaluateBasisValue(d_basisFunctions[i], point);
      for (unsigned int j = i + 1; j < numBasis; ++j) {
        val += 2.0 * d_densityMats[spinIndex][i * numBasis + j] *
               evaluateBasisValue(d_basisFunctions[i], point) *
               evaluateBasisValue(d_basisFunctions[j], point);
      }

      rho[iPoint] = val;
    }
  }
}

//
// get gradient of density
//
void gaussianFunctionManager::getRhoGradient(const double *x,
                                             const int spinIndex,
                                             std::vector<double> &returnValue) {
  std::fill(returnValue.begin(), returnValue.end(), 0.0);
  const int numBasis = d_basisFunctions.size();
  std::vector<double> basisFunctionValues(numBasis, 0.0);
  std::vector<std::vector<double>> basisFunctionGradients(
      numBasis, std::vector<double>(3, 0.0));
  for (unsigned int i = 0; i < numBasis; ++i) {
    basisFunctionValues[i] = getBasisFunctionValue(i, x);
    basisFunctionGradients[i] = getBasisFunctionGradient(i, x);
  }
  for (unsigned int i = 0; i < numBasis; ++i) {
    const double basisFunctionValueI = basisFunctionValues[i];
    const std::vector<double> basisFunctionGradientI =
        basisFunctionGradients[i];
    for (unsigned int j = 0; j < numBasis; ++j) {
      const double basisFunctionValueJ = basisFunctionValues[j];
      const std::vector<double> basisFunctionGradientJ =
          basisFunctionGradients[j];

      for (unsigned int k = 0; k < 3; ++k) {
        const double term1 = basisFunctionValueI * basisFunctionGradientJ[k];
        const double term2 = basisFunctionValueJ * basisFunctionGradientI[k];
        returnValue[k] +=
            d_densityMats[spinIndex][i * numBasis + j] * (term1 + term2);
      }
    }
  }
}
//
//
//  //
//  // get laplacian of density
//  //
//  std::vector<double>
//	gaussianFunctionManager::getRhoDoubleDerivatives(const double * x, const
// int spinIndex)
//	{
//	  std::vector<double> returnValue (9,0.0);
//	  const int numBasis = d_basisFunctions.size();
//	  std::vector<double> basisFunctionValues(numBasis,0.0);
//	  std::vector<std::vector<double> > basisFunctionGradients(numBasis,
// std::vector<double>(3,0.0)); 	  std::vector<std::vector<double> >
// basisFunctionDoubleDerivatives(numBasis, std::vector<double>(9,0.0));
//	  for(unsigned int i = 0; i < numBasis; ++i)
//	  {
//		basisFunctionValues[i] = getBasisFunctionValue(i,x);
//		basisFunctionGradients[i] = getBasisFunctionGradient(i,x);
//		basisFunctionDoubleDerivatives[i] =
// getBasisFunctionDoubleDerivatives(i,x);
//
//	  }
//	  for(unsigned int i = 0; i < numBasis; ++i)
//	  {
//		const double basisFunctionValueI = basisFunctionValues[i];
//		const std::vector<double> basisFunctionGradientI =
//		  basisFunctionGradients[i];
//		const std::vector<double> basisFunctionDoubleDerivativesI =
//		  basisFunctionDoubleDerivatives[i];
//		for(unsigned int j = 0; j < numBasis; ++j)
//		{
//		  const double basisFunctionValueJ = basisFunctionValues[j];
//		  const std::vector<double> basisFunctionGradientJ =
//			basisFunctionGradients[j];
//		  const std::vector<double> basisFunctionDoubleDerivativesJ =
//			basisFunctionDoubleDerivatives[j];
//
//		  for(unsigned int iDim = 0; iDim < 3; ++iDim)
//		  {
//			for(unsigned int jDim = 0; jDim < 3; ++jDim)
//			{
//			  const int logically2DIndex = iDim*3 + jDim;
//			  const double term1 =
// basisFunctionValueI*basisFunctionDoubleDerivativesJ[logically2DIndex];
// const double term2 =
// basisFunctionValueJ*basisFunctionDoubleDerivativesI[logically2DIndex];
// const double term3 =
// basisFunctionGradientI[iDim]*basisFunctionGradientJ[jDim];
// const double term4 =
// basisFunctionGradientI[jDim]*basisFunctionGradientJ[iDim];
//			  returnValue[logically2DIndex] +=
// d_densityMats[spinIndex][i][j]*(term1 + term2 + term3 + term4);
//			}
//		  }
//
//		}
//
//	  }
//
//	  return returnValue;
//	}
//
//  //
//  // get laplacian of density
//  //
//  double gaussianFunctionManager::getRhoLaplacian(const double * x, const
//  int spinIndex)
//  {
//	double returnValue = 0.0;
//	const int numBasis = d_basisFunctions.size();
//	std::vector<double> basisFunctionValues(numBasis,0.0);
//	std::vector<std::vector<double> > basisFunctionGradients(numBasis,
// std::vector<double>(3,0.0)); 	std::vector<double>
// basisFunctionLaplacians(numBasis,0.0); 	for(unsigned int i = 0; i <
// numBasis;
//++i)
//	{
//	  basisFunctionValues[i] = getBasisFunctionValue(i,x);
//	  basisFunctionGradients[i] = getBasisFunctionGradient(i,x);
//	  basisFunctionLaplacians[i] = getBasisFunctionLaplacian(i,x);
//
//	}
//	for(unsigned int i = 0; i < numBasis; ++i)
//	{
//	  const double basisFunctionValueI = basisFunctionValues[i];
//	  const std::vector<double> basisFunctionGradientI =
//		basisFunctionGradients[i];
//	  const double basisFunctionLaplacianI = basisFunctionLaplacians[i];
//	  for(unsigned int j = 0; j < numBasis; ++j)
//	  {
//		const double basisFunctionValueJ = basisFunctionValues[j];
//		const std::vector<double> basisFunctionGradientJ =
//		  basisFunctionGradients[j];
//		const double basisFunctionLaplacianJ =
// basisFunctionLaplacians[j];
//
//		const double term1 =
// basisFunctionValueI*basisFunctionLaplacianJ; 		const double
// term2 = basisFunctionValueJ*basisFunctionLaplacianI; 		const
// double term3 = 2.0*dotProduct(basisFunctionGradientI,
// basisFunctionGradientJ); 		returnValue +=
// d_densityMats[spinIndex][i][j]*(term1 + term2 + term3);
//
//	  }
//
//	}
//
//	return returnValue;
//  }

double gaussianFunctionManager::getBasisFunctionValue(const int basisId,
                                                      const double *x) const {
  return evaluateBasisValue(d_basisFunctions[basisId], x);
}

std::vector<double>
gaussianFunctionManager::getBasisFunctionGradient(const int basisId,
                                                  const double *x) const {
  const invDFT::gaussianFunctionManager::basis *b = d_basisFunctions[basisId];
  const int L = b->L;
  const double *R = b->origin;
  const int l = b->l;
  const int m = b->m;

  std::vector<double> dx(3, 0.0);
  for (unsigned int iCart = 0; iCart < 3; ++iCart)
    dx[iCart] = x[iCart] - R[iCart];

  std::vector<double> returnValue(3, 0.0);
  for (unsigned int i = 0; i < L; ++i) {
    const double alphaVal = b->alpha[i];
    const double cVal = b->c[i];
    const double norm = b->normConsts[i];

    std::vector<double> gradientPrimitiveGaussian =
        getSphericalGaussianGradient(dx, l, m, alphaVal);

    for (unsigned int iCart = 0; iCart < 3; ++iCart)
      returnValue[iCart] += cVal * norm * gradientPrimitiveGaussian[iCart];
  }

  for (unsigned int iCart = 0; iCart < 3; ++iCart)
    returnValue[iCart] *= b->basisNormConst;

  return returnValue;
}

//
// get gradient of the basis
//
std::vector<double> gaussianFunctionManager::getBasisFunctionDoubleDerivatives(
    const int basisId, const double *x) const {
  const int finiteDifferenceOrder = 13;
  const double finiteDifferenceSpacing = DFTFE_FINITE_DIFF_H;
  std::vector<double> returnValue(9, 0.0);
  for (unsigned int iDim = 0; iDim < 3; ++iDim) {
    for (unsigned int jDim = 0; jDim < 3; ++jDim) {
      std::vector<int> indices(2);
      indices[0] = iDim;
      indices[1] = jDim;
      int logically2DIndex = iDim * 3 + jDim;
      returnValue[logically2DIndex] = getBasisHigherOrderDerivativeFD(
          x, d_basisFunctions[basisId], indices, finiteDifferenceOrder,
          finiteDifferenceSpacing);
    }
  }

  return returnValue;
}

double
gaussianFunctionManager::getBasisFunctionLaplacian(const int basisId,
                                                   const double *x) const {
  const invDFT::gaussianFunctionManager::basis *b = d_basisFunctions[basisId];
  const int L = b->L;
  const double *R = b->origin;
  const int l = b->l;
  const int m = b->m;

  std::vector<double> dx(3, 0.0);
  for (unsigned int iCart = 0; iCart < 3; ++iCart)
    dx[iCart] = x[iCart] - R[iCart];

  double returnValue = 0.0;
  for (unsigned int i = 0; i < L; ++i) {
    const double alphaVal = b->alpha[i];
    const double cVal = b->c[i];
    const double norm = b->normConsts[i];

    double laplacianPrimitiveGaussian =
        getSphericalGaussianLaplacian(dx, l, m, alphaVal);

    returnValue += cVal * norm * laplacianPrimitiveGaussian;
  }

  returnValue *= b->basisNormConst;
  return returnValue;
}

const std::vector<double> &
gaussianFunctionManager::getDensityMat(const int spinIndex) const {
  AssertThrow(
      spinIndex < d_numSpins,
      dealii::ExcMessage("Spin index passed to "
                         "gaussianFunctionManager.getDensityMat is invalid. "
                         "Must be less than the number of spins."));
  return d_densityMats[spinIndex];
}

const std::vector<double> &
gaussianFunctionManager::getSMat(const unsigned int quadIndex) const {
  auto it = d_SMat.find(quadIndex);
  AssertThrow(it != d_SMat.end(),
              dealii::ExcMessage(
                  "Gaussian basis overlap (S) matrix is not computed for "
                  "the given quadIndex"));
  return it->second;
}

unsigned int gaussianFunctionManager::getNumberBasisFunctions() const {
  return d_basisFunctions.size();
}

} // namespace invDFT
