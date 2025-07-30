//
// Created by VISHAL SUBRAMANIAN on 7/17/25.
//

#include "SlaterFunctionManager.h"

#include <cassert>
#include <cerrno>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>

#include "dftUtils.h"

#include <boost/math/special_functions/factorials.hpp>
#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>

namespace invDFT {

#define ANGSTROM_TO_BOHR 1.889725989
#define DIST_TOL 1e-8
#define RADIUS_TOL 1e-15
#define POLAR_ANGLE_TOL 1e-12
namespace {

extern "C" {

void dgetrf_(int *m, int *n, double *A, int *lda, int *ipiv, int *info);

void dgetri_(int *n, double *A, int *lda, int *ipiv, double *work, int *lwork,
             int *info);

void dsyevd_(char *jobz, char *uplo, int *n, double *A, int *lda, double *w,
             double *work, int *lwork, int *iwork, int *liwork, int *info);

void dgemv_(char *trans, int *m, int *n, double *alpha, double *A, int *lda,
            double *x, int *incx, double *beta, double *y, int *incy);

void daxpy_(int *n, double *alpha, double *x, int *incx, double *y, int *incy);
}

void callevd(int *n, double *A, int *lda, double *w)

{

  char jobz = 'V';
  char uplo = 'U';
  int lwork = 1 + 6 * (*n) + 2 * (*n) * (*n);
  std::vector<double> work(lwork);
  int liwork = 3 + 5 * (*n);
  std::vector<int> iwork(liwork, 0);
  int info;

  dsyevd_(&jobz, &uplo, n, A, lda, w, &work[0], &lwork, &iwork[0], &liwork,
          &info);
}

void calldgetrf(int m, int n, double *A, int *ipiv)

{

  int info;
  int lda = m;

  dgetrf_(&m, &n, A, &lda, ipiv, &info);

  if (info != 0) {

    std::cout << "Failure info: " << info << std::endl;
    const std::string message("DGETRF failed to converge");

    AssertThrow(
        false,
        dealii::ExcMessage(
            "InvDFT Error: DGETRF in Slater basis eval failed to converge "));
  }
}

void calldgetri(int n, double *A, int *ipiv)

{

  int lda = n;
  int lwork = n * n;
  std::vector<double> work(lwork);

  int info;

  dgetri_(&n, A, &lda, ipiv, &work[0], &lwork, &info);

  if (info != 0) {

    const std::string message("DGETRI failed to converge");

    AssertThrow(
        false,
        dealii::ExcMessage(
            "InvDFT Error: DGETRI in Slater basis eval failed to converge "));
  }
}

void callgemv(char *trans, int *m, int *n, double *alpha, double *A, int *lda,
              double *x, int *incx, double *beta, double *y, int *incy) {

  dgemv_(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

void callaxpy(int *n, double *alpha, double *x, int *incx, double *y,
              int *incy) {
  daxpy_(n, alpha, x, incx, y, incy);
}

double dotProduct(const std::vector<double> &x, const std::vector<double> &y) {

  double returnValue = 0.0;
  for (unsigned int i = 0; i < x.size(); ++i)
    returnValue += x[i] * y[i];

  return returnValue;
}

int factorial(int n) {

  if (n == 0)
    return 1;
  else
    return n * factorial(n - 1);
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

double getDistance(const double *x, const double *y) {
  double r = 0.0;
  for (unsigned int i = 0; i < 3; ++i)
    r += pow(x[i] - y[i], 2.0);
  return sqrt(r);
}

inline double Dm(const int m) {

  if (m == 0)
    return 1.0 / sqrt(2 * M_PI);
  else
    return 1.0 / sqrt(M_PI);
}

inline double Clm(const int l, const int m) {

  assert(m >= 0);
  assert(std::abs(m) <= l);
  // const int modM = std::abs(m);
  return sqrt(((2.0 * l + 1) * boost::math::factorial<double>(l - m)) /
              (2.0 * boost::math::factorial<double>(l + m)));
}

double Qm(const int m, const double phi) {
  if (m > 0)
    return cos(m * phi);
  else if (m == 0)
    return 1.0;
  else // if(m < 0)
    return sin(std::abs(m) * phi);
}

double dQmDPhi(const int m, const double phi) {
  if (m > 0)
    return -m * sin(m * phi);
  else if (m == 0)
    return 0.0;
  else //(m < 0)
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

double slaterRadialPart(const double r, const int n, const double alpha) {

  assert(n > 0);
  if (n == 1)
    return exp(-alpha * r);
  else
    return pow(r, n - 1) * exp(-alpha * r);
}

double slaterRadialPartDerivative(const double r, const double alpha,
                                  const int n, const int derOrder) {
  if (derOrder == 0 && n >= 1)
    return slaterRadialPart(r, n, alpha);
  else if (derOrder == 0 && n < 1)
    return 0.0;
  else
    return (n - 1) * slaterRadialPartDerivative(r, alpha, n - 1, derOrder - 1) -
           alpha * slaterRadialPartDerivative(r, alpha, n, derOrder - 1);
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
    // NOTE: In case theta = 0 or PI, phi is undetermined. The actual value
    // of phi doesn't matter in computing the function value or
    // its gradient. We assign phi = 0.0 here just as a dummy value
    //
    if (fabs(theta - 0.0) >= POLAR_ANGLE_TOL &&
        fabs(theta - M_PI) >= POLAR_ANGLE_TOL)
      phi = atan2(x[1], x[0]);

    else
      phi = 0.0;
  }
}

std::vector<double> getSphericalSlaterGradientAtOrigin(const int n, const int l,
                                                       const int m,
                                                       const double alpha) {
  std::vector<double> returnValue(3);
  const int modM = std::abs(m);
  const double C = Clm(l, modM) * Dm(m);
  if (n == 1) {
    std::string message(
        "Gradient of slater orbital at atomic position is undefined for n=1");
  }

  if (n == 2) {

    if (l == 0) {
      std::string message("Gradient of slater orbital at atomic position is "
                          "undefined for n=2 and l=0");

      AssertThrow(false, dealii::ExcMessage(message));
    }

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

  }

  else {
    returnValue[0] = 0.0;
    returnValue[1] = 0.0;
    returnValue[2] = 0.0;
  }

  return returnValue;
}

std::vector<double> getSphericalSlaterGradientAtPoles(const double r,
                                                      const double theta,
                                                      const int n, const int l,
                                                      const int m,
                                                      const double alpha) {
  const double R = slaterRadialPart(r, n, alpha);
  const double dRDr = slaterRadialPartDerivative(r, alpha, n, 1);
  const int modM = std::abs(m);
  const double C = Clm(l, modM) * Dm(m);
  std::vector<double> returnValue(3);
  if (std::fabs(theta - 0.0) < POLAR_ANGLE_TOL) {
    if (m == 0) {
      returnValue[0] = 0.0;
      returnValue[1] = 0.0;
      returnValue[2] = C * dRDr;

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

  else if (std::fabs(theta - M_PI) < POLAR_ANGLE_TOL) {
    if (m == 0) {
      returnValue[0] = 0.0;
      returnValue[1] = 0.0;
      returnValue[2] = C * dRDr * pow(-1, l + 1);

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
    std::string message(
        "A point that is expected to be lying on the pole is not on the pole.");

    AssertThrow(false, dealii::ExcMessage(message));
  }

  return returnValue;
}

std::vector<double> getSphericalSlaterGradient(const std::vector<double> &x,
                                               const int n, const int l,
                                               const int m,
                                               const double alpha) {
  double r, theta, phi;
  convertCartesianToSpherical(x, r, theta, phi);
  const int modM = std::abs(m);
  const double C = Clm(l, modM) * Dm(m);

  std::vector<double> returnValue(3);
  if (r < RADIUS_TOL) {
    returnValue = getSphericalSlaterGradientAtOrigin(n, l, m, alpha);
  }

  else if (std::fabs(theta - 0.0) < POLAR_ANGLE_TOL ||
           std::fabs(theta - M_PI) < POLAR_ANGLE_TOL) {
    returnValue = getSphericalSlaterGradientAtPoles(r, theta, n, l, m, alpha);
  }

  else {

    const double R = slaterRadialPart(r, n, alpha);
    const double dRDr = slaterRadialPartDerivative(r, alpha, n, 1);
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

double getSphericalSlaterLaplacianAtOrigin(const int n, const int l,
                                           const int m, const double alpha) {

  if (n == 1 || n == 2) {
    std::string message("Laplacian of slater function is undefined at atomic "
                        "position for n=1 and n=2.");

    AssertThrow(false, dealii::ExcMessage(message));

  }

  else if (n == 3) {

    if (l == 0) {
      const int modM = std::abs(m);
      const double C = Clm(l, modM) * Dm(m);
      return 6.0 * C;
    }

    else if (l == 1) {

      std::string message("Laplacian of slater function is undefined at atomic "
                          "position for n=3, l=1.");
      AssertThrow(false, dealii::ExcMessage(message));
      return 0.0;

    }

    else if (l == 2) {
      return 0.0;
    }

    else // l >= 3
      return 0.0;
  }

  else {
    return 0.0;
  }
}

double getSphericalSlaterLaplacianAtPoles(const double r, const double theta,
                                          const int n, const int l, const int m,
                                          const double alpha) {

  double returnValue = 0.0;
  if (m == 0) {

    const int modM = std::abs(m);
    const double C = Clm(l, modM) * Dm(m);
    const double R = slaterRadialPart(r, n, alpha);
    const double dRdr = slaterRadialPartDerivative(r, alpha, n, 1);
    const double d2Rdr2 = slaterRadialPartDerivative(r, alpha, n, 2);
    if (std::fabs(theta - 0.0) < POLAR_ANGLE_TOL) {
      const double term1 = C * (2.0 * dRdr / r + d2Rdr2);
      const double term2 = C * (R / (r * r)) * (-l * (l + 1));
      returnValue = term1 + term2;
    }

    if (std::fabs(theta - M_PI) < POLAR_ANGLE_TOL) {
      const double term1 = C * (2.0 * dRdr / r + d2Rdr2) * pow(-1, l);
      const double term2 = C * (R / (r * r)) * (-l * (l + 1)) * pow(-1, l);
      returnValue = term1 + term2;
    }

  }

  else
    returnValue = 0.0;

  return returnValue;
}

double getSphericalSlaterLaplacian(const std::vector<double> &x, const int n,
                                   const int l, const int m,
                                   const double alpha) {

  double r, theta, phi;
  convertCartesianToSpherical(x, r, theta, phi);
  double returnValue = 0.0;
  if (r < RADIUS_TOL) {
    returnValue = getSphericalSlaterLaplacianAtOrigin(n, l, m, alpha);
  }

  else if (std::fabs(theta - 0.0) < POLAR_ANGLE_TOL ||
           std::fabs(theta - M_PI) < POLAR_ANGLE_TOL) {
    returnValue = getSphericalSlaterLaplacianAtPoles(r, theta, n, l, m, alpha);
  }

  else {
    const int modM = std::abs(m);
    const double C = Clm(l, modM) * Dm(m);
    const double cosTheta = cos(theta);
    const double sinTheta = sin(theta);
    const double R = slaterRadialPart(r, n, alpha);
    const double dRdr = slaterRadialPartDerivative(r, alpha, n, 1);
    const double d2Rdr2 = slaterRadialPartDerivative(r, alpha, n, 2);
    const double P = Plm(l, modM, cosTheta);
    const double Q = Qm(m, phi);
    const double term1 = C * P * Q * (2.0 * dRdr / r + d2Rdr2);
    const double a = dPlmDTheta(l, modM, theta);
    const double b = d2PlmDTheta2(l, modM, theta);
    const double term2 =
        C * (R / (r * r)) * Q * ((cosTheta / sinTheta) * a + b);
    const double term3 =
        -C * m * m * (R / (r * r)) * Q * P / (sinTheta * sinTheta);
    returnValue = term1 + term2 + term3;
  }

  return returnValue;
}

double evaluateBasisValue(const SlaterFunctionManager::basis *b,
                          const double *x) {
  const SlaterFunctionManager::slaterBasis *sb = b->sb;
  const double *x0 = b->origin;
  const double alpha = sb->alpha;
  const int n = sb->n;
  const int l = sb->l;
  const int m = sb->m;
  const double normConst = b->basisNormConst;

  std::vector<double> dx(3);
  for (unsigned int i = 0; i < 3; ++i)
    dx[i] = x[i] - x0[i];

  double r, theta, phi;
  convertCartesianToSpherical(dx, r, theta, phi);

  const double R = slaterRadialPart(r, n, alpha);
  const int modM = std::abs(m);
  const double C = Clm(l, modM) * Dm(m);
  const double cosTheta = cos(theta);
  const double P = Plm(l, modM, cosTheta);
  const double Q = Qm(m, phi);

  const double returnValue = normConst * C * R * P * Q;
  return returnValue;
}

double getBasisHigherOrderDerivativeFD(
    const double *point, const SlaterFunctionManager::basis *basis,
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

double trapezoidal3d(const double L, const double h,
                     std::vector<const SlaterFunctionManager::basis *> funcs) {

  int N = L / h;
  double integral = 0.0;
  int numFuncs = funcs.size();
  for (int i = -N; i <= N; ++i) {
    std::vector<double> point(3);
    point[0] = i * h;
    double f2 = 0.0;
    for (int j = -N; j <= N; ++j) {
      point[1] = j * h;
      double f3 = 0.0;
      for (int k = -N; k <= N; ++k) {
        point[2] = k * h;
        double val = 1.0;
        for (unsigned int l = 0; l < numFuncs; ++l)
          val *= evaluateBasisValue(funcs[l], &point[0]);
        if (k == -N || k == N)
          f3 += 0.5 * h * val;
        else
          f3 += h * val;
      }

      if (j == -N || j == N)
        f2 += 0.5 * h * f3;
      else
        f2 += h * f3;
    }

    if (i == -N || i == N)
      integral += 0.5 * h * f2;
    else
      integral += h * f2;
  }

  return integral;
}

void readAtomicCoordsAndBasisFileNames(
    std::vector<std::vector<double>> &atomicCoords,
    std::vector<std::string> &basisFileNames, std::string fileName) {

  std::ifstream readFile;

  //
  // string to read line
  //
  std::string readLine;

  readFile.open(fileName.c_str());
  assert(readFile.is_open());
  while (std::getline(readFile, readLine)) {
    if (readLine.empty()) {
      std::string message("Empty or invalid line while reading atomic "
                          "coordinates in SlaterFunctionManager");

      AssertThrow(false, dealii::ExcMessage(message));
    }
    std::istringstream lineString(readLine);
    std::string word;
    unsigned int count = 0;
    std::vector<double> coord(0);
    while (lineString >> word) {
      if (count >= 5) {
        std::string message("Invalid column entry while reading atomic "
                            "coordinates in SlaterFunctionManager. "
                            "Number of columns must be 5.");

        AssertThrow(false, dealii::ExcMessage(message));
      }

      std::string wordTrimmed = trim(word);

      if (wordTrimmed.empty()) {
        std::string message("Empty column entry while reading atomic "
                            "coordinates in SlaterFunctionManager");

        AssertThrow(false, dealii::ExcMessage(message));
      }

      if (count >= 1 && count <= 3) {
        double val;
        if (isNumber(val, wordTrimmed))
          coord.push_back(val);
        else {
          std::string message("Coordinate entry in the atomic coordinates in "
                              "SlaterFunctionManager is not a number");

          AssertThrow(false, dealii::ExcMessage(message));
        }
      }

      else if (count == 4)
        basisFileNames.push_back(wordTrimmed);

      count++;
    }

    atomicCoords.push_back(coord);
  }

  assert(atomicCoords.size() == basisFileNames.size());
}

void readSlaterFiles(std::vector<SlaterFunctionManager::slaterBasis *>
                         &atomicSlaterBasisFunctions,
                     const std::string fileName) {

  atomicSlaterBasisFunctions.resize(0);
  std::ifstream readFile;

  //
  // string to read line
  //
  std::string readLine;

  readFile.open(fileName.c_str());
  assert(readFile.is_open());

  //
  // ignore the first line
  //
  std::getline(readFile, readLine);

  while (std::getline(readFile, readLine)) {
    std::istringstream lineString(readLine);
    std::string word;
    unsigned int count = 0;
    while (lineString >> word) {

      if (count == 0) {
        //
        // check if it's a valid string
        // i.e., it contains one of the following string:
        // "1S", "2S", "2P", "3D" ...
        //
        std::size_t pos = word.find_first_not_of("SPDFGHspdfgh0123456789");
        if (pos == std::string::npos) {
          std::string nlChars = trim(word);

          // find the position of the L character, i.e
          // position of "S", "P", "D", ....
          std::size_t lpos = nlChars.find_first_of("SPDFGHspdfgh");
          std::string nchars = nlChars.substr(0, lpos);
          int n;
          if (!isInteger(n, nchars)) {
            std::string message(
                "Undefined behavior in slater file:"
                " The principal quantum number read is not an integer");

            AssertThrow(false, dealii::ExcMessage(message));
          }

          std::string lchars = nlChars.substr(lpos);
          char lchar;
          if (lchars.size() == 1)
            lchar = lchars[0];
          else {
            std::string message("Undefined behavior in slater file: The L "
                                "(angular) quantum character is not proper");

            AssertThrow(false, dealii::ExcMessage(message));
          }

          int l;
          if (lchar == 'S' || lchar == 's')
            l = 0;
          else if (lchar == 'P' || lchar == 'p')
            l = 1;
          else if (lchar == 'D' || lchar == 'd')
            l = 2;
          else if (lchar == 'F' || lchar == 'f')
            l = 3;
          else if (lchar == 'G' || lchar == 'g')
            l = 4;
          else if (lchar == 'H' || lchar == 'h')
            l = 5;
          else {
            std::string message("Undefined behavior in slater file: Invalid L "
                                "quantum character read");

            AssertThrow(false, dealii::ExcMessage(message));
          }

          // read the next word which contains the exponent
          std::string strAlpha;
          lineString >> strAlpha;
          double alpha;
          if (strAlpha.empty()) {
            std::string message("Undefined behavior in slater file: Couldn't "
                                "find the required exponent");

            AssertThrow(false, dealii::ExcMessage(message));
          } else {
            if (!isNumber(alpha, strAlpha)) {
              std::string message("Undefined behavior in slater file: The "
                                  "exponent is not a number");

              AssertThrow(false, dealii::ExcMessage(message));
            }
          }

          //
          // NOTE: QChem even in the spherical form uses cartesian form for the
          // s and p orbitals. The ordering for cartesian orbitals are
          // lexicographic - i.e., for p orbitals it's ordered as x,y,z. While
          // in the spherical form if one uses -l to +l ordering for the m
          // quantum numbers for l=1 (p orbital), then it translates to ordering
          // the p orbitals as y,z,x. To be consistent with QChem's ordering for
          // p orbital, we do it in an ad-hoc manner by ordering the m quantum
          // numbers as 1,-1,0 for l=1 (p orbital).
          //
          if (l == 1) {

            int qChem_p[] = {1, -1, 0};
            for (unsigned int iM = 0; iM < 3; ++iM) {
              SlaterFunctionManager::slaterBasis *sb =
                  new SlaterFunctionManager::slaterBasis;
              sb->alpha = alpha;
              sb->n = n;
              sb->l = l;
              sb->m = qChem_p[iM];
              const double term1 = pow(2.0 * alpha, n + 1.0 / 2.0);
              const double term2 = pow(factorial(2 * n), 1.0 / 2.0);
              sb->normConst = term1 / term2;
              atomicSlaterBasisFunctions.push_back(sb);
            }
          }

          else {
            for (int m = -l; m <= l; ++m) {

              SlaterFunctionManager::slaterBasis *sb =
                  new SlaterFunctionManager::slaterBasis;
              sb->alpha = alpha;
              sb->n = n;
              sb->l = l;
              sb->m = m;
              const double term1 = pow(2.0 * alpha, n + 1.0 / 2.0);
              const double term2 = pow(factorial(2 * n), 1.0 / 2.0);
              sb->normConst = term1 / term2;
              atomicSlaterBasisFunctions.push_back(sb);
            }
          }

        }

        else {

          std::string message("Undefined behavior in slater file: The orbital "
                              "name is not proper");

          AssertThrow(false, dealii::ExcMessage(message));
        }
      }

      count++;
    }
  }

  readFile.close();
}

void readMatrix(std::vector<std::vector<double>> &densityMat,
                const std::string fileName) {
  std::ifstream readFile;

  //
  // string to read line
  //
  std::string readLine;
  readFile.open(fileName.c_str());
  assert(readFile.is_open());
  while (getline(readFile, readLine)) {

    if (!readLine.empty()) {
      std::vector<double> rowVals(0);
      std::istringstream lineString(readLine);
      std::string word;
      while (lineString >> word) {
        double val;
        if (isNumber(val, word))
          rowVals.push_back(val);
        else
          std::cout << "Undefined behavior: Value read in density matrix is "
                       "not a number"
                    << std::endl;
      }
      densityMat.push_back(rowVals);
    } else
      std::cout << "Empty line found in density matrix file: " << fileName
                << std::endl;
  }

  readFile.close();
}

namespace slaterFunctionManager {

// Interpolate field at quad points
//
/*
    QuadratureValuesContainer<DoubleVector>
interpolateFieldPower(const QuadratureValuesContainer<DoubleVector> & f,
                      const int numberComponents,
                      const ArrayNameManager::NameId fieldId,
                      const int exponent,
                      const int                 meshId)
{

    //
    // get mesh manager
    //
    MeshManager & meshManager = MeshManagerSingleton::getInstance();

    //
    // get QuadratureRuleManager
    //
    QuadratureRuleManager & quadratureRuleManager =
QuadratureRuleManagerSingleton::getInstance();

    //
    // get handle to FieldQuadratureTypeManager
    //
    FieldQuadratureTypeManager & fieldQuadratureTypeManager =
FieldQuadratureTypeManagerSingleton::getInstance();

    //
    // Get the quadratureType for the fieldId
    //

    QuadratureRuleManager::QuadratureNameId quadratureType =
fieldQuadratureTypeManager.getFieldQuadratureType(fieldId);

    //
    // get handle to Adaptive quadrature rule container
    //
    const QuadratureRuleContainer & quadratureRuleContainer =
quadratureRuleManager.getQuadratureRuleContainer(quadratureType);

    //
    // get the number of elements in the mesh
    //
    const int numberElements = meshManager.getNumberElements(meshId);

    //
    // instantiate return value
    // FIXME: quadrature id used is 0
    QuadratureValuesContainer<DoubleVector> returnValue(meshId,
                                                        0,
                                                        quadratureType,
                                                        numberComponents,
                                                        0.0);
    //
    // iterate over elements
    //
    for (vtkIdType iElem = 0; iElem < numberElements; ++iElem)
    {

        //
        // get handle to the quadrature rule for the element
        //
        const QuadratureRule & quadratureRule =
                quadratureRuleContainer.getElementQuadratureRule(iElem,
                                                                 0,
                                                                 meshId);

        //
        // get the number of quad points
        //
        const QuadratureRule::quad_point_size_type numberQuadraturePoints =
                quadratureRule.getNumberPoints();

        const DoubleVector & elementQuadValuesCurrent =
returnValue.getElementValues(iElem);

        //
        // get the current quad values for the element
        //
        const DoubleVector & fQuadValues = f.getElementValues(iElem);

        //
        // copy current element quad values to a temporary storage
        //
        DoubleVector elementQuadValues = elementQuadValuesCurrent;

        //
        // compute value and gradient at quad points
        //
        for (int iQuadPoint = 0;
             iQuadPoint < numberQuadraturePoints;
             ++iQuadPoint) {

            //
            // iterate over components of field
            //
            for(int j = 0; j < numberComponents; ++j) {

                double fVal = fQuadValues[iQuadPoint*numberComponents + j];
                elementQuadValues[iQuadPoint*numberComponents + j] =
pow(fVal,exponent);

            }

        }

        //
        // set element quad values
        //
        returnValue.setElementValues(iElem, elementQuadValues);

    }

    //
    //
    //
    return returnValue;

}
*/

// Interpolate field at quad points
//
/*
    QuadratureValuesContainer<DoubleVector>
interpolateFieldProduct(const QuadratureValuesContainer<DoubleVector> & f,
                        const QuadratureValuesContainer<DoubleVector> & g,
                        const int numberComponents,
                        const ArrayNameManager::NameId fieldId,
                        const int                 meshId)
{

    //
    // get mesh manager
    //
    MeshManager & meshManager = MeshManagerSingleton::getInstance();

    //
    // get QuadratureRuleManager
    //
    QuadratureRuleManager & quadratureRuleManager =
QuadratureRuleManagerSingleton::getInstance();

    //
    // get handle to FieldQuadratureTypeManager
    //
    FieldQuadratureTypeManager & fieldQuadratureTypeManager =
FieldQuadratureTypeManagerSingleton::getInstance();

    //
    // Get the quadratureType for the fieldId
    //

    QuadratureRuleManager::QuadratureNameId quadratureType =
fieldQuadratureTypeManager.getFieldQuadratureType(fieldId);

    //
    // get handle to Adaptive quadrature rule container
    //
    const QuadratureRuleContainer & quadratureRuleContainer =
quadratureRuleManager.getQuadratureRuleContainer(quadratureType);

    //
    // get the number of elements in the mesh
    //
    const int numberElements = meshManager.getNumberElements(meshId);

    //
    // instantiate return value
    // FIXME: quadrature id used is 0
    QuadratureValuesContainer<DoubleVector> returnValue(meshId,
                                                        0,
                                                        quadratureType,
                                                        numberComponents,
                                                        0.0);
    //
    // iterate over elements
    //
    for (vtkIdType iElem = 0; iElem < numberElements; ++iElem)
    {

        //
        // get handle to the quadrature rule for the element
        //
        const QuadratureRule & quadratureRule =
                quadratureRuleContainer.getElementQuadratureRule(iElem,
                                                                 0,
                                                                 meshId);

        //
        // get the number of quad points
        //
        const QuadratureRule::quad_point_size_type numberQuadraturePoints =
                quadratureRule.getNumberPoints();

        const DoubleVector & elementQuadValuesCurrent =
returnValue.getElementValues(iElem);

        //
        // get the current quad values for the element
        //
        const DoubleVector & fQuadValues = f.getElementValues(iElem);

        //
        // get the current quad values for the element
        //
        const DoubleVector & gQuadValues = g.getElementValues(iElem);

        //
        // copy current element quad values to a temporary storage
        //
        DoubleVector elementQuadValues = elementQuadValuesCurrent;

        //
        // compute value and gradient at quad points
        //
        for (int iQuadPoint = 0;
             iQuadPoint < numberQuadraturePoints;
             ++iQuadPoint) {

            //
            // iterate over components of field
            //
            for(int j = 0; j < numberComponents; ++j) {

                double fVal = fQuadValues[iQuadPoint*numberComponents + j];
                double gVal = gQuadValues[iQuadPoint*numberComponents + j];
                elementQuadValues[iQuadPoint*numberComponents + j] = fVal*gVal;

            }
        }

        //
        // set element quad values
        //
        returnValue.setElementValues(iElem, elementQuadValues);

    }

    //
    //
    //
    return returnValue;
}
*/

} // namespace slaterFunctionManager

} // namespace

//
// default Constructor
//
SlaterFunctionManager::SlaterFunctionManager(
    const std::string densityMatFilename, const std::string smatrixFilename,
    const std::string atomicCoordsFilename, std::vector<double> quadCoordinates,
    std::vector<double> quadJxW, unsigned int numQuadPoints,
    const MPI_Comm &mpi_comm_parent, const MPI_Comm &mpi_comm_domain)
    : d_atomicCoords(0), d_basisFileNames(0), d_densityMat(0), d_SMat(0),
      d_slaterBasisFunctions(0), d_basisFunctions(0),
      d_mpiComm_domain(mpi_comm_domain), d_mpiComm_parent(mpi_comm_parent) {

  //
  // turn off output if not root task
  //
  const unsigned int rootTaskId = 0;
  const unsigned int taskId =
      dealii::Utilities::MPI::this_mpi_process(d_mpiComm_parent);

  readAtomicCoordsAndBasisFileNames(d_atomicCoords, d_basisFileNames,
                                    atomicCoordsFilename);

  //
  // Convert from angstrom to bohr
  //
  const int numAtoms = d_atomicCoords.size();
  for (unsigned int i = 0; i < numAtoms; ++i) {
    for (unsigned int j = 0; j < d_atomicCoords[i].size(); ++j)
      d_atomicCoords[i][j] *= ANGSTROM_TO_BOHR;
  }

  // NuclearPositionsReader & nuclearPositionsReader =
  //        dft::NuclearPositionsReaderSingleton::getInstance();

  // Get the number of charges present in the system
  // FIXME: meshId passed to getTotalNumberCharges is 0
  //
  // unsigned int totalNumberCharges  =
  // nuclearPositionsReader.getTotalNumberCharges(0);
  for (unsigned int i = 0; i < d_basisFileNames.size(); ++i)
    d_uniqueBasisFileNames.insert(d_basisFileNames[i]);

  const int numUniqueBasisFiles = d_uniqueBasisFileNames.size();
  d_slaterBasisFunctions.resize(numUniqueBasisFiles);
  unsigned int i = 0;
  for (std::set<std::string>::const_iterator iter =
           d_uniqueBasisFileNames.begin();
       iter != d_uniqueBasisFileNames.end(); iter++) {
    std::vector<SlaterFunctionManager::slaterBasis *>
        &atomicSlaterBasisFunctions = d_slaterBasisFunctions[i];
    std::string fileName = *iter;
    readSlaterFiles(atomicSlaterBasisFunctions, fileName);
    i++;
  }

  for (unsigned int iAtom = 0; iAtom < d_atomicCoords.size(); ++iAtom) {
    const std::string basisFileName = d_basisFileNames[iAtom];
    std::set<std::string>::const_iterator iter =
        d_uniqueBasisFileNames.find(basisFileName);
    unsigned int index = std::distance(d_uniqueBasisFileNames.begin(), iter);
    std::vector<SlaterFunctionManager::slaterBasis *>
        &atomicSlaterBasisFunctions = d_slaterBasisFunctions[index];
    unsigned int numAtomicSlaters = atomicSlaterBasisFunctions.size();
    for (unsigned int j = 0; j < numAtomicSlaters; ++j) {
      SlaterFunctionManager::basis *b = new SlaterFunctionManager::basis;
      b->sb = atomicSlaterBasisFunctions[j];
      b->origin = &d_atomicCoords[iAtom][0];
      b->basisNormConst = atomicSlaterBasisFunctions[j]->normConst;
      d_basisFunctions.push_back(b);
    }
  }

  int numBasis = d_basisFunctions.size();

  d_densityMat.resize(0);
  readMatrix(d_densityMat, densityMatFilename);

  d_SMat.resize(0);
  readMatrix(d_SMat, smatrixFilename);

  std::vector<std::vector<double>> SMatEvaluated =
      this->getEvaluatedSMat(quadCoordinates, quadJxW, numQuadPoints);

  // Renormalize the basis
  // for(unsigned int i = 0; i < numBasis; ++i)
  //{
  //	d_basisFunctions[i]->basisNormConst = 1.0/sqrt(SMatEvaluated[i][i]);
  //}

  std::ofstream outfile;
  outfile.open("SlaterTestData");

  double rhoEvaluated = 0.0;
  double rho = 0.0;
  for (unsigned int i = 0; i < numBasis; ++i) {
    for (unsigned int j = 0; j < numBasis; ++j) {
      rho += d_SMat[i][j] * d_densityMat[i][j];
      rhoEvaluated += SMatEvaluated[i][j] * d_densityMat[i][j];
    }
  }

  d_SMatInvFlattened.resize(numBasis * numBasis, 0.0);
  for (unsigned int i = 0; i < numBasis; ++i) {
    for (unsigned int j = 0; j < numBasis; ++j) {
      d_SMatInvFlattened[i * numBasis + j] = d_SMat[i][j];
    }
  }

  std::vector<int> ipiv(numBasis);
  calldgetrf(numBasis, numBasis, &d_SMatInvFlattened[0], &ipiv[0]);

  calldgetri(numBasis, &d_SMatInvFlattened[0], &ipiv[0]);

  if (taskId == rootTaskId) {
    std::cout << "Number Slater basis: " << numBasis << std::endl;
    std::cout << "Integral Rho evaluated: " << rhoEvaluated << std::endl;
    std::cout << "Integral Rho : " << rho << std::endl;

    for (unsigned int i = 0; i < numBasis; ++i) {
      outfile << "\nInfo for basis: " << i << std::endl;
      const SlaterFunctionManager::slaterBasis *sb = d_basisFunctions[i]->sb;
      outfile << "Alpha: " << sb->alpha << std::endl;
      outfile << "Quantum numbers: " << sb->n << "\t" << sb->l << "\t" << sb->m
              << std::endl;
      outfile << "NormConst: " << sb->normConst << std::endl;
      outfile << "Origin: " << d_basisFunctions[i]->origin[0] << "\t"
              << d_basisFunctions[i]->origin[1] << "\t"
              << d_basisFunctions[i]->origin[2] << std::endl;
    }

    outfile << "Printing SMatEvaluated" << std::endl;
    for (unsigned int i = 0; i < numBasis; ++i) {
      for (unsigned int j = 0; j < numBasis; ++j)
        outfile << SMatEvaluated[i][j] << "\t";
      outfile << std::endl;
    }

    outfile << "Printing SMat Given" << std::endl;
    for (unsigned int i = 0; i < numBasis; ++i) {
      for (unsigned int j = 0; j < numBasis; ++j) {
        outfile << d_SMat[i][j] << "\t";
      }
      outfile << std::endl;
    }

    outfile << "\nPrinting DensityMat" << std::endl;
    for (unsigned int i = 0; i < numBasis; ++i) {
      for (unsigned int j = 0; j < numBasis; ++j)
        outfile << d_densityMat[i][j] << "\t";
      outfile << std::endl;
    }

    outfile << "\n Printing SMat mismatches" << std::endl;
    for (unsigned int i = 0; i < numBasis; ++i) {
      for (unsigned int j = 0; j < numBasis; ++j) {
        if (std::fabs(d_SMat[i][j] - SMatEvaluated[i][j]) > 1e-4) {
          outfile << "(" << i << "," << j << ")"
                  << "(" << d_SMat[i][j] << "," << SMatEvaluated[i][j] << ")\t";
        }
      }
      outfile << std::endl;
    }

    outfile << "\n Printing SMat Inverse" << std::endl;
    for (unsigned int i = 0; i < numBasis; ++i) {
      for (unsigned int j = 0; j < numBasis; ++j) {
        outfile << d_SMatInvFlattened[i * numBasis + j] << " ";
      }
      outfile << std::endl;
    }
  }

  outfile.close();
}

//
// Destructor
//
SlaterFunctionManager::~SlaterFunctionManager() {
  unsigned int numAtoms = d_slaterBasisFunctions.size();
  for (unsigned int i = 0; i < numAtoms; ++i) {
    std::vector<SlaterFunctionManager::slaterBasis *>
        &atomicSlaterBasisFunctions = d_slaterBasisFunctions[i];
    const int numAtomicSlaters = atomicSlaterBasisFunctions.size();
    for (unsigned int j = 0; j < numAtomicSlaters; ++j) {
      SlaterFunctionManager::slaterBasis *sb = atomicSlaterBasisFunctions[j];
      delete sb;
    }
  }
}

//
// get density value
//
double SlaterFunctionManager::getRhoValue(const double *x) {
  const int numBasis = d_basisFunctions.size();
  double val = 0.0;
  for (unsigned int i = 0; i < numBasis; ++i) {
    for (unsigned int j = 0; j < numBasis; ++j)
      val += d_densityMat[i][j] * evaluateBasisValue(d_basisFunctions[i], x) *
             evaluateBasisValue(d_basisFunctions[j], x);
  }

  return val;
}

std::vector<double> SlaterFunctionManager::getRhoGradient(const double *x) {
  std::vector<double> returnValue(3, 0.0);
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
        returnValue[k] += d_densityMat[i][j] * (term1 + term2);
      }
    }
  }

  return returnValue;
}

double SlaterFunctionManager::getRhoLaplacian(const double *x) {
  double returnValue = 0.0;
  const int numBasis = d_basisFunctions.size();
  std::vector<double> basisFunctionValues(numBasis, 0.0);
  std::vector<std::vector<double>> basisFunctionGradients(
      numBasis, std::vector<double>(3, 0.0));
  std::vector<double> basisFunctionLaplacians(numBasis, 0.0);
  for (unsigned int i = 0; i < numBasis; ++i) {
    basisFunctionValues[i] = getBasisFunctionValue(i, x);
    basisFunctionGradients[i] = getBasisFunctionGradient(i, x);
    basisFunctionLaplacians[i] = getBasisFunctionLaplacian(i, x);
  }
  for (unsigned int i = 0; i < numBasis; ++i) {
    const double basisFunctionValueI = basisFunctionValues[i];
    const std::vector<double> basisFunctionGradientI =
        basisFunctionGradients[i];
    const double basisFunctionLaplacianI = basisFunctionLaplacians[i];
    for (unsigned int j = 0; j < numBasis; ++j) {
      const double basisFunctionValueJ = basisFunctionValues[j];
      const std::vector<double> basisFunctionGradientJ =
          basisFunctionGradients[j];
      const double basisFunctionLaplacianJ = basisFunctionLaplacians[j];

      const double term1 = basisFunctionValueI * basisFunctionLaplacianJ;
      const double term2 = basisFunctionValueJ * basisFunctionLaplacianI;
      const double term3 =
          2.0 * dotProduct(basisFunctionGradientI, basisFunctionGradientJ);
      returnValue += d_densityMat[i][j] * (term1 + term2 + term3);
    }
  }

  return returnValue;
}

//
// get density value
//
double SlaterFunctionManager::getBasisFunctionValue(const int basisId,
                                                    const double *x) {
  return evaluateBasisValue(d_basisFunctions[basisId], x);
}

std::vector<double>
SlaterFunctionManager::getBasisFunctionGradient(const int basisId,
                                                const double *x) {

  const SlaterFunctionManager::basis *b = d_basisFunctions[basisId];
  const SlaterFunctionManager::slaterBasis *sb = b->sb;
  const double *x0 = b->origin;
  const double alpha = sb->alpha;
  const int n = sb->n;
  const int l = sb->l;
  const int m = sb->m;
  const double normConst = b->basisNormConst;

  std::vector<double> dx(3);
  for (unsigned int i = 0; i < 3; ++i)
    dx[i] = x[i] - x0[i];

  std::vector<double> returnValue =
      getSphericalSlaterGradient(dx, n, l, m, alpha);

  for (unsigned int iCart = 0; iCart < 3; ++iCart)
    returnValue[iCart] *= normConst;

  return returnValue;
}

double SlaterFunctionManager::getBasisFunctionLaplacian(const int basisId,
                                                        const double *x) {
  const SlaterFunctionManager::basis *b = d_basisFunctions[basisId];
  const SlaterFunctionManager::slaterBasis *sb = b->sb;
  const double *x0 = b->origin;
  const double alpha = sb->alpha;
  const int n = sb->n;
  const int l = sb->l;
  const int m = sb->m;
  const double normConst = b->basisNormConst;

  std::vector<double> dx(3);
  for (unsigned int i = 0; i < 3; ++i)
    dx[i] = x[i] - x0[i];

  double returnValue = getSphericalSlaterLaplacian(dx, n, l, m, alpha);
  returnValue *= normConst;

  return returnValue;
}
//
// get number of basis functions
//
int SlaterFunctionManager::getNumberBasisFunctions() {
  return d_basisFunctions.size();
}

std::vector<std::vector<double>>
SlaterFunctionManager::getEvaluatedSMat(std::vector<double> quadCoordinates,
                                        std::vector<double> quadJxW,
                                        unsigned int numQuadPoints) {

  const int numBasis = d_basisFunctions.size();
  std::vector<std::vector<double>> SMat(numBasis,
                                        std::vector<double>(numBasis));
  for (unsigned int i = 0; i < numBasis; ++i) {
    std::fill(SMat[i].begin(), SMat[i].end(), 0.0);
    for (unsigned int j = 0; j < numBasis; ++j) {
      for (unsigned int iQuad = 0; iQuad < numQuadPoints; iQuad++) {

        SMat[i][j] += evaluateBasisValue(d_basisFunctions[i],
                                         &quadCoordinates[3 * iQuad]) *
                      evaluateBasisValue(d_basisFunctions[j],
                                         &quadCoordinates[3 * iQuad]) *
                      quadJxW[iQuad];
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, &SMat[i][0], numBasis, MPI_DOUBLE, MPI_SUM,
                  d_mpiComm_domain);
  }

  return SMat;
}

} // namespace invDFT

//    std::vector<double>
//    SlaterFunctionManager::getProjectedMO(QuadratureValuesContainer<DoubleVector>
//    moIn,
//                                          const int meshId)
//    {
//#if defined(HAVE_MPI)
//
//        //
//	  // turn off output if not root task
//	  //
//	  const Utils::MPIController & mpiController =
//		Utils::MPIControllerSingleton::getInstance();
//	  const Utils::MPIController::mpi_task_id_type rootTaskId =
//		mpiController.getRootId();
//	  const Utils::MPIController::mpi_task_id_type taskId =
//		mpiController.getId();
//
//#endif // HAVE_MPI
//
//        int numBasis = d_basisFunctions.size();
//        std::vector<double> moCoeffs(numBasis, 0.0);
//
//        std::vector<double> moSlaterIntegral(numBasis,0.0);
//        for(unsigned int i = 0; i < numBasis; ++i)
//        {
//            QuadratureValuesContainer<DoubleVector> moSlaterProduct =
//                    this->computeMOSlaterFunction(moIn,
//                                                  i,
//                                                  meshId);
//
//            moSlaterIntegral[i] =
//                    dft::FieldIntegrator().integrate(moSlaterProduct,meshId);
//
//        }
//
//        char transA = 'N';
//        double scalar1 = 1.0;
//        double scalar2 = 0.0;
//        int inc = 1;
//        callgemv(&transA,
//                 &numBasis,
//                 &numBasis,
//                 &scalar1,
//                 &d_SMatInvFlattened[0],
//                 &numBasis,
//                 &moSlaterIntegral[0],
//                 &inc,
//                 &scalar2,
//                 &moCoeffs[0],
//                 &inc);
//
//        QuadratureValuesContainer<DoubleVector>  moOut =
//                this->computeMOFromDensityCoeffs(moCoeffs,meshId);
//        QuadratureValuesContainer<DoubleVector>  diff = moIn - moOut;
//
//        QuadratureValuesContainer<DoubleVector>  diffL2 =
//                dft::slaterFunctionManager::interpolateFieldPower(diff,
//                                                                  1,
//                                                                  ArrayNameManager::PSI,
//                                                                  2,
//                                                                  meshId);
//
//        double diffL2Norm = dft::FieldIntegrator().integrate(diffL2,meshId);
//
//        if(taskId == rootTaskId)
//        {
//            std::cout << "Diff Slater projection: " << diffL2Norm <<
//            std::endl;
//        }
//
//        return moCoeffs;
//    }

//    QuadratureValuesContainer<DoubleVector>
//    SlaterFunctionManager::computeMOSlaterFunction(const
//    QuadratureValuesContainer<DoubleVector> & mo,
//                                                   const int slaterFunctionId,
//                                                   const int meshId)
//    {
//        //
//        // get mesh manager
//        //
//        MeshManager & meshManager = MeshManagerSingleton::getInstance();
//
//        //
//        // get QuadratureRuleManager
//        //
//        QuadratureRuleManager & quadratureRuleManager =
//        QuadratureRuleManagerSingleton::getInstance();
//
//        //
//        // get handle to FieldQuadratureTypeManager
//        //
//        FieldQuadratureTypeManager & fieldQuadratureTypeManager =
//        FieldQuadratureTypeManagerSingleton::getInstance();
//
//
//        //
//        // Get the quadratureType for the fieldId
//        //
//
//        QuadratureRuleManager::QuadratureNameId quadratureType =
//        fieldQuadratureTypeManager.getFieldQuadratureType(dft::ArrayNameManager::PSI);
//
//        //
//        // get handle to Adaptive quadrature rule container
//        //
//        const QuadratureRuleContainer & quadratureRuleContainer =
//        quadratureRuleManager.getQuadratureRuleContainer(quadratureType);
//
//        //
//        // get the number of elements in the mesh
//        //
//        const int numberElements = meshManager.getNumberElements(meshId);
//
//        //
//        // instantiate return value by getting the QuadratureValuesContainer
//        associated with quadratureValuesManager
//        // FIXME: quadrature id used is 0
//        QuadratureValuesContainer<DoubleVector> quadValues(meshId,
//                                                           0,
//                                                           quadratureType,
//                                                           1,
//                                                           //numberComponents
//                                                           0.0);
//
//
//        //
//        // iterate over elements
//        //
//        for (vtkIdType iElem = 0; iElem < numberElements; ++iElem) {
//
//
//            //
//            // get handle to the quadrature rule for the element
//            //
//            const QuadratureRule & quadratureRule =
//                    quadratureRuleContainer.getElementQuadratureRule(iElem,
//                                                                     0,
//                                                                     meshId);
//
//            //
//            // get the number of quad points
//            //
//            const QuadratureRule::quad_point_size_type numberQuadraturePoints
//            =
//                    quadratureRule.getNumberPoints();
//
//
//            //
//            // get the current quad values for the element
//            //
//            const DoubleVector & elementQuadValuesCurrent =
//            quadValues.getElementValues(iElem); const DoubleVector &
//            elementMOQuadValues = mo.getElementValues(iElem);
//
//            //
//            // copy current element quad values to a temporary storage
//            //
//            DoubleVector elementQuadValues = elementQuadValuesCurrent;
//
//            //
//            // compute value and gradient at quad points
//            //
//            for (int iQuadPoint = 0;
//                 iQuadPoint < numberQuadraturePoints;
//                 ++iQuadPoint) {
//
//                //
//                // get quad point coordinates
//                //
//                const Point & quadraturePoint =
//                        quadratureRule.getGlobalPoint(iQuadPoint);
//
//                elementQuadValues[iQuadPoint] =
//                elementMOQuadValues[iQuadPoint]*
//                                                evaluateBasisValue(d_basisFunctions[slaterFunctionId],&quadraturePoint[0]);
//
//            }
//
//            quadValues.setElementValues(iElem, elementQuadValues);
//
//        }
//
//        return quadValues;
//
//
//    }

//    QuadratureValuesContainer<DoubleVector>
//    SlaterFunctionManager::computeMOFromDensityCoeffs(const
//    std::vector<double> & densityCoeffs,
//                                                      const int meshId)
//    {
//
//        //
//        // get mesh manager
//        //
//        MeshManager & meshManager = MeshManagerSingleton::getInstance();
//
//        //
//        // get QuadratureRuleManager
//        //
//        QuadratureRuleManager & quadratureRuleManager =
//        QuadratureRuleManagerSingleton::getInstance();
//
//        //
//        // get handle to FieldQuadratureTypeManager
//        //
//        FieldQuadratureTypeManager & fieldQuadratureTypeManager =
//        FieldQuadratureTypeManagerSingleton::getInstance();
//
//
//        //
//        // Get the quadratureType for the fieldId
//        //
//
//        QuadratureRuleManager::QuadratureNameId quadratureType =
//        fieldQuadratureTypeManager.getFieldQuadratureType(dft::ArrayNameManager::PSI);
//
//        //
//        // get handle to Adaptive quadrature rule container
//        //
//        const QuadratureRuleContainer & quadratureRuleContainer =
//        quadratureRuleManager.getQuadratureRuleContainer(quadratureType);
//
//        //
//        // get the number of elements in the mesh
//        //
//        const int numberElements = meshManager.getNumberElements(meshId);
//
//        //
//        // instantiate return value by getting the QuadratureValuesContainer
//        associated with quadratureValuesManager
//        // FIXME: quadrature id used is 0
//        QuadratureValuesContainer<DoubleVector> returnValue(meshId,
//                                                            0,
//                                                            quadratureType,
//                                                            1,
//                                                            //numberComponents
//                                                            0.0);
//
//        const int numBasis = d_basisFunctions.size();
//
//
//        //
//        // iterate over elements
//        //
//        for (vtkIdType iElem = 0; iElem < numberElements; ++iElem) {
//
//            //
//            // get handle to the quadrature rule for the element
//            //
//            const QuadratureRule & quadratureRule =
//                    quadratureRuleContainer.getElementQuadratureRule(iElem,
//                                                                     0,
//                                                                     meshId);
//
//            //
//            // get the number of quad points
//            //
//            const QuadratureRule::quad_point_size_type numberQuadraturePoints
//            =
//                    quadratureRule.getNumberPoints();
//
//            //
//            // copy current element quad values to a temporary storage
//            //
//            DoubleVector elementQuadValues(numberQuadraturePoints, 0.0);
//
//            //
//            // compute value and gradient at quad points
//            //
//            for (int iQuadPoint = 0;
//                 iQuadPoint < numberQuadraturePoints;
//                 ++iQuadPoint)
//            {
//
//                for(unsigned int i = 0; i < numBasis; ++i)
//                {
//
//                    //
//                    // get quad point coordinates
//                    //
//                    const Point & quadraturePoint =
//                            quadratureRule.getGlobalPoint(iQuadPoint);
//
//                    elementQuadValues[iQuadPoint] += densityCoeffs[i]*
//                                                     evaluateBasisValue(d_basisFunctions[i],
//                                                     &quadraturePoint[0]);
//
//                }
//            }
//
//            returnValue.setElementValues(iElem, elementQuadValues);
//        }
//
//        return returnValue;
//    }
