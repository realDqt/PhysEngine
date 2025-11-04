#pragma once
#include "common.h"
#include "matrix_interface.h"

namespace linear_math {

#if defined(EIGEN_BACKEND)
using Matrix3 = Matrix<Real, 3, 3>;
using Matrix4 = Matrix<Real, 4, 4>;
#elif defined(ENOKI_BACKEND)
using Matrix3 = Matrix<Real, 3>;
using Matrix4 = Matrix<Real, 4>;
#endif
// using MatrixX = Matrix<Real, -1, -1>;

}  // namespace linear_math