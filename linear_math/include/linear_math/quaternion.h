#pragma once
#include "common.h"

namespace linear_math {

#if defined(EIGEN_BACKEND)
using Quaternion = Eigen::Quaternion<Real>;
#elif defined(ENOKI_BACKEND)
using Quaternion = QuaternionImpl<Real>;
#endif

}  // namespace linear_math