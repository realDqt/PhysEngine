#pragma once

#include "common/real.h"


//// if use backend wrapper
#if PE_USE_BACKEND_WRAPPER

#include <linear_math/vector.h>
#include <linear_math/matrix.h>
#include <linear_math/quaternion.h>

PHYS_NAMESPACE_BEGIN

using Vector3 = linear_math::Vector3;
using Matrix3x3 = linear_math::Matrix3;
using Quaternion = linear_math::Quaternion;

PHYS_NAMESPACE_END

//// else use backend wrapper
#else

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/Geometry>
#include <Eigen/SVD>

PHYS_NAMESPACE_BEGIN

#if USE_DOUBLE

using Vector3 = Eigen::Vector3d;
using Matrix3x3 = Eigen::Matrix3d;
using Quaternion = Eigen::Quaterniond;

#else

using Vector3 = Eigen::Vector3f;
using Matrix3x3 = Eigen::Matrix3f;
using Quaternion = Eigen::Quaternionf;

#endif

PHYS_NAMESPACE_END
#endif
