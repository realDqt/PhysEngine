#include <iostream>

#include "linear_math/constants.h"
#include "linear_math/matrix.h"
#include "linear_math/quaternion.h"
#include "linear_math/vector.h"

using namespace linear_math;

#ifdef __APPLE__
#define __FILENAME__ (strrchr(__FILE__, '/') + 1)
#else
#define __FILENAME__ (strrchr(__FILE__, '\\') + 1)
#endif

#define MY_ASSERT(cond, message)           \
    if (!(cond)) {                         \
        std::cout << message << std::endl; \
    }
// std::cout << "[" << __FILENAME__ << ":" << __LINE__ << "] " << message << std::endl; \

template <typename T, typename T2>
inline bool eq(T a, T2 b, Real EPSILON = 1e-6) {
    return fabs(a - b) < EPSILON;
}

void testVector() {
    // test constructor
    Vector3 v;
    Vector3 v1(0.5, 1.0, -1.0);
    Vector4 v2(0, 1.0, 0, 1);
    Vector4 v3 = Vector4::Zero();
    Vector4 v4 = Vector4::Ones();
    Vector4 vUnit = Vector4::Unit(1);
    MY_ASSERT(vUnit == Vector4(0, 1, 0, 0), "unit error");

    // test wrapper has no memory overhead
#ifdef EIGEN_BACKEND
    std::cout<<"eigen"<<std::endl;
    static_assert(sizeof(Vector3) == sizeof(Eigen::Matrix<Real, 3, 1>), "wrapper has memory overhead");
    static_assert(sizeof(Vector4) == sizeof(Eigen::Matrix<Real, 4, 1>), "wrapper has memory overhead");
#else
    std::cout<<"enoki"<<std::endl;
    static_assert(sizeof(Vector3) == sizeof(enoki::Array<Real, 3>), "wrapper has memory overhead");
    static_assert(sizeof(Vector4) == sizeof(enoki::Array<Real, 4>), "wrapper has memory overhead");
#endif

    MY_ASSERT(v1 * 2 == Vector3(1.0, 2.0, -2.0), "scalar product error");

    // test scalar multiply
    MY_ASSERT(v1 / 2.0 == Vector3(0.25, 0.5, -0.5), "scalar division error");

    // test dot product
    Vector3 v1Prime(2.0, 1.0, 3.0);
    MY_ASSERT(v1.dot(v1Prime) == -1.0, "dot product error");
    // MY_ASSERT(dot(v1, v1Prime)==-1.0, "dot product error");

    // test cross product
    Vector3 crossRes = v1.cross(v1Prime);
    MY_ASSERT(v1.cross(v1Prime) == Vector3(4.0, -3.5, -1.5), "cross product error");
    Vector2 v5(3.0, 4.0);
    auto v5n = v5.normalized();
    MY_ASSERT(eq(v5n[0], 0.6) && eq(v5n[1], 0.8), "normalized error");

    // test normalize
    v5.normalize();
    MY_ASSERT(eq(v5[0], 0.6) && eq(v5[1], 0.8), "normalize error");

    // test angle  TODO remove, not a standard Eigen interface
    // Vector2 a1(1, 0), a2(1, 1);
    // std::cout<<a1.angle(a2)<<std::endl;
    // MY_ASSERT(fabs(a1.angle(a2) - 3.1415926 / 4.0) < 1e-6, "angle error");

    // element-wise abs
    MY_ASSERT(v1.abs() == Vector3(0.5, 1.0, 1.0), "element-wise abs error");
    // element-wise squared
    v5 = Vector2(3.0, 4.0);
    MY_ASSERT(v5.square() == Vector2(9, 16), "element-wise squared error");
    // element-wise inverse
    MY_ASSERT(v1.inverse() == Vector3(2.0, 1.0, -1.0), "element-wise inverse error");

    // reduction
    MY_ASSERT(v1.min() == -1, "min error");
    MY_ASSERT(v1.max() == 1, "max error");
    MY_ASSERT(v1.sum() == 0.5, "sum error");
    MY_ASSERT(v1.prod() == -0.5, "product error");
    MY_ASSERT(fabs(v1.mean() - 0.5 / 3.0) < 1e-6, "mean error");
    MY_ASSERT(v1.norm() == 1.5, "norm error");
    MY_ASSERT(v1.squaredNorm() == 2.25, "squared norm error");
    Vector4 v6(0, 1, 2, 3);
    MY_ASSERT(v1.all() == true, "all error");
    MY_ASSERT(v6.all() == false, "all error");
    MY_ASSERT(v6.any() == true, "any error");
    MY_ASSERT(v3.any() == false, "any error");

    // algebra operations
    v1Prime = Vector3(2.0, 1.0, 3.0);
    MY_ASSERT(v1 + 1 == Vector3(1.5, 2.0, 0.0), "addition error");
    MY_ASSERT(1 + v1 == Vector3(1.5, 2.0, 0.0), "addition error");
    MY_ASSERT(v1 + v1Prime == Vector3(2.5, 2.0, 2.0), "addition error");
    MY_ASSERT(v1 * v1Prime == Vector3(1.0, 1.0, -3.0), "element-wise multiply error");
    MY_ASSERT(v1Prime / v1 == Vector3(4.0, 1.0, -3.0), "element-wise multiply error");
    MY_ASSERT(v1 - 1 == Vector3(-0.5, 0, -2.0), "substraction error");
    MY_ASSERT(-v1 == Vector3(-0.5, -1, 1.0), "negative error");
    Vector3 v7 = Vector3::Zero();
    v7 += 1;
    MY_ASSERT(v7 == Vector3(1.0, 1.0, 1.0), "inplace addition error");
    v7 -= 2;
    MY_ASSERT(v7 == Vector3(-1.0, -1.0, -1.0), "inplace substraction error");
}

void testMatrix() {
    // std::cout<<Matrix3(1, 2, 3, 4, 5, 6, 7, 8, 9)<<std::endl;
    Matrix3 m0 = Matrix3::Ones();
    Matrix3 m(1, 2, 3, 4, 5, 6, 7, 8, 9);
    // m << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    // MY_ASSERT(m == Matrix3(1, 2, 3, 4, 5, 6, 7, 8, 9), "initialize dismatch");

    MY_ASSERT(Matrix3::Ones() == Matrix3(1, 1, 1, 1, 1, 1, 1, 1, 1), "Ones dismatch");
    MY_ASSERT(Matrix3::Ones() * 3 == Matrix3::Constant(3), "constant dismatch");
    MY_ASSERT(Matrix3::Zero() == Matrix3(0, 0, 0, 0, 0, 0, 0, 0, 0), "zero constant dismatch");
    MY_ASSERT(Matrix3::Identity() == Matrix3(1, 0, 0, 0, 1, 0, 0, 0, 1), "identity constant dismatch");

    Vector3 v(1, 1, 1);
    MY_ASSERT(m * v == Vector3(6, 15, 24), "ERROR: matirx/vector multiply");
    MY_ASSERT(m * m0 == Matrix3(6, 6, 6, 15, 15, 15, 24, 24, 24), "ERROR: matrix/matrix multiply");
    MY_ASSERT(m0 * 3 == Matrix3::Constant(3), "ERROR: matrix/scalar multiply");
    MY_ASSERT(3 * m0 == Matrix3::Constant(3), "ERROR: scalar/matrix multiply");
    MY_ASSERT(Matrix3::Constant(2) / 2 == Matrix3::Constant(1), "ERROR: matrix/scalar division");

    // NOTE: no dynamic matrix for now
    // MatrixX m_dynamic(6, 6);
    // std::cout << m_dynamic << std::endl;
}

template <typename T>
bool isQuaternionEq(const T& q0, const T& q1) {
    return eq(q0.x(), q1.x()) && eq(q0.y(), q1.y()) && eq(q0.z(), q1.z()) && eq(q0.w(), q1.w());
}

void testQuaternion() {
    ///// TODO: ERROR
    //Quaternion q(PI / 2.0, 0.0, 0.0, 1.0);
    //// std::cout<<q.matrix()<<std::endl;
    //Matrix3 rot(-1, -PI, 0, PI, -1, 0, 0, 0, 1);
    //Quaternion q1(rot);
    //MY_ASSERT(isQuaternionEq(q, q1), "ERROR: matrix construction");
    //MY_ASSERT(eq(q.matrix().coeff(0, 1), -PI), "ERROR: matrix interface");

    //MY_ASSERT(isQuaternionEq(Quaternion::Identity(), Quaternion(1.0, 0.0, 0.0, 0.0)), "ERROR: identity");

    //MY_ASSERT(eq(PI, (q * q1).z()), "ERROR: multiply");
}

int main() {
    testVector();
    testMatrix();
    testQuaternion();
    // Eigen::Vector4f p(3,4,5,6);
    // std::cout<<p.sum()<<std::endl;
    // Eigen::Matrix<int, 3, 1> p(2, 1, 1);
    // std::cout<<p.norm()<<std::endl;
    // std::cout<<p * p<<std::endl;
    // p += Eigen::Vector4f::Ones();
    // std::cout<< p <<std::endl;

    return 0;
}