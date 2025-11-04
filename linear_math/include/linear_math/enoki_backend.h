#include <enoki/array.h>
#include <enoki/matrix.h>
#include <enoki/quaternion.h>

#include "vector_interface.h"

namespace linear_math {

// TODO cast interface

template <class Scalar_, int Size_>
class Vector : public enoki::StaticArrayImpl<Scalar_, Size_, false, Vector<Scalar_, Size_>> {
  public:
    using VectorT = Vector<Scalar_, Size_>;
    using Base = enoki::StaticArrayImpl<Scalar_, Size_, false, Vector<Scalar_, Size_>>;
    template <typename T>
    using ReplaceValue = Vector<T, Size_>;
    using MaskType = enoki::Mask<Scalar_, Size_>;
    ENOKI_ARRAY_IMPORT_BASIC(Base, Vector);
    static constexpr bool IsMatrix = false;
    static constexpr bool IsVector = true;
    using Base::Base;
    Vector() : Base() {}
    Vector(const Base& other) : Base(other) {}
    Vector& operator=(const Base& other) {
        Base::operator=(other);
        return *this;
    }

    Vector(Scalar_ a, Scalar_ b, Scalar_ c) : Base(a, b, c) {}
    Vector(Scalar_ a, Scalar_ b, Scalar_ c, Scalar_ d) : Base(a, b, c, d) {}

    FUNC_ALIAS(dot, Base::dot_, const);
    VectorT cross(const VectorT& rhs) const { return enoki::cross(*this, rhs); }
    VectorT normalized() const { return *this / this->norm(); }
    void normalize() { *this /= enoki::norm(*this); }
    VectorT abs() const { return enoki::abs(*this); }
    VectorT square() const { return enoki::pow(*this, 2); }
    VectorT inverse() const { return enoki::pow(*this, -1); }

    Scalar_ min() const { return enoki::hmin(*this); }
    Scalar_ max() const { return enoki::hmax(*this); }
    Scalar_ sum() const { return enoki::hsum(*this); }
    Scalar_ prod() const { return enoki::hprod(*this); }
    Scalar_ mean() const { return enoki::hmean(*this); }
    Scalar_ norm() const { return enoki::norm(*this); }
    Scalar_ squaredNorm() const { return enoki::squared_norm(*this); }

    bool all() const { return enoki::all(neq(*this, Scalar_(0))); }
    bool any() const { return enoki::any(neq(*this, Scalar_(0))); }

    static auto Zero() { return enoki::zero<VectorT>(); }
    static auto Ones() { return Base(1); }
    static auto Unit(int axis) {
        auto f = VectorT::Zero();
        f[axis] = 1;
        return f;
    }
};

template <typename Scalar_, int Size_>
struct Matrix : public enoki::Matrix<Scalar_, Size_> {
  public:
    using MatrixT = Matrix<Scalar_, Size_>;
    using Base = enoki::Matrix<Scalar_, Size_>;
    template <typename T>
    using ReplaceValue = Matrix<T, Size_>;
    using MaskType = enoki::Mask<Scalar_, Size_>;
    //ENOKI_ARRAY_IMPORT_BASIC(Base, Matrix);
    Matrix(const Matrix&) = default;
    Matrix(Matrix&&) = default;
    Matrix& operator=(const Matrix&) = default;
    Matrix& operator=(Matrix&&) = default;
	//using typename Base::Derived;
    using typename Base::Value;
    using typename Base::Scalar;
    using Base::Size;
    using Base::derived;

    static constexpr bool IsMatrix = true;
    static constexpr bool IsVector = false;

    // template <typename... Args>
    // Matrix(const Args&... args) : Base(args...) { }
    using Base::Base;
    // Matrix() : Base() {}
    Matrix(const Base& other) : Base(other) {}
    Matrix& operator=(const Base& other) {
        Base::operator=(other);
        return *this;
    }

    decltype(auto) coeff(size_t i) { return Base::coeff(i); }
    decltype(auto) coeff(size_t i, size_t j) { return Base::coeff(j, i); }

    // TODO, << initializer

    // NOTE: We have to re-define this rather than reuse the operator in matrix.h because the operator* defined
    //       in array_router.h hides the former one, causing compilation error.
    template <typename T1>
    auto operator*(const Vector<T1, Size>& s) {
        return static_cast<const Base&>(*this) * s;
    }

    static MatrixT Zero() { return enoki::zero<MatrixT>(); }
    static MatrixT Ones() { return enoki::full<MatrixT>(1); }
    static MatrixT Constant(Scalar_ v) { return enoki::full<MatrixT>(v); }
    static MatrixT Identity() { return Base(1); }
};

// NOTE: improve the naming using nested namespace
template <typename Scalar_>
class QuaternionImpl : public enoki::Quaternion<Scalar_> {
  public:
    using QuaternionT = QuaternionImpl<Scalar_>;
    using Base = enoki::Quaternion<Scalar_>;
    template <typename T>
    using ReplaceValue = QuaternionImpl<T>;
    using MaskType = enoki::Mask<Scalar_, 4>;
    //ENOKI_ARRAY_IMPORT_BASIC(Base, QuaternionImpl);
    QuaternionImpl(const QuaternionImpl&) = default;
    QuaternionImpl(QuaternionImpl&&) = default;
    QuaternionImpl& operator=(const QuaternionImpl&) = default;
    QuaternionImpl& operator=(QuaternionImpl&&) = default;
    using Derived = Base::Derived;
    using Base::derived;
    using Base::Size;
    using typename Base::Scalar;
    using typename Base::Value;

    using Matrix3S = Matrix<Scalar, 3>;

    using Base::Base;
    using Base::operator=;
    // The enoki quaternion use the opposite order, x, y, z, w
    QuaternionImpl(Scalar w, Scalar x, Scalar y, Scalar z) : Base(x, y, z, w) {}
    // construct from Matrix
    QuaternionImpl(const Matrix3S& mat) : Base(enoki::matrix_to_quat(mat)) {}

    Matrix3S matrix() const { return enoki::quat_to_matrix<Matrix3S>(*this); }
    using Base::w;
    using Base::x;
    using Base::y;
    using Base::z;

    static constexpr bool IsQuaternion = true;
    static constexpr bool IsVector = false;

    static QuaternionT Identity() { return QuaternionT(1, 0, 0, 0); }
};

}  // namespace linear_math