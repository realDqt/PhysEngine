#pragma once

#include <math.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace linear_math {

template <class Scalar, int Rows_>
class Vector : public Eigen::Matrix<Scalar, Rows_, 1> {
    static constexpr int rows = Rows_;
  public:
    using VectorT = Vector<Scalar, rows>;
    using Base = Eigen::Matrix<Scalar, rows, 1>;

    // reuse eigen's constructor
    using Base::Base;
    Vector() : Base() {}

    // This constructor allows you to construct Vector from Eigen expressions
    template <typename OtherDerived>
    Vector(const Eigen::MatrixBase<OtherDerived>& other) : Base(other) {}

    // This method allows you to assign Eigen expressions to Vector
    template <typename OtherDerived>
    Vector& operator=(const Eigen::MatrixBase<OtherDerived>& other) {
        Base::operator=(other);
        return *this;
    }

    VectorT abs() const { return Base::array().abs(); }
    VectorT square() const { return Base::array().abs2(); }
    VectorT inverse() const { return Base::array().inverse(); }
    Scalar min() const { return Base::minCoeff(); }
    Scalar max() const { return Base::maxCoeff(); }
    // using Base::dot;
    // using Base::cross;
    // using Base::normalize;
    // using Base::normalized;
    // using Base::sum;
    // using Base::prod;
    // using Base::mean;
    // using Base::norm;
    // using Base::squaredNorm;
    // using Base::determinant;
    // using Base::transpose;
    // using Base::trace;
    // using Base::all;
    // using Base::any;
    using Base::operator+;
    using Base::operator=;
    using Base::operator-;
    using Base::operator*;
    using Base::operator/;
    using Base::operator+=;
    using Base::operator-=;
    using Base::operator*=;
    using Base::operator/=;

    VectorT operator+(Scalar rhs) { return *this + Ones(rhs); }
    VectorT operator-(Scalar rhs) { return *this - Ones(rhs); }
    friend VectorT operator+(Scalar rhs, const VectorT& lhs) { return lhs + Ones(rhs); }
    friend VectorT operator-(Scalar rhs, const VectorT& lhs) { return Ones(rhs) - lhs; }
    VectorT operator*(const VectorT& rhs) { return this->array() * rhs.array(); }
    VectorT operator/(const VectorT& rhs) { return this->array() / rhs.array(); }
    VectorT& operator+=(Scalar rhs) {
        *this += Ones(rhs);
        return *this;
    }
    VectorT& operator-=(Scalar rhs) {
        *this -= Ones(rhs);
        return *this;
    }

    using Base::Ones;
    using Base::Zero;
    static const VectorT Ones(Scalar n) { return Ones() * n; }
    using Base::Unit;
};

template <class Scalar, int Rows_, int Cols_>
class Matrix : public Eigen::Matrix<Scalar, Rows_, Cols_> {
    static constexpr int rows = Rows_;
    static constexpr int cols = Cols_;
  public:
    using MatrixT = Matrix<Scalar, rows, cols>;
    using Base = Eigen::Matrix<Scalar, rows, cols>;
    // using Vector3 = Vector<Scalar, 3>;
    // using Vector4 = Vector<Scalar, 4>;

    using Base::Base;
    Matrix() : Base() {}

    // This constructor allows you to construct Matrix from Eigen expressions
    template <typename OtherDerived>
    Matrix(const Eigen::MatrixBase<OtherDerived>& other) : Base(other) {}

    // This method allows you to assign Eigen expressions to Matrix
    template <typename OtherDerived>
    Matrix& operator=(const Eigen::MatrixBase<OtherDerived>& other) {
        Base::operator=(other);
        return *this;
    }

    Matrix(Scalar x00, Scalar x01, Scalar x02, Scalar x10, Scalar x11, Scalar x12, Scalar x20, Scalar x21, Scalar x22) {
        (*this) << x00, x01, x02, x10, x11, x12, x20, x21, x22;
    }

    using Base::operator+;
    using Base::operator=;
    using Base::operator-;
    using Base::operator*;
    using Base::operator/;
    using Base::operator+=;
    using Base::operator-=;
    using Base::operator*=;
    using Base::operator/=;
    using Base::operator();

    MatrixT& operator+(const Scalar& v) const { return (*this) + MatrixT::Constant(v); }

    using Base::Constant;
    using Base::Identity;
    using Base::Ones;
    using Base::Zero;
};

}  // namespace linear_math