#pragma once
#include <iostream>

namespace linear_math {

template <typename Scalar, int rows, template<typename, int> class Derived_T>
class VectorInterface {
  private:
    using DerivedT = Derived_T<Scalar, rows>;
  public:
    Scalar dot(const DerivedT& rhs) const { return getThis().dot_(rhs); }
    friend Scalar dot(const DerivedT& lhs, const DerivedT& rhs) { return lhs.dot(rhs); }
    DerivedT cross(const DerivedT& rhs) const { return getThis().cross_(rhs); }
    void normalize() { getThis().normalize_(); }
    DerivedT normalized() const { return getThis().normalized_(); }
    Real angle(const DerivedT& rhs) const { return getThis().angle_(rhs); }

    // element-wise operations
    DerivedT abs() const { return getThis().abs_(); }
    DerivedT squared() const { return getThis().squared_(); }
    // e.g., Vector2(2,4) -> Vector2(0.5, 0.25)
    DerivedT inverse() const { return getThis().inverse_(); }

    // reduction operations
    Scalar min() const { return getThis().min_(); }
    Scalar max() const { return getThis().max_(); }
    Scalar sum() const { return getThis().sum_(); }
    Scalar product() const { return getThis().product_(); }
    Scalar mean() const { return getThis().mean_(); }
    Scalar norm() const { return getThis().norm_(); }
    Scalar squaredNorm() const { return getThis().squaredNorm_(); }
    bool all() const { return getThis().all_(); }
    bool any() const { return getThis().any_(); }

    // algebra operations
    // DerivedT operator+ (const DerivedT& rhs);
    // DerivedT operator- (const DerivedT& rhs);
    // DerivedT operator- ();  // unary negative
    // DerivedT operator* (Scalar rhs);
    // friend DerivedT operator* (const DerivedT& lhs, const DerivedT& rhs);
    // DerivedT operator/ (Scalar rhs);
    // DerivedT operator+ (Scalar rhs);
    // DerivedT operator- (Scalar rhs);
    // inplace algebra operations
    // DerivedT& operator+= (const DerivedT& rhs);
    // DerivedT& operator-= (const DerivedT& rhs);
    // DerivedT& operator*= (Scalar rhs);
    // DerivedT& operator/= (Scalar rhs);
    // DerivedT& operator+= (Scalar rhs);
    // DerivedT& operator-= (Scalar rhs);

    // useful constants
    static const DerivedT Zero() {return DerivedT::Zero(); }
    static const DerivedT Ones() {return DerivedT::Ones(); }
    static const DerivedT Ones(Scalar n) {return DerivedT::Ones() * n; }
    static const DerivedT Unit(int axis) {return DerivedT::Unit(axis); }

  private:
    DerivedT& getThis() { return static_cast<DerivedT&>(*this); }
    DerivedT const& getThis() const { return static_cast<DerivedT const&>(*this); }
};

}  // namespace linear_math