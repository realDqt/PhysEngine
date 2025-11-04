#pragma once
#include "vector.h"

namespace linear_math {

template <typename Scalar, int rows, int cols, template <typename, int, int> class Derived_T>
class MatrixInterface {
    using Derived = Derived_T<Scalar, rows, cols>;
    // using RowVectorT = Vector;
  public:
    // MatrixInterface() {}
    // MatrixInterface(Scalar x00, Scalar x01, Scalar x02, Scalar x10, Scalar x11, Scalar x12, Scalar x20, Scalar x21,
    //                 Scalar x22) {}

    // useful constants
    static const Derived Zero() { return Derived::Zero(); }
    static const Derived Ones() { return Derived::Ones(); }
    static const Derived Identity() { return Derived::Identity(); }
    static const Derived Constant(Scalar n) { return Derived::Constant(n); }

  private:
    Derived& getThis() { return static_cast<Derived&>(*this); }
    Derived const& getThis() const { return static_cast<Derived const&>(*this); }
};

}  // namespace linear_math