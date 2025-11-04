
#ifndef CUDA_MATH_H
#define CUDA_MATH_H

#include <math.h>

// #include <nvVector.h>
// #include <nvMatrix.h>
// #include <nvQuaternion.h>

// #include "math/math.h"
#include "common/real.h"
#ifndef USE_DOUBLE
#include "helper_math.h"

#define make_vec2r(...) make_float2(__VA_ARGS__)
#define make_vec3r(...) make_float3(__VA_ARGS__)
#define make_vec4r(...) make_float4(__VA_ARGS__)
#else
static_assert("double unimplemented");
#endif

// typedef vec2<float> vec2f;
// typedef vec3<float> vec3f;
// typedef vec3<int> vec3i;
// typedef vec3<unsigned int> vec3ui;
// typedef vec4<float> vec4f;
// typedef matrix4<float> matrix4f;
// typedef quaternion<float> quaternionf;

#ifndef USE_DOUBLE
typedef float2 vec2r;
typedef float3 vec3r;
typedef float4 vec4r;
#else
    /// cause error now!
    typedef double2 vec2r;
typedef double3 vec3r;
typedef double4 vec4r;
#endif

inline bool inverse(const vec3r *mat, Real det, vec3r *inv)
{
  det = mat[0].x * (mat[1].y * mat[2].z - mat[2].y * mat[1].z) -
        mat[0].y * (mat[1].x * mat[2].z - mat[1].z * mat[2].x) +
        mat[0].z * (mat[1].x * mat[2].y - mat[1].y * mat[2].x);

  if (fabs(det) < 1e-8)
    return false;

  Real invdet = 1 / det;
  inv[0].x = (mat[1].y * mat[2].z - mat[2].y * mat[1].z) * invdet;
  inv[0].y = (mat[0].z * mat[2].y - mat[0].y * mat[2].z) * invdet;
  inv[0].z = (mat[0].y * mat[1].z - mat[0].z * mat[1].y) * invdet;
  inv[1].x = (mat[1].z * mat[2].x - mat[1].x * mat[2].z) * invdet;
  inv[1].y = (mat[0].x * mat[2].z - mat[0].z * mat[2].x) * invdet;
  inv[1].z = (mat[1].x * mat[0].z - mat[0].x * mat[1].z) * invdet;
  inv[2].x = (mat[1].x * mat[2].y - mat[2].x * mat[1].y) * invdet;
  inv[2].y = (mat[2].x * mat[0].y - mat[0].x * mat[2].y) * invdet;
  inv[2].z = (mat[0].x * mat[1].y - mat[1].x * mat[0].y) * invdet;

  return true;
}

#include <iostream>
// ref to https://github.com/wyegelwel/snow/blob/b504448296e6c161f25098d12c4b5358220e767a/project/cuda/matrix.h
class mat3r
{
public:
  union
  {
    vec3r rows[3];
    Real data[9];
  };

  __device__ __host__ __forceinline__ mat3r()
  {
    rows[0] = make_vec3r(0.0f);
    rows[1] = make_vec3r(0.0f);
    rows[2] = make_vec3r(0.0f);
  }

  __device__ __host__ __forceinline__ mat3r(Real i)
  {
    rows[0] = make_vec3r(i, 0.0f, 0.0f);
    rows[1] = make_vec3r(0.0f, i, 0.0f);
    rows[2] = make_vec3r(0.0f, 0.0f, i);
  }

  __device__ __host__ __forceinline__ mat3r(const vec3r &diag)
  {
    rows[0] = make_vec3r(diag.x, 0.0f, 0.0f);
    rows[1] = make_vec3r(0.0f, diag.y, 0.0f);
    rows[2] = make_vec3r(0.0f, 0.0f, diag.z);
  }

  __device__ __host__ __forceinline__ mat3r(const vec3r &r0, const vec3r &r1, const vec3r &r2)
  {
    rows[0] = r0;
    rows[1] = r1;
    rows[2] = r2;
  }

  __host__ __device__ __forceinline__
  mat3r(float a, float b, float c, float d, float e, float f, float g, float h, float i)
  {
    data[0] = a;
    data[3] = d;
    data[6] = g;
    data[1] = b;
    data[4] = e;
    data[7] = h;
    data[2] = c;
    data[5] = f;
    data[8] = i;
  }

  __host__ __device__ __forceinline__
      mat3r &
      operator=(const mat3r &rhs)
  {
    data[0] = rhs[0];
    data[3] = rhs[3];
    data[6] = rhs[6];
    data[1] = rhs[1];
    data[4] = rhs[4];
    data[7] = rhs[7];
    data[2] = rhs[2];
    data[5] = rhs[5];
    data[8] = rhs[8];
    return *this;
  }

  __host__ __device__ __forceinline__ Real &operator[](int i) { return data[i]; }
  __host__ __device__ __forceinline__ Real operator[](int i) const { return data[i]; }
  __host__ __device__ __forceinline__ Real &operator()(int i) { return data[i]; }
  __host__ __device__ __forceinline__ Real operator()(int i) const { return data[i]; }
  __host__ __device__ __forceinline__ Real &operator()(int r, int c) { return data[r * 3 + c]; }
  __host__ __device__ __forceinline__ Real operator()(int r, int c) const { return data[r * 3 + c]; }

  __host__ __device__ __forceinline__
      mat3r &
      operator*=(const mat3r &rhs)
  {
    mat3r tmp;
    // tmp[0] = data[0]*rhs[0] + data[3]*rhs[1] + data[6]*rhs[2];
    // tmp[1] = data[1]*rhs[0] + data[4]*rhs[1] + data[7]*rhs[2];
    // tmp[2] = data[2]*rhs[0] + data[5]*rhs[1] + data[8]*rhs[2];
    // tmp[3] = data[0]*rhs[3] + data[3]*rhs[4] + data[6]*rhs[5];
    // tmp[4] = data[1]*rhs[3] + data[4]*rhs[4] + data[7]*rhs[5];
    // tmp[5] = data[2]*rhs[3] + data[5]*rhs[4] + data[8]*rhs[5];
    // tmp[6] = data[0]*rhs[6] + data[3]*rhs[7] + data[6]*rhs[8];
    // tmp[7] = data[1]*rhs[6] + data[4]*rhs[7] + data[7]*rhs[8];
    // tmp[8] = data[2]*rhs[6] + data[5]*rhs[7] + data[8]*rhs[8];

    tmp[0] = rhs[0] * data[0] + rhs[3] * data[1] + rhs[6] * data[2];
    tmp[1] = rhs[1] * data[0] + rhs[4] * data[1] + rhs[7] * data[2];
    tmp[2] = rhs[2] * data[0] + rhs[5] * data[1] + rhs[8] * data[2];
    tmp[3] = rhs[0] * data[3] + rhs[3] * data[4] + rhs[6] * data[5];
    tmp[4] = rhs[1] * data[3] + rhs[4] * data[4] + rhs[7] * data[5];
    tmp[5] = rhs[2] * data[3] + rhs[5] * data[4] + rhs[8] * data[5];
    tmp[6] = rhs[0] * data[6] + rhs[3] * data[7] + rhs[6] * data[8];
    tmp[7] = rhs[1] * data[6] + rhs[4] * data[7] + rhs[7] * data[8];
    tmp[8] = rhs[2] * data[6] + rhs[5] * data[7] + rhs[8] * data[8];
    return (*this = tmp);
  }

  __host__ __device__ __forceinline__
      mat3r
      operator*(const mat3r &rhs) const
  {
    mat3r result;
    // result[0] = data[0]*rhs[0] + data[3]*rhs[1] + data[6]*rhs[2];
    // result[1] = data[1]*rhs[0] + data[4]*rhs[1] + data[7]*rhs[2];
    // result[2] = data[2]*rhs[0] + data[5]*rhs[1] + data[8]*rhs[2];
    // result[3] = data[0]*rhs[3] + data[3]*rhs[4] + data[6]*rhs[5];
    // result[4] = data[1]*rhs[3] + data[4]*rhs[4] + data[7]*rhs[5];
    // result[5] = data[2]*rhs[3] + data[5]*rhs[4] + data[8]*rhs[5];
    // result[6] = data[0]*rhs[6] + data[3]*rhs[7] + data[6]*rhs[8];
    // result[7] = data[1]*rhs[6] + data[4]*rhs[7] + data[7]*rhs[8];
    // result[8] = data[2]*rhs[6] + data[5]*rhs[7] + data[8]*rhs[8];
    result[0] = rhs[0] * data[0] + rhs[3] * data[1] + rhs[6] * data[2];
    result[1] = rhs[1] * data[0] + rhs[4] * data[1] + rhs[7] * data[2];
    result[2] = rhs[2] * data[0] + rhs[5] * data[1] + rhs[8] * data[2];
    result[3] = rhs[0] * data[3] + rhs[3] * data[4] + rhs[6] * data[5];
    result[4] = rhs[1] * data[3] + rhs[4] * data[4] + rhs[7] * data[5];
    result[5] = rhs[2] * data[3] + rhs[5] * data[4] + rhs[8] * data[5];
    result[6] = rhs[0] * data[6] + rhs[3] * data[7] + rhs[6] * data[8];
    result[7] = rhs[1] * data[6] + rhs[4] * data[7] + rhs[7] * data[8];
    result[8] = rhs[2] * data[6] + rhs[5] * data[7] + rhs[8] * data[8];
    return result;
  }

  __host__ __device__ __forceinline__
      vec3r
      operator*(const vec3r &rhs) const
  {
    vec3r result;
    // result.x = data[0]*rhs.x + data[3]*rhs.y + data[6]*rhs.z;
    // result.y = data[1]*rhs.x + data[4]*rhs.y + data[7]*rhs.z;
    // result.z = data[2]*rhs.x + data[5]*rhs.y + data[8]*rhs.z;
    result.x = data[0] * rhs.x + data[1] * rhs.y + data[2] * rhs.z;
    result.y = data[3] * rhs.x + data[4] * rhs.y + data[5] * rhs.z;
    result.z = data[6] * rhs.x + data[7] * rhs.y + data[8] * rhs.z;
    return result;
  }

  __host__ __device__ __forceinline__
      mat3r &
      operator+=(const mat3r &rhs)
  {
    data[0] += rhs[0];
    data[3] += rhs[3];
    data[6] += rhs[6];
    data[1] += rhs[1];
    data[4] += rhs[4];
    data[7] += rhs[7];
    data[2] += rhs[2];
    data[5] += rhs[5];
    data[8] += rhs[8];
    return *this;
  }

  __host__ __device__ __forceinline__
      mat3r
      operator+(const mat3r &rhs) const
  {
    mat3r tmp = *this;
    tmp[0] += rhs[0];
    tmp[3] += rhs[3];
    tmp[6] += rhs[6];
    tmp[1] += rhs[1];
    tmp[4] += rhs[4];
    tmp[7] += rhs[7];
    tmp[2] += rhs[2];
    tmp[5] += rhs[5];
    tmp[8] += rhs[8];
    return tmp;
  }

  __host__ __device__ __forceinline__
      mat3r &
      operator-=(const mat3r &rhs)
  {
    data[0] -= rhs[0];
    data[3] -= rhs[3];
    data[6] -= rhs[6];
    data[1] -= rhs[1];
    data[4] -= rhs[4];
    data[7] -= rhs[7];
    data[2] -= rhs[2];
    data[5] -= rhs[5];
    data[8] -= rhs[8];
    return *this;
  }

  __host__ __device__ __forceinline__
      mat3r
      operator-(const mat3r &rhs) const
  {
    mat3r tmp = *this;
    tmp[0] -= rhs[0];
    tmp[3] -= rhs[3];
    tmp[6] -= rhs[6];
    tmp[1] -= rhs[1];
    tmp[4] -= rhs[4];
    tmp[7] -= rhs[7];
    tmp[2] -= rhs[2];
    tmp[5] -= rhs[5];
    tmp[8] -= rhs[8];
    return tmp;
  }

  __host__ __device__ __forceinline__
      mat3r &
      operator*=(float f)
  {
    data[0] *= f;
    data[3] *= f;
    data[6] *= f;
    data[1] *= f;
    data[4] *= f;
    data[7] *= f;
    data[2] *= f;
    data[5] *= f;
    data[8] *= f;
    return *this;
  }

  __host__ __device__ __forceinline__
      mat3r
      operator*(float f) const
  {
    mat3r tmp = *this;
    tmp[0] *= f;
    tmp[3] *= f;
    tmp[6] *= f;
    tmp[1] *= f;
    tmp[4] *= f;
    tmp[7] *= f;
    tmp[2] *= f;
    tmp[5] *= f;
    tmp[8] *= f;
    return tmp;
  }

  __host__ __device__ __forceinline__
      mat3r &
      operator/=(float f)
  {
    float fi = 1.f / f;
    data[0] *= fi;
    data[3] *= fi;
    data[6] *= fi;
    data[1] *= fi;
    data[4] *= fi;
    data[7] *= fi;
    data[2] *= fi;
    data[5] *= fi;
    data[8] *= fi;
    return *this;
  }

  __host__ __device__ __forceinline__
      mat3r
      operator/(float f) const
  {
    mat3r tmp = *this;
    float fi = 1.f / f;
    tmp[0] *= fi;
    tmp[3] *= fi;
    tmp[6] *= fi;
    tmp[1] *= fi;
    tmp[4] *= fi;
    tmp[7] *= fi;
    tmp[2] *= fi;
    tmp[5] *= fi;
    tmp[8] *= fi;
    return tmp;
  }

  __host__ __device__ __forceinline__
      mat3r
      transpose() const
  {
    return mat3r(data[0], data[3], data[6],
                 data[1], data[4], data[7],
                 data[2], data[5], data[8]);
  }

  __host__ __device__ __forceinline__
      Real
      trace() const
  {
    return data[0] + data[4] + data[8];
  }

  __host__ __device__ __forceinline__
      Real
      sum2() const
  {
    return data[0] * data[0] + data[3] * data[3] + data[6] * data[6] +
           data[1] * data[1] + data[4] * data[4] + data[7] * data[7] +
           data[2] * data[2] + data[5] * data[5] + data[8] * data[8];
  }

  __host__ __device__ __forceinline__ static mat3r transpose(const mat3r &m)
  {
    return mat3r(m[0], m[3], m[6],
                 m[1], m[4], m[7],
                 m[2], m[5], m[8]);
  }

  // __host__ __device__ __forceinline__
  // static mat3r inverse( const mat3r &M )
  // {
  //     Real det = (M[0]*(M[4]*M[8]-M[7]*M[5]) -
  //                 M[3]*(M[1]*M[8]-M[7]*M[2]) +
  //                 M[6]*(M[1]*M[5]-M[4]*M[2]));
  //     mat3r A;
  //     if(fabs(det)<1e-8) return A;
  //     Real invDet = 1.0 / det;
  //     A[0] = invDet * (M[4]*M[8]-M[5]*M[7]);
  //     A[1] = invDet * (M[2]*M[7]-M[1]*M[8]);
  //     A[2] = invDet * (M[1]*M[5]-M[2]*M[4]);
  //     A[3] = invDet * (M[5]*M[6]-M[3]*M[8]);
  //     A[4] = invDet * (M[0]*M[8]-M[2]*M[6]);
  //     A[5] = invDet * (M[2]*M[3]-M[0]*M[5]);
  //     A[6] = invDet * (M[3]*M[7]-M[4]*M[6]);
  //     A[7] = invDet * (M[1]*M[6]-M[0]*M[7]);
  //     A[8] = invDet * (M[0]*M[4]-M[1]*M[3]);
  //     return A;
  // }

  __host__ __device__ __forceinline__ static mat3r inverse(const mat3r &mat)
  {
    Real det = mat(0, 0) * (mat(1, 1) * mat(2, 2) - mat(2, 1) * mat(1, 2)) -
               mat(0, 1) * (mat(1, 0) * mat(2, 2) - mat(1, 2) * mat(2, 0)) +
               mat(0, 2) * (mat(1, 0) * mat(2, 1) - mat(1, 1) * mat(2, 0));

    mat3r inv;
    if (fabs(det) < 1e-8)
      return inv;

    Real invdet = 1 / det;
    inv(0, 0) = (mat(1, 1) * mat(2, 2) - mat(2, 1) * mat(1, 2)) * invdet;
    inv(0, 1) = (mat(0, 2) * mat(2, 1) - mat(0, 1) * mat(2, 2)) * invdet;
    inv(0, 2) = (mat(0, 1) * mat(1, 2) - mat(0, 2) * mat(1, 1)) * invdet;
    inv(1, 0) = (mat(1, 2) * mat(2, 0) - mat(1, 0) * mat(2, 2)) * invdet;
    inv(1, 1) = (mat(0, 0) * mat(2, 2) - mat(0, 2) * mat(2, 0)) * invdet;
    inv(1, 2) = (mat(1, 0) * mat(0, 2) - mat(0, 0) * mat(1, 2)) * invdet;
    inv(2, 0) = (mat(1, 0) * mat(2, 1) - mat(2, 0) * mat(1, 1)) * invdet;
    inv(2, 1) = (mat(2, 0) * mat(0, 1) - mat(0, 0) * mat(2, 1)) * invdet;
    inv(2, 2) = (mat(0, 0) * mat(1, 1) - mat(1, 0) * mat(0, 1)) * invdet;

    return inv;
  }
  __host__ __device__ __forceinline__
      mat3r
      inverse() const
  {

    Real det = data[0] * (data[4] * data[8] - data[7] * data[5]) -
               data[1] * (data[3] * data[8] - data[5] * data[6]) +
               data[2] * (data[3] * data[7] - data[4] * data[6]);

    mat3r inv;
    if (fabs(det) < 1e-8)
      return inv;

    Real invdet = 1 / det;
    inv(0, 0) = (data[4] * data[8] - data[7] * data[5]) * invdet;
    inv(0, 1) = (data[2] * data[7] - data[1] * data[8]) * invdet;
    inv(0, 2) = (data[1] * data[5] - data[2] * data[4]) * invdet;
    inv(1, 0) = (data[5] * data[6] - data[3] * data[8]) * invdet;
    inv(1, 1) = (data[0] * data[8] - data[2] * data[6]) * invdet;
    inv(1, 2) = (data[3] * data[2] - data[0] * data[5]) * invdet;
    inv(2, 0) = (data[3] * data[7] - data[6] * data[4]) * invdet;
    inv(2, 1) = (data[6] * data[1] - data[0] * data[7]) * invdet;
    inv(2, 2) = (data[0] * data[4] - data[3] * data[1]) * invdet;

    return inv;
  }

  friend std::ostream &operator<<(std::ostream &os, const mat3r &m);
};

class quaternionf
{
public:
  union
  {
    vec4r _array;
  };
  __host__ __device__ __forceinline__ quaternionf()
  {
    _array = make_vec4r(0.0, 0.0, 0.0, 0.0);
  }

  __host__ __device__ __forceinline__ quaternionf(const vec3r &axis, Real radians)
  {
    set_value(axis, radians);
  }

  __host__ __device__ __forceinline__ quaternionf(const vec3r &point)
  {
    _array.x = point.x;
    _array.y = point.y;
    _array.z = point.z;
    _array.w = 0;
  }

  __host__ __device__ __forceinline__ quaternionf(const Real v[4])
  {
    set_value(v);
  }

  __host__ __device__ __forceinline__ quaternionf(const quaternionf &v)
  {
    set_value(v);
  }

  __host__ __device__ __forceinline__ quaternionf(Real q0, Real q1, Real q2, Real q3)
  {
    set_value(q0, q1, q2, q3);
  }

  // __host__ __device__ __forceinline__ quaternionf(const mat4r &m)
  // {
  //   set_value(m);
  // }

  __host__ __device__ __forceinline__ quaternionf(const vec3r &rotateFrom, const vec3r &rotateTo)
  {
    set_value(rotateFrom, rotateTo);
  }

  // __host__ __device__ __forceinline__ quaternionf(const vec3r &from_look, const vec3r &from_up,
  //                                                 const vec3r &to_look, const vec3r &to_up)
  // {
  //   set_value(from_look, from_up, to_look, to_up);
  // }

  __host__ __device__ __forceinline__ const vec4r get_value() const
  {
    return _array;
  }

  __host__ __device__ __forceinline__ void get_value(Real &q0, Real &q1, Real &q2, Real &q3) const
  {
    q0 = _array.x;
    q1 = _array.y;
    q2 = _array.z;
    q3 = _array.w;
  }

  __host__ __device__ __forceinline__ quaternionf &set_value(Real q0, Real q1, Real q2, Real q3)
  {
    _array.x = q0;
    _array.y = q1;
    _array.z = q2;
    _array.w = q3;
    return *this;
  }

  __host__ __device__ __forceinline__ void get_value(vec3r &axis, Real &radians) const
  {
    radians = Real(acos(_array.w) * Real(2.0));

    if (radians == Real(0.0))
    {
      axis = make_vec3r(0.0, 0.0, 1.0);
    }
    else
    {
      axis.x = _array.x;
      axis.y = _array.y;
      axis.z = _array.z;
      axis = normalize(axis);
    }
  }

  __host__ __device__ __forceinline__ Real &x()
  {
    return _array.x;
  }

  __host__ __device__ __forceinline__ Real &y()
  {
    return _array.y;
  }

  __host__ __device__ __forceinline__ Real &z()
  {
    return _array.z;
  }

  __host__ __device__ __forceinline__ Real &w()
  {
    return _array.w;
  }

  // __host__ __device__ __forceinline__ void get_value(matrix4<Real> &m) const
  // {
  //   Real s, xs, ys, zs, wx, wy, wz, xx, xy, xz, yy, yz, zz;

  //   Real norm = _array[0] * _array[0] + _array[1] * _array[1] + _array[2] * _array[2] + _array[3] * _array[3];

  //   s = (norm == Real(0.0)) ? Real(0.0) : (Real(2.0) / norm);

  //   xs = _array[0] * s;
  //   ys = _array[1] * s;
  //   zs = _array[2] * s;

  //   wx = _array[3] * xs;
  //   wy = _array[3] * ys;
  //   wz = _array[3] * zs;

  //   xx = _array[0] * xs;
  //   xy = _array[0] * ys;
  //   xz = _array[0] * zs;

  //   yy = _array[1] * ys;
  //   yz = _array[1] * zs;
  //   zz = _array[2] * zs;

  //   m(0, 0) = Real(Real(1.0) - (yy + zz));
  //   m(1, 0) = Real(xy + wz);
  //   m(2, 0) = Real(xz - wy);

  //   m(0, 1) = Real(xy - wz);
  //   m(1, 1) = Real(Real(1.0) - (xx + zz));
  //   m(2, 1) = Real(yz + wx);

  //   m(0, 2) = Real(xz + wy);
  //   m(1, 2) = Real(yz - wx);
  //   m(2, 2) = Real(Real(1.0) - (xx + yy));

  //   m(3, 0) = m(3, 1) = m(3, 2) = m(0, 3) = m(1, 3) = m(2, 3) = Real(0.0);
  //   m(3, 3) = Real(1.0);
  // }

  __host__ __device__ __forceinline__ quaternionf &set_value(const Real *qp)
  {

    _array.x = qp[0];
    _array.y = qp[1];
    _array.z = qp[2];
    _array.w = qp[3];

    return *this;
  }

  __host__ __device__ __forceinline__ quaternionf &set_value(const quaternionf &v)
  {
    _array = v._array;
    return *this;
  }

  // __host__ __device__ __forceinline__ quaternionf &set_value(const matrix4<Real> &m)
  // {
  //   Real tr, s;
  //   int i, j, k;
  //   const int nxt[3] = {1, 2, 0};

  //   tr = m(0, 0) + m(1, 1) + m(2, 2);

  //   if (tr > Real(0))
  //   {
  //     s = Real(sqrt(tr + m(3, 3)));
  //     _array[3] = Real(s * 0.5);
  //     s = Real(0.5) / s;

  //     _array[0] = Real((m(1, 2) - m(2, 1)) * s);
  //     _array[1] = Real((m(2, 0) - m(0, 2)) * s);
  //     _array[2] = Real((m(0, 1) - m(1, 0)) * s);
  //   }
  //   else
  //   {
  //     i = 0;

  //     if (m(1, 1) > m(0, 0))
  //     {
  //       i = 1;
  //     }

  //     if (m(2, 2) > m(i, i))
  //     {
  //       i = 2;
  //     }

  //     j = nxt[i];
  //     k = nxt[j];

  //     s = Real(sqrt((m(i, j) - (m(j, j) + m(k, k))) + Real(1.0)));

  //     _array[i] = Real(s * 0.5);
  //     s = Real(0.5 / s);

  //     _array[3] = Real((m(j, k) - m(k, j)) * s);
  //     _array[j] = Real((m(i, j) + m(j, i)) * s);
  //     _array[k] = Real((m(i, k) + m(k, i)) * s);
  //   }

  //   return *this;
  // }

  __host__ __device__ __forceinline__ quaternionf &set_value(const vec3r &axis, Real theta)
  {
    Real sqnorm = dot(axis, axis);

    if (sqnorm == Real(0.0))
    {
      // axis too small.
      _array.x = _array.y = _array.z = Real(0.0);
      _array.w = Real(1.0);
    }
    else
    {
      theta *= Real(0.5);
      Real sin_theta = Real(sin(theta));

      if (sqnorm != Real(1))
      {
        sin_theta /= Real(sqrt(sqnorm));
      }

      _array.x = sin_theta * axis.x;
      _array.y = sin_theta * axis.y;
      _array.z = sin_theta * axis.z;
      _array.w = Real(cos(theta));
    }

    return *this;
  }

  __host__ __device__ __forceinline__ quaternionf &set_value(const vec3r &rotateFrom, const vec3r &rotateTo)
  {
    vec3r p1, p2;
    Real alpha;

    p1 = normalize(rotateFrom);
    p2 = normalize(rotateTo);

    alpha = dot(p1, p2);
    if (alpha == Real(1.0))
    {
      *this = quaternionf();
      return *this;
    }

    // ensures that the anti-parallel case leads to a positive dot
    if (alpha == Real(-1.0))
    {
      vec3r v;

      if (p1.x != p1.y || p1.x != p1.z)
      {
        v = make_vec3r(p1.y, p1.z, p1.x);
      }
      else
      {
        v = make_vec3r(-p1.x, p1.y, p1.z);
      }

      v -= p1 * dot(p1, v);
      v = normalize(v);

      set_value(v, Real(3.1415926));
      return *this;
    }

    p1 = normalize(cross(p1, p2));

    set_value(p1, Real(acos(alpha)));

    return *this;
  }

  // __host__ __device__ __forceinline__ quaternionf &set_value(const vec3r &from_look, const vec3r &from_up,
  //                       const vec3r &to_look, const vec3r &to_up)
  // {
  //   quaternionf r_look = quaternionf(from_look, to_look);

  //   vec3r rotated_from_up(from_up);
  //   r_look.mult_vec(rotated_from_up);

  //   quaternionf r_twist = quaternionf(rotated_from_up, to_up);

  //   *this = r_twist;
  //   *this *= r_look;
  //   return *this;
  // }

  __host__ __device__ __forceinline__ quaternionf &operator*=(quaternionf &qr)
  {
    quaternionf ql(*this);
    _array.w = ql.w() * qr.w() - ql.x() * qr.x() - ql.y() * qr.y() - ql.z() * qr.z();
    _array.x = ql.w() * qr.x() + ql.x() * qr.w() + ql.y() * qr.z() - ql.z() * qr.y();
    _array.y = ql.w() * qr.y() + ql.y() * qr.w() + ql.z() * qr.x() - ql.x() * qr.z();
    _array.z = ql.w() * qr.z() + ql.z() * qr.w() + ql.x() * qr.y() - ql.y() * qr.x();
    return *this;
  }

  __host__ __device__ __forceinline__ friend quaternionf normalize(quaternionf &q)
  {
    quaternionf r(q);
    Real rnorm = Real(1.0) / Real(sqrt(q.w() * q.w() + q.x() * q.x() + q.y() * q.y() + q.z() * q.z()));

    r._array.x *= rnorm;
    r._array.y *= rnorm;
    r._array.z *= rnorm;
    r._array.w *= rnorm;
    return r;
  }

  __host__ __device__ __forceinline__ friend Real squaredNorm(quaternionf &q)
  {
    return Real(q.w() * q.w() + q.x() * q.x() + q.y() * q.y() + q.z() * q.z());
  }

  __host__ __device__ __forceinline__ friend quaternionf conjugate(const quaternionf &q)
  {
    quaternionf r(q);
    r._array.x *= Real(-1.0);
    r._array.y *= Real(-1.0);
    r._array.z *= Real(-1.0);
    return r;
  }

  __host__ __device__ __forceinline__ friend quaternionf inverse(const quaternionf &q)
  {
    return conjugate(q);
  }

  //
  // quaternionf multiplication with cartesian vector
  // v' = q*v*q(star)
  //
  __host__ __device__ __forceinline__ void mult_vec(vec3r &src, vec3r &dst)
  {
    Real v_coef = _array.w * _array.w - _array.x * _array.x - _array.y * _array.y - _array.z * _array.z;
    Real u_coef = Real(2.0) * (src.x * _array.x + src.y * _array.y + src.z * _array.z);
    Real c_coef = Real(2.0) * _array.w;

    dst.x = v_coef * src.x + u_coef * x() + c_coef * (y() * src.z - z() * src.y);
    dst.y = v_coef * src.y + u_coef * y() + c_coef * (z() * src.x - x() * src.z);
    dst.z = v_coef * src.z + u_coef * z() + c_coef * (x() * src.y - y() * src.x);
  }

  // __host__ __device__ __forceinline__ void mult_vec(vec3r &src_and_dst) const
  // {
  //   mult_vec(vec3r(src_and_dst), src_and_dst);
  // }

  __host__ __device__ __forceinline__ void scale_angle(Real scaleFactor)
  {
    vec3r axis;
    Real radians;

    get_value(axis, radians);
    radians *= scaleFactor;
    set_value(axis, radians);
  }

  __host__ __device__ __forceinline__ friend quaternionf slerp(quaternionf &p, quaternionf &q, Real alpha)
  {
    quaternionf r;

    Real cos_omega = p.x() * q.x() + p.y() * q.y() + p.z() * q.z() + p.w() * q.w();
    // if B is on opposite hemisphere from A, use -B instead

    int bflip;

    if ((bflip = (cos_omega < Real(0))))
    {
      cos_omega = -cos_omega;
    }

    // complementary interpolation parameter
    Real beta = Real(1) - alpha;

    if (cos_omega >= Real(1))
    {
      return p;
    }

    Real omega = Real(acos(cos_omega));
    Real one_over_sin_omega = Real(1.0) / Real(sin(omega));

    beta = Real(sin(omega * beta) * one_over_sin_omega);
    alpha = Real(sin(omega * alpha) * one_over_sin_omega);

    if (bflip)
    {
      alpha = -alpha;
    }

    r._array.x = beta * p._array.x + alpha * q._array.x;
    r._array.y = beta * p._array.y + alpha * q._array.y;
    r._array.z = beta * p._array.z + alpha * q._array.z;
    r._array.w = beta * p._array.w + alpha * q._array.w;
    return r;
  }

  __host__ __device__ __forceinline__ vec4r &coeffs()
  {
    return _array;
  }

  // __host__ __device__ __forceinline__ Real &operator[](int i)
  // {
  //   return _array[i];
  // }

  // __host__ __device__ __forceinline__ const Real &operator[](int i) const
  // {
  //   return _array[i];
  // }

  __host__ __device__ __forceinline__ friend bool operator==(const quaternionf &lhs, const quaternionf &rhs)
  {
    bool r = true;

    r &= lhs._array.x == rhs._array.x;
    r &= lhs._array.y == rhs._array.y;
    r &= lhs._array.z == rhs._array.z;
    r &= lhs._array.w == rhs._array.w;

    return r;
  }

  __host__ __device__ __forceinline__ friend bool operator!=(const quaternionf &lhs, const quaternionf &rhs)
  {
    return !(lhs == rhs);
  }

  __host__ __device__ __forceinline__ quaternionf &operator*=(const Real r)
  {
    quaternionf ql(*this);

    _array.w *= r;
    _array.x *= r;
    _array.y *= r;
    _array.z *= r;

    return *this;
  }

  __host__ __device__ __forceinline__ friend quaternionf operator*(quaternionf &lhs, quaternionf &rhs)
  {
    quaternionf r(lhs);
    r *= rhs;
    return r;
  }

  __host__ __device__ __forceinline__ friend quaternionf operator*(const Real lhs, const quaternionf &rhs)
  {
    quaternionf r(rhs);
    r *= lhs;
    return r;
  }

  __host__ __device__ __forceinline__ friend quaternionf operator*(const quaternionf lhs, const Real &rhs)
  {
    quaternionf r(lhs);
    r *= rhs;
    return r;
  }

  __host__ __device__ __forceinline__ quaternionf &operator/=(const Real r)
  {
    quaternionf ql(*this);

    _array.w /= r;
    _array.x /= r;
    _array.y /= r;
    _array.z /= r;

    return *this;
  }

  __host__ __device__ __forceinline__ friend quaternionf operator/(const quaternionf lhs, const Real &rhs)
  {
    quaternionf r(lhs);
    r /= rhs;
    return r;
  }

  __host__ __device__ __forceinline__ friend quaternionf operator+(const quaternionf &lhs, const quaternionf &rhs)
  {
    quaternionf r(lhs);
    r._array += rhs._array;
    return r;
  }

  __host__ __device__ __forceinline__ friend quaternionf operator-(const quaternionf &lhs, const quaternionf &rhs)
  {
    quaternionf r(lhs);
    r._array -= rhs._array;
    return r;
  }

  __host__ __device__ __forceinline__ mat3r toRotationMatrix()
  {
    mat3r res;

    Real tx = Real(2) * this->x();
    Real ty = Real(2) * this->y();
    Real tz = Real(2) * this->z();
    Real twx = tx * this->w();
    Real twy = ty * this->w();
    Real twz = tz * this->w();
    Real txx = tx * this->x();
    Real txy = ty * this->x();
    Real txz = tz * this->x();
    Real tyy = ty * this->y();
    Real tyz = tz * this->y();
    Real tzz = tz * this->z();

    res(0, 0) = Real(1) - (tyy + tzz);
    res(0, 1) = txy - twz;
    res(0, 2) = txz + twy;
    res(1, 0) = txy + twz;
    res(1, 1) = Real(1) - (txx + tzz);
    res(1, 2) = tyz - twx;
    res(2, 0) = txz - twy;
    res(2, 1) = tyz + twx;
    res(2, 2) = Real(1) - (txx + tyy);

    return res;
  }
};

#endif
