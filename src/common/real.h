#pragma once

#include <float.h>
#include <math.h>
#include "general.h"

//// real number
#if PE_USE_BACKEND_WRAPPER
	#include <linear_math/common.h>
	using Real = linear_math::Real;
#else
	#if defined(USE_DOUBLE)
		typedef double Real;
	#else
		typedef float Real;
	#endif
#endif

//// const number
#define REAL_PI Real(3.1415926535897932384626433832795029)
#define REAL_2_PI (Real(2.0) * REAL_PI)
#define REAL_HALF_PI (REAL_PI * Real(0.5))
#define REAL_RADS_PER_DEG (REAL_2_PI / Real(360.0))
#define REAL_DEGS_PER_RAD (Real(360.0) / REAL_2_PI)

//// const number
#ifdef USE_DOUBLE
	#define REAL_EPSILON DBL_EPSILON
	#define REAL_INFINITY DBL_MAX
	#define REAL_ONE 1.0
	#define REAL_ZERO 0.0
	#define REAL_TWO 2.0
	#define REAL_HALF 0.5
	#define REAL_LARGE_FLOAT 1e30

#else
	#define REAL_EPSILON FLT_EPSILON
	#define REAL_INFINITY FLT_MAX
	#define REAL_ONE 1.0f
	#define REAL_ZERO 0.0f
	#define REAL_TWO 2.0f
	#define REAL_HALF 0.5f
	#define REAL_LARGE_FLOAT 1e18f
#endif

//// math functions
#if defined(USE_DOUBLE)
////double
    FORCE_INLINE Real fabsr(Real x) { return fabs(x); }
	FORCE_INLINE Real cosr(Real x) { return cos(x); }
	FORCE_INLINE Real sinr(Real x) { return sin(x); }
	FORCE_INLINE Real tanr(Real x) { return tan(x); }
	FORCE_INLINE Real acosr(Real x)
	{
		if (x < Real(-1))
			x = Real(-1);
		if (x > Real(1))
			x = Real(1);
		return acos(x);
	}
	FORCE_INLINE Real asinr(Real x)
	{
		if (x < Real(-1))
			x = Real(-1);
		if (x > Real(1))
			x = Real(1);
		return asin(x);
	}
	FORCE_INLINE Real atanr(Real x) { return atan(x); }
	FORCE_INLINE Real atan2r(Real x, Real y) { return atan2(x, y); }
	FORCE_INLINE Real expr(Real x) { return exp(x); }
	FORCE_INLINE Real logr(Real x) { return log(x); }
	FORCE_INLINE Real powr(Real x, Real y) { return pow(x, y); }
	FORCE_INLINE Real fmodr(Real x, Real y) { return fmod(x, y); }

#else
//// float
    FORCE_INLINE Real fabsr(Real x) { return fabsf(x); }
	FORCE_INLINE Real cosr(Real x) { return cosf(x); }
	FORCE_INLINE Real sinr(Real x) { return sinf(x); }
	FORCE_INLINE Real tanr(Real x) { return tanf(x); }
	FORCE_INLINE Real acosr(Real x)
	{
		if (x < Real(-1))
			x = Real(-1);
		if (x > Real(1))
			x = Real(1);
		return acosf(x);
	}
	FORCE_INLINE Real asinr(Real x)
	{
		if (x < Real(-1))
			x = Real(-1);
		if (x > Real(1))
			x = Real(1);
		return asinf(x);
	}
	FORCE_INLINE Real atanr(Real x) { return atanf(x); }
	FORCE_INLINE Real atan2r(Real x, Real y) { return atan2f(x, y); }
	FORCE_INLINE Real expr(Real x) { return expf(x); }
	FORCE_INLINE Real logr(Real x) { return logf(x); }
	FORCE_INLINE Real powr(Real x, Real y) { return powf(x, y); }
	FORCE_INLINE Real fmodr(Real x, Real y) { return fmodf(x, y); }

#endif


FORCE_INLINE Real sqrtr(Real y){
#if defined(USE_DOUBLE)
    return sqrt(y);
#else
    return sqrtf(y);
#endif
}

template <typename T>
inline T squared(T v) {
    return v * v;
}

