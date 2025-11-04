#pragma once

#include <iostream>
#include "math/math.h"

PHYS_NAMESPACE_BEGIN

#ifdef USE_DOUBLE
#define Vector3Data Vector3DoubleData
#define Vector3DataName "Vector3DoubleData"
#else
#define Vector3Data Vector3FloatData
#define Vector3DataName "Vector3FloatData"
#endif  //USE_DOUBLE


ATTRIBUTE_ALIGNED16(class)
Vector3
{
public:

	union InnerData {
    	Real reals[4];
		struct XYZ{
			Real x;
			Real y;
			Real z;
			Real w;
		}xyz;
	} m_data;
    
    FORCE_INLINE Vector3()
	{
	}

	/**@brief Constructor from scalars 
     * @param x X value
     * @param y Y value 
     * @param z Z value 
     */
	FORCE_INLINE Vector3(const Real& _x, const Real& _y, const Real& _z)
	{
		m_data.xyz.x = _x;
		m_data.xyz.y = _y;
		m_data.xyz.z = _z;
		m_data.xyz.w = Real(0.f);
	}


	// Copy constructor
	FORCE_INLINE Vector3(const Vector3& rhs)
	{
        // m_data.xyz.x = rhs.m_data.xyz.x;
		// m_data.xyz.y = rhs.m_data.xyz.y;
		// m_data.xyz.z = rhs.m_data.xyz.z;
		// m_data.xyz.w = rhs.m_data.xyz.w;
		m_data = rhs.m_data;
	}

	// Assignment Operator
	FORCE_INLINE Vector3&
	operator=(const Vector3& v)
	{
		// m_data.xyz.x = v.m_data.xyz.x;
		// m_data.xyz.y = v.m_data.xyz.y;
		// m_data.xyz.z = v.m_data.xyz.z;
		// m_data.xyz.w = v.m_data.xyz.w;
		m_data = v.m_data;

		return *this;
	}

    /**@brief Add a vector to this one 
     * @param The vector to add to this one */
	FORCE_INLINE Vector3& operator+=(const Vector3& v)
	{
		m_data.xyz.x += v.m_data.xyz.x;
		m_data.xyz.y += v.m_data.xyz.y;
		m_data.xyz.z += v.m_data.xyz.z;

		return *this;
	}

	/**@brief Subtract a vector from this one
     * @param The vector to subtract */
	FORCE_INLINE Vector3& operator-=(const Vector3& v)
	{
		m_data.xyz.x -= v.m_data.xyz.x;
		m_data.xyz.y -= v.m_data.xyz.y;
		m_data.xyz.z -= v.m_data.xyz.z;
		return *this;
	}

	/**@brief Scale the vector
     * @param s Scale factor */
	FORCE_INLINE Vector3& operator*=(const Real& s)
	{
		m_data.xyz.x *= s;
		m_data.xyz.y *= s;
		m_data.xyz.z *= s;
		return *this;
	}

	/**@brief Inversely scale the vector 
     * @param s Scale factor to divide by */
	FORCE_INLINE Vector3& operator/=(const Real& s)
	{
		DEBUG_ASSERT(s != Real(0.0));

		return *this *= Real(1.0) / s;
	}

	/**@brief Return the dot product
     * @param v The other vector in the dot product */
	FORCE_INLINE Real dot(const Vector3& v) const
	{
		return m_data.xyz.x * v.m_data.xyz.x +
			   m_data.xyz.y * v.m_data.xyz.y +
			   m_data.xyz.z * v.m_data.xyz.z;
	}

	/**@brief Return the norm (length) of the vector */
	FORCE_INLINE Real norm() const
	{
		return sqrtr(squaredNorm());
	}

	/**@brief Return the squaredNorm (length2) of the vector */
	FORCE_INLINE Real squaredNorm() const
	{
		return dot(*this);
	}

	/**@brief Normalize this vector 
     * x^2 + y^2 + z^2 = 1 */
	FORCE_INLINE Vector3& normalize()
	{
		ASSERT(!fuzzyZero());
		return *this /= norm();
	}

 	/**@brief Return a normalized version of this vector */
	FORCE_INLINE Vector3 normalized() const {
		Vector3 nrm = *this;
		return nrm.normalize();
	}

	/**@brief Return the angle between this and another vector
   * @param v The other vector */
	FORCE_INLINE Real angle(const Vector3& v) const
	{
		Real s = sqrtr(squaredNorm() * v.squaredNorm());
		DEBUG_ASSERT(s != Real(0.0));
		return acosr(dot(v) / s);
	}

	FORCE_INLINE Vector3 transpose() const {
		return *this;
	}

	/**@brief Return the cross product between this and another vector 
   * @param v The other vector */
	FORCE_INLINE Vector3 cross(const Vector3& v) const
	{
		return Vector3(
			m_data.xyz.y * v.m_data.xyz.z - m_data.xyz.z * v.m_data.xyz.y,
			m_data.xyz.z * v.m_data.xyz.x - m_data.xyz.x * v.m_data.xyz.z,
			m_data.xyz.x * v.m_data.xyz.y - m_data.xyz.y * v.m_data.xyz.x);
	}

	FORCE_INLINE Real triple(const Vector3& v1, const Vector3& v2) const
	{
		return m_data.xyz.x * (v1.m_data.xyz.y * v2.m_data.xyz.z - v1.m_data.xyz.z * v2.m_data.xyz.y) +
			   m_data.xyz.y * (v1.m_data.xyz.z * v2.m_data.xyz.x - v1.m_data.xyz.x * v2.m_data.xyz.z) +
			   m_data.xyz.z * (v1.m_data.xyz.x * v2.m_data.xyz.y - v1.m_data.xyz.y * v2.m_data.xyz.x);
	}

	/**@brief Return the axis with the smallest value 
   * Note return values are 0,1,2 for x, y, or z */
	FORCE_INLINE int minAxis() const
	{
		return m_data.xyz.x < m_data.xyz.y ? (m_data.xyz.x < m_data.xyz.z ? 0 : 2) : (m_data.xyz.y < m_data.xyz.z ? 1 : 2);
	}

	/**@brief Return the axis with the largest value 
   * Note return values are 0,1,2 for x, y, or z */
	FORCE_INLINE int maxAxis() const
	{
		return m_data.xyz.x < m_data.xyz.y ? (m_data.xyz.y < m_data.xyz.z ? 2 : 1) : (m_data.xyz.x < m_data.xyz.z ? 2 : 0);
	}

	// FORCE_INLINE int furthestAxis() const
	// {
	// 	return absolute().minAxis();
	// }

	// FORCE_INLINE int closestAxis() const
	// {
	// 	return absolute().maxAxis();
	// }

// 	FORCE_INLINE void setInterpolate3(const Vector3& v0, const Vector3& v1, Real rt)
// 	{
// #if defined(BT_USE_SSE_IN_API) && defined(BT_USE_SSE)
// 		__m128 vrt = _mm_load_ss(&rt);  //	(rt 0 0 0)
// 		Real s = Real(1.0) - rt;
// 		__m128 vs = _mm_load_ss(&s);  //	(S 0 0 0)
// 		vs = bt_pshufd_ps(vs, 0x80);  //	(S S S 0.0)
// 		__m128 r0 = _mm_mul_ps(v0.mVec128, vs);
// 		vrt = bt_pshufd_ps(vrt, 0x80);  //	(rt rt rt 0.0)
// 		__m128 r1 = _mm_mul_ps(v1.mVec128, vrt);
// 		__m128 tmp3 = _mm_add_ps(r0, r1);
// 		mVec128 = tmp3;
// #elif defined(BT_USE_NEON)
// 		float32x4_t vl = vsubq_f32(v1.mVec128, v0.mVec128);
// 		vl = vmulq_n_f32(vl, rt);
// 		mVec128 = vaddq_f32(vl, v0.mVec128);
// #else
// 		Real s = Real(1.0) - rt;
// 		m_data.xyz.x = s * v0.m_data.xyz.x + rt * v1.m_data.xyz.x;
// 		m_data.xyz.y = s * v0.m_data.xyz.y + rt * v1.m_data.xyz.y;
// 		m_data.xyz.z = s * v0.m_data.xyz.z + rt * v1.m_data.xyz.z;
// 		//don't do the unused w component
// 		//		m_co[3] = s * v0[3] + rt * v1[3];
// #endif
// 	}

// 	/**@brief Return the linear interpolation between this and another vector 
//    * @param v The other vector 
//    * @param t The ration of this to v (t = 0 => return this, t=1 => return other) */
// 	FORCE_INLINE Vector3 lerp(const Vector3& v, const Real& t) const
// 	{
// #if defined(BT_USE_SSE_IN_API) && defined(BT_USE_SSE)
// 		__m128 vt = _mm_load_ss(&t);  //	(t 0 0 0)
// 		vt = bt_pshufd_ps(vt, 0x80);  //	(rt rt rt 0.0)
// 		__m128 vl = _mm_sub_ps(v.mVec128, mVec128);
// 		vl = _mm_mul_ps(vl, vt);
// 		vl = _mm_add_ps(vl, mVec128);

// 		return Vector3(vl);
// #elif defined(BT_USE_NEON)
// 		float32x4_t vl = vsubq_f32(v.mVec128, mVec128);
// 		vl = vmulq_n_f32(vl, t);
// 		vl = vaddq_f32(vl, mVec128);

// 		return Vector3(vl);
// #else
// 		return Vector3(m_data.xyz.x + (v.m_data.xyz.x - m_data.xyz.x) * t,
// 						 m_data.xyz.y + (v.m_data.xyz.y - m_data.xyz.y) * t,
// 						 m_data.xyz.z + (v.m_data.xyz.z - m_data.xyz.z) * t);
// #endif
// 	}

	/**@brief Elementwise multiply this vector by the other 
   * @param v The other vector */
	FORCE_INLINE Vector3& operator*=(const Vector3& v)
	{
		m_data.xyz.x *= v.m_data.xyz.x;
		m_data.xyz.y *= v.m_data.xyz.y;
		m_data.xyz.z *= v.m_data.xyz.z;
		return *this;
	}

	/**@brief Return the x value */
	FORCE_INLINE const Real& getX() const { return m_data.xyz.x; }
	/**@brief Return the y value */
	FORCE_INLINE const Real& getY() const { return m_data.xyz.y; }
	/**@brief Return the z value */
	FORCE_INLINE const Real& getZ() const { return m_data.xyz.z; }
	/**@brief Set the x value */
	FORCE_INLINE void setX(Real _x) { m_data.xyz.x = _x; };
	/**@brief Set the y value */
	FORCE_INLINE void setY(Real _y) { m_data.xyz.y = _y; };
	/**@brief Set the z value */
	FORCE_INLINE void setZ(Real _z) { m_data.xyz.z = _z; };
	/**@brief Set the w value */
	FORCE_INLINE void setW(Real _w) { m_data.xyz.w = _w; };
	/**@brief Return the x value */
	FORCE_INLINE const Real& x() const { return m_data.xyz.x; }
	/**@brief Return the y value */
	FORCE_INLINE const Real& y() const { return m_data.xyz.y; }
	/**@brief Return the z value */
	FORCE_INLINE const Real& z() const { return m_data.xyz.z; }
	/**@brief Return the w value */
	FORCE_INLINE const Real& w() const { return m_data.xyz.w; }

	// FORCE_INLINE Real&       operator()(int i)       { return (&m_data.xyz.x)[i];	}
	// FORCE_INLINE const Real& operator()(int i) const { return (&m_data.xyz.x)[i]; }

	FORCE_INLINE Real& operator()(int i){ return m_data.reals[i]; }
	FORCE_INLINE const Real& operator()(int i) const{ return m_data.reals[i]; }

	///operator Real*() replaces operator[], using implicit conversion. We added operator != and operator == to avoid pointer comparisons.
	FORCE_INLINE operator Real*() { return &m_data.xyz.x; }
	FORCE_INLINE operator const Real*() const { return &m_data.xyz.x; }

	FORCE_INLINE bool operator==(const Vector3& other) const
	{
		return ((m_data.xyz.w == other.m_data.xyz.w) &&
				(m_data.xyz.z == other.m_data.xyz.z) &&
				(m_data.xyz.y == other.m_data.xyz.y) &&
				(m_data.xyz.x == other.m_data.xyz.x));
	}

	FORCE_INLINE bool operator!=(const Vector3& other) const
	{
		return !(*this == other);
	}

// 	/**@brief Set each element to the max of the current values and the values of another Vector3
//    * @param other The other Vector3 to compare with 
//    */
// 	FORCE_INLINE void setMax(const Vector3& other)
// 	{
// #if defined(BT_USE_SSE_IN_API) && defined(BT_USE_SSE)
// 		mVec128 = _mm_max_ps(mVec128, other.mVec128);
// #elif defined(BT_USE_NEON)
// 		mVec128 = vmaxq_f32(mVec128, other.mVec128);
// #else
// 		btSetMax(m_data.xyz.x, other.m_data.xyz.x);
// 		btSetMax(m_data.xyz.y, other.m_data.xyz.y);
// 		btSetMax(m_data.xyz.z, other.m_data.xyz.z);
// 		btSetMax(m_data.xyz.w, other.w());
// #endif
// 	}

// 	/**@brief Set each element to the min of the current values and the values of another Vector3
//    * @param other The other Vector3 to compare with 
//    */
// 	FORCE_INLINE void setMin(const Vector3& other)
// 	{
// #if defined(BT_USE_SSE_IN_API) && defined(BT_USE_SSE)
// 		mVec128 = _mm_min_ps(mVec128, other.mVec128);
// #elif defined(BT_USE_NEON)
// 		mVec128 = vminq_f32(mVec128, other.mVec128);
// #else
// 		btSetMin(m_data.xyz.x, other.m_data.xyz.x);
// 		btSetMin(m_data.xyz.y, other.m_data.xyz.y);
// 		btSetMin(m_data.xyz.z, other.m_data.xyz.z);
// 		btSetMin(m_data.xyz.w, other.w());
// #endif
// 	}

	FORCE_INLINE void setValue(const Real& _x, const Real& _y, const Real& _z)
	{
		m_data.xyz.x = _x;
		m_data.xyz.y = _y;
		m_data.xyz.z = _z;
		m_data.xyz.w = Real(0.f);
	}

// 	void getSkewSymmetricMatrix(Vector3 * v0, Vector3 * v1, Vector3 * v2) const
// 	{
// #if defined BT_USE_SIMD_VECTOR3 && defined(BT_USE_SSE_IN_API) && defined(BT_USE_SSE)

// 		__m128 V = _mm_and_ps(mVec128, btvFFF0fMask);
// 		__m128 V0 = _mm_xor_ps(btvMzeroMask, V);
// 		__m128 V2 = _mm_movelh_ps(V0, V);

// 		__m128 V1 = _mm_shuffle_ps(V, V0, 0xCE);

// 		V0 = _mm_shuffle_ps(V0, V, 0xDB);
// 		V2 = _mm_shuffle_ps(V2, V, 0xF9);

// 		v0->mVec128 = V0;
// 		v1->mVec128 = V1;
// 		v2->mVec128 = V2;
// #else
// 		v0->setValue(0., -z(), y());
// 		v1->setValue(z(), 0., -x());
// 		v2->setValue(-y(), x(), 0.);
// #endif
// 	}

	void setZero()
	{
		setValue(Real(0.), Real(0.), Real(0.));
	}
	
	static const Vector3& Zero()
	{
		static const Vector3 zeroVector((Real)0.0, (Real)0.0, (Real)0.0);
		return zeroVector;
	}

	static const Vector3& Unit(int i)
	{
		static const Vector3 unitVectors[3] = {
			Vector3((Real)1.0, (Real)0.0, (Real)0.0),
			Vector3((Real)0.0, (Real)1.0, (Real)0.0),
			Vector3((Real)0.0, (Real)0.0, (Real)1.0)
		};
		return unitVectors[i];
	}
	/** 
     * @brief Compute the tangent space, given a normal vector.
     * The implementation is mainly based on bullet3 now.
     * (bullet3/src/Bullet3Common/b3Vector.h/b3PlaneSpace1())
     */
	static void getOrthoUnits(const Vector3 v, Vector3 &p, Vector3 &q){
		const Vector3 n = v.normalized();
        if (fabs(n[2]) > 0.707)
        {
            // choose p in y-z plane
            Real a = n[1] * n[1] + n[2] * n[2];
            Real k = (Real)1.0/sqrtf(a);
            p[0] = 0;
            p[1] = -n[2] * k;
            p[2] = n[1] * k;
            // set q = n x p
            q[0] = a * k;
            q[1] = -n[0] * p[2];
            q[2] = n[0] * p[1];
        }
        else
        {
            // choose p in x-y plane
            Real a = n[0] * n[0] + n[1] * n[1];
            Real k = (Real)1.0/sqrtf(a);
            p[0] = -n[1] * k;
            p[1] = n[0] * k;
            p[2] = 0;
            // set q = n x p
            q[0] = -n[2] * p[1];
            q[1] = n[2] * p[0];
            q[2] = a * k;
        }
	}

	FORCE_INLINE bool isZero() const
	{
		return m_data.xyz.x == Real(0) && m_data.xyz.y == Real(0) && m_data.xyz.z == Real(0);
	}

	FORCE_INLINE bool fuzzyZero() const
	{
		return squaredNorm() < REAL_EPSILON * REAL_EPSILON;
	}

	FORCE_INLINE void serialize(struct Vector3Data & dataOut) const {/* TODO */}

	FORCE_INLINE void deSerialize(const struct Vector3DoubleData& dataIn) {/* TODO */}

	FORCE_INLINE void deSerialize(const struct Vector3FloatData& dataIn) {/* TODO */}

	FORCE_INLINE void serializeFloat(struct Vector3FloatData & dataOut) const {/* TODO */}

	FORCE_INLINE void deSerializeFloat(const struct Vector3FloatData& dataIn) {/* TODO */}

	FORCE_INLINE void serializeDouble(struct Vector3DoubleData & dataOut) const {/* TODO */}

	FORCE_INLINE void deSerializeDouble(const struct Vector3DoubleData& dataIn) {/* TODO */}

	/**@brief returns index of maximum dot product between this and vectors in array[]
         * @param array The other vectors 
         * @param array_count The number of other vectors 
         * @param dotOut The maximum dot product */
	FORCE_INLINE long maxDot(const Vector3* array, long array_count, Real& dotOut) const;

	/**@brief returns index of minimum dot product between this and vectors in array[]
         * @param array The other vectors 
         * @param array_count The number of other vectors 
         * @param dotOut The minimum dot product */
	FORCE_INLINE long minDot(const Vector3* array, long array_count, Real& dotOut) const;

	/* create a vector as  Vector3( this->dot( Vector3 v0 ), this->dot( Vector3 v1), this->dot( Vector3 v2 ))  */
	FORCE_INLINE Vector3 dot3(const Vector3& v0, const Vector3& v1, const Vector3& v2) const
	{
		return Vector3(dot(v0), dot(v1), dot(v2));
	}

	friend std::ostream& operator<<(std::ostream&o,const Vector3&v){
		o << "(" << v.x() <<", "<< v.y() <<", " << v.z()<<")";
		return o;	
	}
};

struct Vector3FloatData
{
	float m_floats[4];
};

struct Vector3DoubleData
{
	double m_floats[4];
};


/**@brief Return the sum of two vectors (Point symantics)*/
FORCE_INLINE Vector3
operator+(const Vector3& v1, const Vector3& v2)
{
	return Vector3(
		v1.m_data.xyz.x + v2.m_data.xyz.x,
		v1.m_data.xyz.y + v2.m_data.xyz.y,
		v1.m_data.xyz.z + v2.m_data.xyz.z);
}

/**@brief Return the elementwise product of two vectors */
FORCE_INLINE Vector3
operator*(const Vector3& v1, const Vector3& v2)
{
	return Vector3(
		v1.m_data.xyz.x * v2.m_data.xyz.x,
		v1.m_data.xyz.y * v2.m_data.xyz.y,
		v1.m_data.xyz.z * v2.m_data.xyz.z);
}

/**@brief Return the difference between two vectors */
FORCE_INLINE Vector3
operator-(const Vector3& v1, const Vector3& v2)
{
	return Vector3(
		v1.m_data.xyz.x - v2.m_data.xyz.x,
		v1.m_data.xyz.y - v2.m_data.xyz.y,
		v1.m_data.xyz.z - v2.m_data.xyz.z);
}

/**@brief Return the negative of the vector */
FORCE_INLINE Vector3
operator-(const Vector3& v)
{
	return Vector3(-v.m_data.xyz.x, -v.m_data.xyz.y, -v.m_data.xyz.z);
}

/**@brief Return the vector scaled by s */
FORCE_INLINE Vector3
operator*(const Vector3& v, const Real& s)
{
	return Vector3(v.m_data.xyz.x * s, v.m_data.xyz.y * s, v.m_data.xyz.z * s);
}

/**@brief Return the vector scaled by s */
FORCE_INLINE Vector3
operator*(const Real& s, const Vector3& v)
{
	return v * s;
}

/**@brief Return the vector inversely scaled by s */
FORCE_INLINE Vector3
operator/(const Vector3& v, const Real& s)
{
	DEBUG_ASSERT(s != Real(0.0));
	return v * (Real(1.0) / s);
}

/**@brief Return the vector inversely scaled by s */
FORCE_INLINE Vector3
operator/(const Vector3& v1, const Vector3& v2)
{
	return Vector3(
		v1.m_data.xyz.x / v2.m_data.xyz.x,
		v1.m_data.xyz.y / v2.m_data.xyz.y,
		v1.m_data.xyz.z / v2.m_data.xyz.z);
}

/**@brief Return the dot product between two vectors */
FORCE_INLINE Real
dot(const Vector3& v1, const Vector3& v2)
{
	return v1.dot(v2);
}

/**@brief Return the angle between two vectors */
FORCE_INLINE Real
angle(const Vector3& v1, const Vector3& v2)
{
	return v1.angle(v2);
}

/**@brief Return the cross product of two vectors */
FORCE_INLINE Vector3
cross(const Vector3& v1, const Vector3& v2)
{
	return v1.cross(v2);
}

FORCE_INLINE Real
triple(const Vector3& v1, const Vector3& v2, const Vector3& v3)
{
	return v1.triple(v2, v3);
}

PHYS_NAMESPACE_END
