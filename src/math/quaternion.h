#pragma once

#include "math/math.h"
#include "math/vector3.h"
#include "math/matrix3x3.h"

PHYS_NAMESPACE_BEGIN

#ifdef USE_DOUBLE
#define QuaternionData QuaternionDoubleData
#define QuaternionDataName "QuaternionDoubleData"
#else
#define QuaternionData QuaternionFloatData
#define QuaternionDataName "QuaternionFloatData"
#endif


ATTRIBUTE_ALIGNED16(class)
Quaternion
{
	///Data storage for the matrix, each vector is a row of the matrix
    Real m_floats[4];
public:
	/**@brief No initialization constructor */
	Quaternion() {}

	//		template <typename Real>
	//		explicit Quaternion(const Real *v) : Tuple4<Real>(v) {}
	/**@brief Constructor from scalars */
	Quaternion(const Real& _w, const Real& _x, const Real& _y, const Real& _z)
	{
		m_floats[0] = _w;
        m_floats[1] = _x;
		m_floats[2] = _y;
		m_floats[3] = _z;
	}
	///use
	Quaternion(const Matrix3x3& m){

		Real trace = m[0][0] + m[1][1] + m[2][2];

		Real temp[4];

		if (trace > Real(0.0))
		{
			Real s = sqrtr(trace + Real(1.0));
			temp[3] = (s * Real(0.5));
			s = Real(0.5) / s;

			temp[0] = ((m[2][1] - m[1][2]) * s);//x
			temp[1] = ((m[0][2] - m[2][0]) * s);//y
			temp[2] = ((m[1][0] - m[0][1]) * s);//z
		}
		else
		{
			int i = m[0][0] < m[1][1] ? (m[1][1] < m[2][2] ? 2 : 1) : (m[0][0] < m[2][2] ? 2 : 0);
			int j = (i + 1) % 3;
			int k = (i + 2) % 3;

			Real s = sqrtr(m[i][i] - m[j][j] - m[k][k] + Real(1.0));
			temp[i] = s * Real(0.5);
			s = Real(0.5) / s;

			temp[3] = (m[k][j] - m[j][k]) * s;
			temp[j] = (m[j][i] + m[i][j]) * s;
			temp[k] = (m[k][i] + m[i][k]) * s;
		}
		setValue(temp[3], temp[0], temp[1], temp[2]);
	}

	//// used
	Matrix3x3 toRotationMatrix() const {
		Real d = this->squaredNorm();
		DEBUG_ASSERT(d != Real(0.0));
		Real s = Real(2.0) / d;

		Real xs = this->x() * s, ys = this->y() * s, zs = this->z() * s;
		Real wx = this->w() * xs, wy = this->w() * ys, wz = this->w() * zs;
		Real xx = this->x() * xs, xy = this->x() * ys, xz = this->x() * zs;
		Real yy = this->y() * ys, yz = this->y() * zs, zz = this->z() * zs;
		return Matrix3x3(
			Real(1.0) - (yy + zz), xy - wz, xz + wy,
			xy + wz, Real(1.0) - (xx + zz), yz - wx,
			xz - wy, yz + wx, Real(1.0) - (xx + yy));
	}


	/**@brief Return the x value */
	FORCE_INLINE const Real& getX() const { return m_floats[1]; }
	/**@brief Return the y value */
	FORCE_INLINE const Real& getY() const { return m_floats[2]; }
	/**@brief Return the z value */
	FORCE_INLINE const Real& getZ() const { return m_floats[3]; }
	/**@brief Set the x value */
	FORCE_INLINE void setX(Real _x) { m_floats[1] = _x; };
	/**@brief Set the y value */
	FORCE_INLINE void setY(Real _y) { m_floats[2] = _y; };
	/**@brief Set the z value */
	FORCE_INLINE void setZ(Real _z) { m_floats[3] = _z; };
	/**@brief Set the w value */
	FORCE_INLINE void setW(Real _w) { m_floats[0] = _w; };
	/**@brief Return the x value */
	FORCE_INLINE const Real& x() const { return m_floats[1]; }
	/**@brief Return the y value */
	FORCE_INLINE const Real& y() const { return m_floats[2]; }
	/**@brief Return the z value */
	FORCE_INLINE const Real& z() const { return m_floats[3]; }
	/**@brief Return the w value */
	FORCE_INLINE const Real& w() const { return m_floats[0]; }

    FORCE_INLINE bool operator==(const Quaternion& other) const
	{
		return ((m_floats[3] == other.m_floats[3]) &&
				(m_floats[2] == other.m_floats[2]) &&
				(m_floats[1] == other.m_floats[1]) &&
				(m_floats[0] == other.m_floats[0]));
	}

	FORCE_INLINE bool operator!=(const Quaternion& other) const
	{
		return !(*this == other);
	}


    /**@brief Set x,y,z and zero w 
   * @param x Value of x
   * @param y Value of y
   * @param z Value of z
   */
	FORCE_INLINE void setValue(const Real& _x, const Real& _y, const Real& _z)
	{
		m_floats[0] = 0.f;
		m_floats[1] = _x;
		m_floats[2] = _y;
		m_floats[3] = _z;
	}

    /**@brief Set the values 
   * @param x Value of x
   * @param y Value of y
   * @param z Value of z
   * @param w Value of w
   */
	FORCE_INLINE void setValue(const Real& _w, const Real& _x, const Real& _y, const Real& _z)
	{
		m_floats[0] = _w;
		m_floats[1] = _x;
		m_floats[2] = _y;
		m_floats[3] = _z;
	}

	/**@brief Add two quaternions
   * @param q The quaternion to add to this one */
	FORCE_INLINE Quaternion& operator+=(const Quaternion& q)
	{
		m_floats[0] += q.w();
		m_floats[1] += q.x();
		m_floats[2] += q.y();
		m_floats[3] += q.z();
		return *this;
	}

	/**@brief Subtract out a quaternion
   * @param q The quaternion to subtract from this one */
	Quaternion& operator-=(const Quaternion& q)
	{
		m_floats[0] -= q.w();
		m_floats[1] -= q.x();
		m_floats[2] -= q.y();
		m_floats[3] -= q.z();
		return *this;
	}

	/**@brief Scale this quaternion
   * @param s The scalar to scale by */
	Quaternion& operator*=(const Real& s)
	{
		m_floats[0] *= s;
		m_floats[1] *= s;
		m_floats[2] *= s;
		m_floats[3] *= s;
		return *this;
	}

	/**@brief Multiply this quaternion by q on the right
   * @param q The other quaternion 
   * Equivilant to this = this * q */
	Quaternion& operator*=(const Quaternion& q)
	{
		setValue(
			// m_floats[3] * q.x() + m_floats[0] * q.m_floats[3] + m_floats[1] * q.z() - m_floats[2] * q.y(),
			// m_floats[3] * q.y() + m_floats[1] * q.m_floats[3] + m_floats[2] * q.x() - m_floats[0] * q.z(),
			// m_floats[3] * q.z() + m_floats[2] * q.m_floats[3] + m_floats[0] * q.y() - m_floats[1] * q.x(),
			// m_floats[3] * q.m_floats[3] - m_floats[0] * q.x() - m_floats[1] * q.y() - m_floats[2] * q.z());
			m_floats[0] * q.m_floats[0] - m_floats[1] * q.x() - m_floats[2] * q.y() - m_floats[3] * q.z(),
			m_floats[0] * q.x() + m_floats[1] * q.m_floats[0] + m_floats[2] * q.z() - m_floats[3] * q.y(),
			m_floats[0] * q.y() + m_floats[2] * q.m_floats[0] + m_floats[3] * q.x() - m_floats[1] * q.z(),
			m_floats[0] * q.z() + m_floats[3] * q.m_floats[0] + m_floats[1] * q.y() - m_floats[2] * q.x());
		return *this;
	}
	/**@brief Return the dot product between this quaternion and another
   * @param q The other quaternion */
	Real dot(const Quaternion& q) const
	{
		return m_floats[0] * q.m_floats[0] +
			   m_floats[1] * q.m_floats[1] +
			   m_floats[2] * q.m_floats[2] +
			   m_floats[3] * q.m_floats[3];
	}

	/**@brief Return the length squared of the quaternion */
	Real squaredNorm() const
	{
		return dot(*this);
	}

	/**@brief Return the length of the quaternion */
	Real norm() const
	{
		return sqrtr(squaredNorm());
	}
	Quaternion& safeNormalize()
	{
		Real l2 = squaredNorm();
		if (l2 > REAL_EPSILON)
		{
			normalize();
		}
		return *this;
	}
	/**@brief Normalize the quaternion 
   * Such that x^2 + y^2 + z^2 +w^2 = 1 */
	Quaternion& normalize()
	{
		return *this /= norm();
	}

	/**@brief Return a scaled version of this quaternion
   * @param s The scale factor */
	FORCE_INLINE Quaternion
	operator*(const Real& s) const
	{
		return Quaternion(m_floats[0] * s, m_floats[1] * s, m_floats[2] * s, m_floats[3] * s);
	}

	/**@brief Return an inversely scaled versionof this quaternion
   * @param s The inverse scale factor */
	Quaternion operator/(const Real& s) const
	{
		ASSERT(s != Real(0.0));
		return *this * (Real(1.0) / s);
	}

	/**@brief Inversely scale this quaternion
   * @param s The scale factor */
	Quaternion& operator/=(const Real& s)
	{
		ASSERT(s != Real(0.0));
		return *this *= Real(1.0) / s;
	}

	/**@brief Return a normalized version of this quaternion */
	Quaternion normalized() const
	{
		return *this / norm();
	}


	/**@brief Return the inverse of this quaternion */
	Quaternion inverse() const
	{
		return Quaternion(-m_floats[0], -m_floats[1], -m_floats[2], m_floats[3]);
	}

	/**@brief Return the sum of this quaternion and the other 
   * @param q2 The other quaternion */
	FORCE_INLINE Quaternion
	operator+(const Quaternion& q2) const
	{
		const Quaternion& q1 = *this;
		return Quaternion(q1.m_floats[0] + q2.m_floats[0], q1.m_floats[1] + q2.m_floats[1], q1.m_floats[2] + q2.m_floats[2], q1.m_floats[3] + q2.m_floats[3]);
	}

	/**@brief Return the difference between this quaternion and the other 
   * @param q2 The other quaternion */
	FORCE_INLINE Quaternion
	operator-(const Quaternion& q2) const
	{
		const Quaternion& q1 = *this;
		return Quaternion(q1.m_floats[0] - q2.m_floats[0], q1.m_floats[1] - q2.m_floats[1], q1.m_floats[2] - q2.m_floats[2], q1.m_floats[3] - q2.m_floats[3]);
	}

	/**@brief Return the negative of this quaternion 
   * This simply negates each element */
	FORCE_INLINE Quaternion operator-() const
	{
		const Quaternion& q2 = *this;
		return Quaternion(-q2.m_floats[0], -q2.m_floats[1], -q2.m_floats[2], -q2.m_floats[3]);
	}

	static const Quaternion& Identity()
	{
		static const Quaternion identityQuat(Real(1.), Real(0.), Real(0.), Real(0.));
		return identityQuat;
	}
	
	friend std::ostream& operator<<(std::ostream&o,const Quaternion&q){
		o << "(" << q.m_floats[0] <<", "<< q.m_floats[1] <<", " << q.m_floats[2]<<", " << q.m_floats[3]<<")";
		return o;	
	}

	FORCE_INLINE const Real& getW() const { return m_floats[0]; }
};

/**@brief Return the product of two quaternions */
FORCE_INLINE Quaternion
operator*(const Quaternion& q1, const Quaternion& q2)
{
	return Quaternion(
		q1.w() * q2.w() - q1.x() * q2.x() - q1.y() * q2.y() - q1.z() * q2.z(),
		q1.w() * q2.x() + q1.x() * q2.w() + q1.y() * q2.z() - q1.z() * q2.y(),
		q1.w() * q2.y() + q1.y() * q2.w() + q1.z() * q2.x() - q1.x() * q2.z(),
		q1.w() * q2.z() + q1.z() * q2.w() + q1.x() * q2.y() - q1.y() * q2.x()
		);
}

FORCE_INLINE Quaternion
operator*(const Quaternion& q, const Vector3& w)
{

	return Quaternion(
		-q.x() * w.x() - q.y() * w.y() - q.z() * w.z(),
		q.w() * w.x() + q.y() * w.z() - q.z() * w.y(),
		q.w() * w.y() + q.z() * w.x() - q.x() * w.z(),
		q.w() * w.z() + q.x() * w.y() - q.y() * w.x());
}

FORCE_INLINE Quaternion
operator*(const Vector3& w, const Quaternion& q)
{
	return Quaternion(
		-w.x() * q.x() - w.y() * q.y() - w.z() * q.z(),
		+w.x() * q.w() + w.y() * q.z() - w.z() * q.y(),
		+w.y() * q.w() + w.z() * q.x() - w.x() * q.z(),
		+w.z() * q.w() + w.x() * q.y() - w.y() * q.x());
}

/**@brief Calculate the dot product between two quaternions */
FORCE_INLINE Real
dot(const Quaternion& q1, const Quaternion& q2)
{
	return q1.dot(q2);
}

/**@brief Return the length of a quaternion */
FORCE_INLINE Real
length(const Quaternion& q)
{
	return q.norm();
}

/**@brief Return the inverse of a quaternion*/
FORCE_INLINE Quaternion
inverse(const Quaternion& q)
{
	return q.inverse();
}


PHYS_NAMESPACE_END