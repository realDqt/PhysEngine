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
	Quaternion(const Real& _x, const Real& _y, const Real& _z, const Real& _w)
	{
        m_floats[0] = _x;
		m_floats[1] = _y;
		m_floats[2] = _z;
		m_floats[3] = _w;
	}
	/**@brief Axis angle Constructor
   * @param axis The axis which the rotation is around
   * @param angle The magnitude of the rotation around the angle (Radians) */
	Quaternion(const Vector3& _axis, const Real& _angle)
	{
		setRotation(_axis, _angle);
	}
	/**@brief Constructor from Euler angles
   * @param yaw Angle around Y unless BT_EULER_DEFAULT_ZYX defined then Z
   * @param pitch Angle around X unless BT_EULER_DEFAULT_ZYX defined then Y
   * @param roll Angle around Z unless BT_EULER_DEFAULT_ZYX defined then X */
	Quaternion(const Real& yaw, const Real& pitch, const Real& roll)
	{
		setEuler(yaw, pitch, roll);
	}

	Quaternion(const Matrix3x3& m){

		Real trace = m[0][0] + m[1][1] + m[2][2];

		Real temp[4];

		if (trace > Real(0.0))
		{
			Real s = sqrtr(trace + Real(1.0));
			temp[3] = (s * Real(0.5));
			s = Real(0.5) / s;

			temp[0] = ((m[2][1] - m[1][2]) * s);
			temp[1] = ((m[0][2] - m[2][0]) * s);
			temp[2] = ((m[1][0] - m[0][1]) * s);
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
		setValue(temp[0], temp[1], temp[2], temp[3]);
	}




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
	FORCE_INLINE const Real& getX() const { return m_floats[0]; }
	/**@brief Return the y value */
	FORCE_INLINE const Real& getY() const { return m_floats[1]; }
	/**@brief Return the z value */
	FORCE_INLINE const Real& getZ() const { return m_floats[2]; }
	/**@brief Set the x value */
	FORCE_INLINE void setX(Real _x) { m_floats[0] = _x; };
	/**@brief Set the y value */
	FORCE_INLINE void setY(Real _y) { m_floats[1] = _y; };
	/**@brief Set the z value */
	FORCE_INLINE void setZ(Real _z) { m_floats[2] = _z; };
	/**@brief Set the w value */
	FORCE_INLINE void setW(Real _w) { m_floats[3] = _w; };
	/**@brief Return the x value */
	FORCE_INLINE const Real& x() const { return m_floats[0]; }
	/**@brief Return the y value */
	FORCE_INLINE const Real& y() const { return m_floats[1]; }
	/**@brief Return the z value */
	FORCE_INLINE const Real& z() const { return m_floats[2]; }
	/**@brief Return the w value */
	FORCE_INLINE const Real& w() const { return m_floats[3]; }

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
		m_floats[0] = _x;
		m_floats[1] = _y;
		m_floats[2] = _z;
		m_floats[3] = 0.f;
	}

    /**@brief Set the values 
   * @param x Value of x
   * @param y Value of y
   * @param z Value of z
   * @param w Value of w
   */
	FORCE_INLINE void setValue(const Real& _x, const Real& _y, const Real& _z, const Real& _w)
	{
		m_floats[0] = _x;
		m_floats[1] = _y;
		m_floats[2] = _z;
		m_floats[3] = _w;
	}



	/**@brief Set the rotation using axis angle notation 
   * @param axis The axis around which to rotate
   * @param angle The magnitude of the rotation in Radians */
	void setRotation(const Vector3& axis, const Real& _angle)
	{
		Real d = axis.norm();
		ASSERT(d != Real(0.0));
		Real s = sinr(_angle * Real(0.5)) / d;
		setValue(axis.x() * s, axis.y() * s, axis.z() * s,
				 cosr(_angle * Real(0.5)));
	}
	/**@brief Set the quaternion using Euler angles
   * @param yaw Angle around Y
   * @param pitch Angle around X
   * @param roll Angle around Z */
	void setEuler(const Real& yaw, const Real& pitch, const Real& roll)
	{
		Real halfYaw = Real(yaw) * Real(0.5);
		Real halfPitch = Real(pitch) * Real(0.5);
		Real halfRoll = Real(roll) * Real(0.5);
		Real cosYaw = cosr(halfYaw);
		Real sinYaw = sinr(halfYaw);
		Real cosPitch = cosr(halfPitch);
		Real sinPitch = sinr(halfPitch);
		Real cosRoll = cosr(halfRoll);
		Real sinRoll = sinr(halfRoll);
		setValue(cosRoll * sinPitch * cosYaw + sinRoll * cosPitch * sinYaw,
				 cosRoll * cosPitch * sinYaw - sinRoll * sinPitch * cosYaw,
				 sinRoll * cosPitch * cosYaw - cosRoll * sinPitch * sinYaw,
				 cosRoll * cosPitch * cosYaw + sinRoll * sinPitch * sinYaw);
	}
	/**@brief Set the quaternion using euler angles 
   * @param yaw Angle around Z
   * @param pitch Angle around Y
   * @param roll Angle around X */
	void setEulerZYX(const Real& yawZ, const Real& pitchY, const Real& rollX)
	{
		Real halfYaw = Real(yawZ) * Real(0.5);
		Real halfPitch = Real(pitchY) * Real(0.5);
		Real halfRoll = Real(rollX) * Real(0.5);
		Real cosYaw = cosr(halfYaw);
		Real sinYaw = sinr(halfYaw);
		Real cosPitch = cosr(halfPitch);
		Real sinPitch = sinr(halfPitch);
		Real cosRoll = cosr(halfRoll);
		Real sinRoll = sinr(halfRoll);
		setValue(sinRoll * cosPitch * cosYaw - cosRoll * sinPitch * sinYaw,   //x
				 cosRoll * sinPitch * cosYaw + sinRoll * cosPitch * sinYaw,   //y
				 cosRoll * cosPitch * sinYaw - sinRoll * sinPitch * cosYaw,   //z
				 cosRoll * cosPitch * cosYaw + sinRoll * sinPitch * sinYaw);  //formerly yzx
	}

	/**@brief Get the euler angles from this quaternion
	   * @param yaw Angle around Z
	   * @param pitch Angle around Y
	   * @param roll Angle around X */
	void getEulerZYX(Real& yawZ, Real& pitchY, Real& rollX) const
	{
		Real squ;
		Real sqx;
		Real sqy;
		Real sqz;
		Real sarg;
		sqx = m_floats[0] * m_floats[0];
		sqy = m_floats[1] * m_floats[1];
		sqz = m_floats[2] * m_floats[2];
		squ = m_floats[3] * m_floats[3];
		sarg = Real(-2.) * (m_floats[0] * m_floats[2] - m_floats[3] * m_floats[1]);

		// If the pitch angle is PI/2 or -PI/2, we can only compute
		// the sum roll + yaw.  However, any combination that gives
		// the right sum will produce the correct orientation, so we
		// set rollX = 0 and compute yawZ.
		if (sarg <= -Real(0.99999))
		{
			pitchY = Real(-0.5) * REAL_PI;
			rollX = 0;
			yawZ = Real(2) * atan2r(m_floats[0], -m_floats[1]);
		}
		else if (sarg >= Real(0.99999))
		{
			pitchY = Real(0.5) * REAL_PI;
			rollX = 0;
			yawZ = Real(2) * atan2r(-m_floats[0], m_floats[1]);
		}
		else
		{
			pitchY = asinr(sarg);
			rollX = atan2r(2 * (m_floats[1] * m_floats[2] + m_floats[3] * m_floats[0]), squ - sqx - sqy + sqz);
			yawZ = atan2r(2 * (m_floats[0] * m_floats[1] + m_floats[3] * m_floats[2]), squ + sqx - sqy - sqz);
		}
	}

	/**@brief Add two quaternions
   * @param q The quaternion to add to this one */
	FORCE_INLINE Quaternion& operator+=(const Quaternion& q)
	{
		m_floats[0] += q.x();
		m_floats[1] += q.y();
		m_floats[2] += q.z();
		m_floats[3] += q.m_floats[3];
		return *this;
	}

	/**@brief Subtract out a quaternion
   * @param q The quaternion to subtract from this one */
	Quaternion& operator-=(const Quaternion& q)
	{
		m_floats[0] -= q.x();
		m_floats[1] -= q.y();
		m_floats[2] -= q.z();
		m_floats[3] -= q.m_floats[3];
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
			m_floats[3] * q.x() + m_floats[0] * q.m_floats[3] + m_floats[1] * q.z() - m_floats[2] * q.y(),
			m_floats[3] * q.y() + m_floats[1] * q.m_floats[3] + m_floats[2] * q.x() - m_floats[0] * q.z(),
			m_floats[3] * q.z() + m_floats[2] * q.m_floats[3] + m_floats[0] * q.y() - m_floats[1] * q.x(),
			m_floats[3] * q.m_floats[3] - m_floats[0] * q.x() - m_floats[1] * q.y() - m_floats[2] * q.z());
		return *this;
	}
	/**@brief Return the dot product between this quaternion and another
   * @param q The other quaternion */
	Real dot(const Quaternion& q) const
	{
		return m_floats[0] * q.x() +
			   m_floats[1] * q.y() +
			   m_floats[2] * q.z() +
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
		return Quaternion(x() * s, y() * s, z() * s, m_floats[3] * s);
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
	/**@brief Return the ***half*** angle between this quaternion and the other
   * @param q The other quaternion */
	Real angle(const Quaternion& q) const
	{
		Real s = sqrtr(squaredNorm() * q.squaredNorm());
		ASSERT(s != Real(0.0));
		return acosr(dot(q) / s);
	}

	/**@brief Return the angle between this quaternion and the other along the shortest path
	* @param q The other quaternion */
	Real angleShortestPath(const Quaternion& q) const
	{
		Real s = sqrtr(squaredNorm() * q.squaredNorm());
		ASSERT(s != Real(0.0));
		if (dot(q) < 0)  // Take care of long angle case see http://en.wikipedia.org/wiki/Slerp
			return acosr(dot(-q) / s) * Real(2.0);
		else
			return acosr(dot(q) / s) * Real(2.0);
	}

	/**@brief Return the angle [0, 2Pi] of rotation represented by this quaternion */
	Real getAngle() const
	{
		Real s = Real(2.) * acosr(m_floats[3]);
		return s;
	}

	/**@brief Return the angle [0, Pi] of rotation represented by this quaternion along the shortest path */
	Real getAngleShortestPath() const
	{
		Real s;
		if (m_floats[3] >= 0)
			s = Real(2.) * acosr(m_floats[3]);
		else
			s = Real(2.) * acosr(-m_floats[3]);
		return s;
	}

	/**@brief Return the axis of the rotation represented by this quaternion */
	Vector3 getAxis() const
	{
		Real s_squared = 1.f - m_floats[3] * m_floats[3];

		if (s_squared < Real(10.) * REAL_EPSILON)  //Check for divide by zero
			return Vector3(1.0, 0.0, 0.0);           // Arbitrary
		Real s = 1.f / sqrtr(s_squared);
		return Vector3(m_floats[0] * s, m_floats[1] * s, m_floats[2] * s);
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
		return Quaternion(q1.x() + q2.x(), q1.y() + q2.y(), q1.z() + q2.z(), q1.m_floats[3] + q2.m_floats[3]);
	}

	/**@brief Return the difference between this quaternion and the other 
   * @param q2 The other quaternion */
	FORCE_INLINE Quaternion
	operator-(const Quaternion& q2) const
	{
		const Quaternion& q1 = *this;
		return Quaternion(q1.x() - q2.x(), q1.y() - q2.y(), q1.z() - q2.z(), q1.m_floats[3] - q2.m_floats[3]);
	}

	/**@brief Return the negative of this quaternion 
   * This simply negates each element */
	FORCE_INLINE Quaternion operator-() const
	{
		const Quaternion& q2 = *this;
		return Quaternion(-q2.x(), -q2.y(), -q2.z(), -q2.m_floats[3]);
	}
	/**@todo document this and it's use */
	FORCE_INLINE Quaternion farthest(const Quaternion& qd) const
	{
		Quaternion diff, sum;
		diff = *this - qd;
		sum = *this + qd;
		if (diff.dot(diff) > sum.dot(sum))
			return qd;
		return (-qd);
	}

	/**@todo document this and it's use */
	FORCE_INLINE Quaternion nearest(const Quaternion& qd) const
	{
		Quaternion diff, sum;
		diff = *this - qd;
		sum = *this + qd;
		if (diff.dot(diff) < sum.dot(sum))
			return qd;
		return (-qd);
	}

	/**@brief Return the quaternion which is the result of Spherical Linear Interpolation between this and the other quaternion
   * @param q The other quaternion to interpolate with 
   * @param t The ratio between this and q to interpolate.  If t = 0 the result is this, if t=1 the result is q.
   * Slerp interpolates assuming constant velocity.  */
	Quaternion slerp(const Quaternion& q, const Real& t) const
	{
		const Real magnitude = sqrtr(squaredNorm() * q.squaredNorm());
		ASSERT(magnitude > Real(0));

		const Real product = dot(q) / magnitude;
		const Real absproduct = fabsr(product);

		if (absproduct < Real(1.0 - REAL_EPSILON))
		{
			// Take care of long angle case see http://en.wikipedia.org/wiki/Slerp
			const Real theta = acosr(absproduct);
			const Real d = sinr(theta);
			ASSERT(d > Real(0));

			const Real sign = (product < 0) ? Real(-1) : Real(1);
			const Real s0 = sinr((Real(1.0) - t) * theta) / d;
			const Real s1 = sinr(sign * t * theta) / d;

			return Quaternion(
				(m_floats[0] * s0 + q.x() * s1),
				(m_floats[1] * s0 + q.y() * s1),
				(m_floats[2] * s0 + q.z() * s1),
				(m_floats[3] * s0 + q.w() * s1));
		}
		else
		{
			return *this;
		}
	}

	static const Quaternion& Identity()
	{
		static const Quaternion identityQuat(Real(0.), Real(0.), Real(0.), Real(1.));
		return identityQuat;
	}
	
	friend std::ostream& operator<<(std::ostream&o,const Quaternion&q){
		o << "(" << q.x() <<", "<< q.y() <<", " << q.z()<<", " << q.w()<<")";
		return o;	
	}

	FORCE_INLINE const Real& getW() const { return m_floats[3]; }

	FORCE_INLINE void serialize(struct QuaternionData& dataOut) const;

	FORCE_INLINE void deSerialize(const struct QuaternionFloatData& dataIn);

	FORCE_INLINE void deSerialize(const struct QuaternionDoubleData& dataIn);

	FORCE_INLINE void serializeFloat(struct QuaternionFloatData& dataOut) const;

	FORCE_INLINE void deSerializeFloat(const struct QuaternionFloatData& dataIn);

	FORCE_INLINE void serializeDouble(struct QuaternionDoubleData& dataOut) const;

	FORCE_INLINE void deSerializeDouble(const struct QuaternionDoubleData& dataIn);
};

/**@brief Return the product of two quaternions */
FORCE_INLINE Quaternion
operator*(const Quaternion& q1, const Quaternion& q2)
{
	return Quaternion(
		q1.w() * q2.x() + q1.x() * q2.w() + q1.y() * q2.z() - q1.z() * q2.y(),
		q1.w() * q2.y() + q1.y() * q2.w() + q1.z() * q2.x() - q1.x() * q2.z(),
		q1.w() * q2.z() + q1.z() * q2.w() + q1.x() * q2.y() - q1.y() * q2.x(),
		q1.w() * q2.w() - q1.x() * q2.x() - q1.y() * q2.y() - q1.z() * q2.z());
}

FORCE_INLINE Quaternion
operator*(const Quaternion& q, const Vector3& w)
{

	return Quaternion(
		q.w() * w.x() + q.y() * w.z() - q.z() * w.y(),
		q.w() * w.y() + q.z() * w.x() - q.x() * w.z(),
		q.w() * w.z() + q.x() * w.y() - q.y() * w.x(),
		-q.x() * w.x() - q.y() * w.y() - q.z() * w.z());
}

FORCE_INLINE Quaternion
operator*(const Vector3& w, const Quaternion& q)
{
	return Quaternion(
		+w.x() * q.w() + w.y() * q.z() - w.z() * q.y(),
		+w.y() * q.w() + w.z() * q.x() - w.x() * q.z(),
		+w.z() * q.w() + w.x() * q.y() - w.y() * q.x(),
		-w.x() * q.x() - w.y() * q.y() - w.z() * q.z());
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

/**@brief Return the angle between two quaternions*/
FORCE_INLINE Real
btAngle(const Quaternion& q1, const Quaternion& q2)
{
	return q1.angle(q2);
}

/**@brief Return the inverse of a quaternion*/
FORCE_INLINE Quaternion
inverse(const Quaternion& q)
{
	return q.inverse();
}

/**@brief Return the result of spherical linear interpolation betwen two quaternions 
 * @param q1 The first quaternion
 * @param q2 The second quaternion 
 * @param t The ration between q1 and q2.  t = 0 return q1, t=1 returns q2 
 * Slerp assumes constant velocity between positions. */
FORCE_INLINE Quaternion
slerp(const Quaternion& q1, const Quaternion& q2, const Real& t)
{
	return q1.slerp(q2, t);
}

FORCE_INLINE Vector3
quatRotate(const Quaternion& rotation, const Vector3& v)
{
	Quaternion q = rotation * v;
	q *= rotation.inverse();
	return Vector3(q.getX(), q.getY(), q.getZ());
}


PHYS_NAMESPACE_END