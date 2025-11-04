#pragma once

#include "math/math.h"

PHYS_NAMESPACE_BEGIN


/**@brief The Transform class supports rigid transforms with only translation and rotation and no scaling/shear.
 *It can be used in combination with Vector3, Quaternion and Matrix3x3 linear algebra classes. */
ATTRIBUTE_ALIGNED16(class)
Transform
{
public:
	///Storage for the rotation
	Matrix3x3 m_rot;
	///Storage for the translation
	Vector3 m_pos;

public:
	/**@brief No initialization constructor */
	Transform():m_rot(Matrix3x3::Zero()), m_pos(Vector3::Zero()) {}
	/**@brief Constructor from Quaternion (optional Vector3 )
   * @param q Rotation from quaternion 
   * @param c Translation from Vector (default 0,0,0) */
	// explicit FORCE_INLINE Transform(const Quaternion& q,
	// 									   const Vector3& c = Vector3(Real(0), Real(0), Real(0)))
	// 	: m_rot(q),
	// 	  m_pos(c)
	// {
	// }

	/**@brief Constructor from Matrix3x3 (optional Vector3)
   * @param b Rotation from Matrix 
   * @param c Translation from Vector default (0,0,0)*/
	explicit FORCE_INLINE Transform(const Matrix3x3& b,
										   const Vector3& c = Vector3(Real(0), Real(0), Real(0)))
		: m_rot(b),
		  m_pos(c)
	{
	}
	/**@brief Copy constructor */
	FORCE_INLINE Transform(const Transform& other)
		: m_rot(other.m_rot),
		  m_pos(other.m_pos)
	{
	}
	/**@brief Assignment Operator */
	FORCE_INLINE Transform& operator=(const Transform& other)
	{
		m_rot = other.m_rot;
		m_pos = other.m_pos;
		return *this;
	}

	/**@brief Set the current transform as the value of the product of two transforms
   * @param t1 Transform 1
   * @param t2 Transform 2
   * This = Transform1 * Transform2 */
	FORCE_INLINE void mult(const Transform& t1, const Transform& t2)
	{
		m_rot = t1.m_rot * t2.m_rot;
		m_pos = t1(t2.m_pos);
	}

	/*		void multInverseLeft(const Transform& t1, const Transform& t2) {
			Vector3 v = t2.m_pos - t1.m_pos;
			m_rot = btMultTransposeLeft(t1.m_rot, t2.m_rot);
			m_pos = v * t1.m_rot;
		}
		*/

	/**@brief Return the transform of the vector */
	FORCE_INLINE Vector3 operator()(const Vector3& x) const
	{
		// return x.dot3(m_rot[0], m_rot[1], m_rot[2]) + m_pos;
        return m_rot * x + m_pos;
	}

	/**@brief Return the transform of the vector */
	FORCE_INLINE Vector3 operator*(const Vector3& x) const
	{
		return (*this)(x);
	}

	/**@brief Return the transform of the Quaternion */
	FORCE_INLINE Quaternion operator*(const Quaternion& q) const
	{
		return getRotation() * q;
	}

	/**@brief Return the basis matrix for the rotation */
	FORCE_INLINE Matrix3x3& getBasis() { return m_rot; }
	/**@brief Return the basis matrix for the rotation */
	FORCE_INLINE const Matrix3x3& getBasis() const { return m_rot; }

	/**@brief Return the origin vector translation */
	FORCE_INLINE Vector3& getOrigin() { return m_pos; }
	/**@brief Return the origin vector translation */
	FORCE_INLINE const Vector3& getOrigin() const { return m_pos; }

	/**@brief Return a quaternion representing the rotation */
	Quaternion getRotation() const
	{
        Quaternion q(m_rot);
		return q;
	}
	/**@brief Set the translational element
   * @param origin The vector to set the translation to */
	FORCE_INLINE void setOrigin(const Vector3& origin)
	{
		m_pos = origin;
	}

	FORCE_INLINE Vector3 invXform(const Vector3& inVec) const {
        Vector3 v = inVec - m_pos;
        return (m_rot.transpose() * v);
    }

	/**@brief Set the rotational element by Matrix3x3 */
	FORCE_INLINE void setBasis(const Matrix3x3& basis)
	{
		m_rot = basis;
	}

	/**@brief Set the rotational element by Quaternion */
	FORCE_INLINE void setRotation(const Quaternion& q)
	{
		// m_rot.setRotation(q);
        m_rot = q.toRotationMatrix();
	}

	/**@brief Set this transformation to the identity */
	void setIdentity()
	{
		m_rot.setIdentity();
		// m_pos.setValue(Real(0.0), Real(0.0), Real(0.0));
        m_pos = Vector3::Zero();
	}

	/**@brief Multiply this Transform by another(this = this * another) 
   * @param t The other transform */
	Transform& operator*=(const Transform& t)
	{
		m_pos += m_rot * t.m_pos;
		m_rot *= t.m_rot;
		return *this;
	}

	/**@brief Return an identity transform */
	static const Transform& Identity()
	{
		static const Transform identityTransform(Matrix3x3::Identity());
		return identityTransform;
	}

};




PHYS_NAMESPACE_END
