#pragma once

// #include <iostream>
#include "math/math.h"
#include "math/vector3.h"
// #include "math/quaternion.h"

PHYS_NAMESPACE_BEGIN

#ifdef USE_DOUBLE
#define Matrix3x3Data Matrix3x3DoubleData
#define Matrix3x3DataName "Matrix3x3DoubleData"
#else
#define Matrix3x3Data Matrix3x3FloatData
#define Matrix3x3DataName "Matrix3x3FloatData"
#endif

ATTRIBUTE_ALIGNED16(class)
Matrix3x3
{
public:
	///Data storage for the matrix, each vector is a row of the matrix
	Vector3 m_el[3];

public:
	/** @brief No initializaion constructor */
	Matrix3x3() {}

	//		explicit Matrix3x3(const Real *m) { setFromOpenGLSubMatrix(m); }

	/**@brief Constructor from Quaternion */
	// explicit Matrix3x3(const Quaternion& q) { setRotation(q); }
	/*
	template <typename Real>
	Matrix3x3(const Real& yaw, const Real& pitch, const Real& roll)
	{ 
	setEulerYPR(yaw, pitch, roll);
	}
	*/
	/** @brief Constructor with row major formatting */
	Matrix3x3(const Real& xx, const Real& xy, const Real& xz,
				const Real& yx, const Real& yy, const Real& yz,
				const Real& zx, const Real& zy, const Real& zz)
	{
		setValue(xx, xy, xz,
				 yx, yy, yz,
				 zx, zy, zz);
	}

	/** @brief Copy constructor */
	FORCE_INLINE Matrix3x3(const Matrix3x3& other)
	{
		m_el[0] = other.m_el[0];
		m_el[1] = other.m_el[1];
		m_el[2] = other.m_el[2];
	}

	/** @brief Assignment Operator */
	FORCE_INLINE Matrix3x3& operator=(const Matrix3x3& other)
	{
		m_el[0] = other.m_el[0];
		m_el[1] = other.m_el[1];
		m_el[2] = other.m_el[2];
		return *this;
	}
    
    FORCE_INLINE Matrix3x3(const Vector3& v0, const Vector3& v1, const Vector3& v2)
    {
        m_el[0] = v0;
        m_el[1] = v1;
        m_el[2] = v2;
    }

	/** @brief Get a column of the matrix as a vector 
	*  @param i Column number 0 indexed */
	FORCE_INLINE Vector3 getColumn(int i) const
	{
		return Vector3(m_el[0][i], m_el[1][i], m_el[2][i]);
	}

	/** @brief Get a row of the matrix as a vector 
	*  @param i Row number 0 indexed */
	FORCE_INLINE const Vector3& getRow(int i) const
	{
		DEBUG_ASSERT(0 <= i && i < 3);
		return m_el[i];
	}

	/** @brief Get a mutable reference to a row of the matrix as a vector 
	*  @param i Row number 0 indexed */
	FORCE_INLINE Vector3& operator[](int i)
	{
		DEBUG_ASSERT(0 <= i && i < 3);
		return m_el[i];
	}

	/** @brief Get a const reference to a row of the matrix as a vector 
	*  @param i Row number 0 indexed */
	FORCE_INLINE const Vector3& operator[](int i) const
	{
		DEBUG_ASSERT(0 <= i && i < 3);
		return m_el[i];
	}

	FORCE_INLINE Real& operator()(int r, int c)
	{
		return m_el[r][c];
	}

	FORCE_INLINE const Real& operator()(int r, int c) const
	{
		return m_el[r][c];
	}

	/** @brief Multiply by the target matrix on the right
	*  @param m Rotation matrix to be applied 
	* Equivilant to this = this * m */
	Matrix3x3& operator*=(const Matrix3x3& m);

	/** @brief Adds by the target matrix on the right
	*  @param m matrix to be applied 
	* Equivilant to this = this + m */
	Matrix3x3& operator+=(const Matrix3x3& m);

	/** @brief Substractss by the target matrix on the right
	*  @param m matrix to be applied 
	* Equivilant to this = this - m */
	Matrix3x3& operator-=(const Matrix3x3& m);

	/** @brief Set from the rotational part of a 4x4 OpenGL matrix
	*  @param m A pointer to the beginning of the array of scalars*/
	void setFromOpenGLSubMatrix(const Real* m)
	{
		m_el[0].setValue(m[0], m[4], m[8]);
		m_el[1].setValue(m[1], m[5], m[9]);
		m_el[2].setValue(m[2], m[6], m[10]);
	}
	/** @brief Set the values of the matrix explicitly (row major)
	*  @param xx Top left
	*  @param xy Top Middle
	*  @param xz Top Right
	*  @param yx Middle Left
	*  @param yy Middle Middle
	*  @param yz Middle Right
	*  @param zx Bottom Left
	*  @param zy Bottom Middle
	*  @param zz Bottom Right*/
	void setValue(const Real& xx, const Real& xy, const Real& xz,
				  const Real& yx, const Real& yy, const Real& yz,
				  const Real& zx, const Real& zy, const Real& zz)
	{
		m_el[0].setValue(xx, xy, xz);
		m_el[1].setValue(yx, yy, yz);
		m_el[2].setValue(zx, zy, zz);
	}

	// /** @brief Set the matrix from a quaternion
	// *  @param q The Quaternion to match */
	// void setRotation(const Quaternion& q)
	// {
	// 	Real d = q.squaredNorm();
	// 	DEBUG_ASSERT(d != Real(0.0));
	// 	Real s = Real(2.0) / d;

	// 	Real xs = q.x() * s, ys = q.y() * s, zs = q.z() * s;
	// 	Real wx = q.w() * xs, wy = q.w() * ys, wz = q.w() * zs;
	// 	Real xx = q.x() * xs, xy = q.x() * ys, xz = q.x() * zs;
	// 	Real yy = q.y() * ys, yz = q.y() * zs, zz = q.z() * zs;
	// 	setValue(
	// 		Real(1.0) - (yy + zz), xy - wz, xz + wy,
	// 		xy + wz, Real(1.0) - (xx + zz), yz - wx,
	// 		xz - wy, yz + wx, Real(1.0) - (xx + yy));
	// }

	/** @brief Set the matrix from euler angles using YPR around YXZ respectively
	*  @param yaw Yaw about Y axis
	*  @param pitch Pitch about X axis
	*  @param roll Roll about Z axis 
	*/
	void setEulerYPR(const Real& yaw, const Real& pitch, const Real& roll)
	{
		setEulerZYX(roll, pitch, yaw);
	}

	/** @brief Set the matrix from euler angles YPR around ZYX axes
	* @param eulerX Roll about X axis
	* @param eulerY Pitch around Y axis
	* @param eulerZ Yaw about Z axis
	* 
	* These angles are used to produce a rotation matrix. The euler
	* angles are applied in ZYX order. I.e a vector is first rotated 
	* about X then Y and then Z
	**/
	void setEulerZYX(Real eulerX, Real eulerY, Real eulerZ)
	{
		///@todo proposed to reverse this since it's labeled zyx but takes arguments xyz and it will match all other parts of the code
		Real ci(cosf(eulerX));
		Real cj(cosf(eulerY));
		Real ch(cosf(eulerZ));
		Real si(sinf(eulerX));
		Real sj(sinf(eulerY));
		Real sh(sinf(eulerZ));
		Real cc = ci * ch;
		Real cs = ci * sh;
		Real sc = si * ch;
		Real ss = si * sh;

		setValue(cj * ch, sj * sc - cs, sj * cc + ss,
				 cj * sh, sj * ss + cc, sj * cs - sc,
				 -sj, cj * si, cj * ci);
	}

	/**@brief Set the matrix to the identity */
	void setIdentity()
	{
		setValue(Real(1.0), Real(0.0), Real(0.0),
				 Real(0.0), Real(1.0), Real(0.0),
				 Real(0.0), Real(0.0), Real(1.0));
	}
    
    /**@brief Set the matrix to the identity */
    void setZero()
    {
        setValue(Real(0.0), Real(0.0), Real(0.0),
                 Real(0.0), Real(0.0), Real(0.0),
                 Real(0.0), Real(0.0), Real(0.0));
    }

	static const Matrix3x3& Identity()
	{
		static const Matrix3x3
			identityMatrix(
				Real(1.0), Real(0.0), Real(0.0),
				Real(0.0), Real(1.0), Real(0.0),
				Real(0.0), Real(0.0), Real(1.0));
		return identityMatrix;
	}
	
	static const Matrix3x3& Zero()
	{
		static const Matrix3x3
			zeroMatrix(
				Real(0.0), Real(0.0), Real(0.0),
				Real(0.0), Real(0.0), Real(0.0),
				Real(0.0), Real(0.0), Real(0.0));
		return zeroMatrix;
	}

	/**@brief Fill the rotational part of an OpenGL matrix and clear the shear/perspective
	* @param m The array to be filled */
	void getOpenGLSubMatrix(Real * m) const
	{
		m[0] = Real(m_el[0].x());
		m[1] = Real(m_el[1].x());
		m[2] = Real(m_el[2].x());
		m[3] = Real(0.0);
		m[4] = Real(m_el[0].y());
		m[5] = Real(m_el[1].y());
		m[6] = Real(m_el[2].y());
		m[7] = Real(0.0);
		m[8] = Real(m_el[0].z());
		m[9] = Real(m_el[1].z());
		m[10] = Real(m_el[2].z());
		m[11] = Real(0.0);
	}

	// /**@brief Get the matrix represented as a quaternion 
	// * @param q The quaternion which will be set */
	// void getRotation(Quaternion & q) const
	// {
	// 	Real trace = m_el[0].x() + m_el[1].y() + m_el[2].z();

	// 	Real temp[4];

	// 	if (trace > Real(0.0))
	// 	{
	// 		Real s = sqrtr(trace + Real(1.0));
	// 		temp[3] = (s * Real(0.5));
	// 		s = Real(0.5) / s;

	// 		temp[0] = ((m_el[2].y() - m_el[1].z()) * s);
	// 		temp[1] = ((m_el[0].z() - m_el[2].x()) * s);
	// 		temp[2] = ((m_el[1].x() - m_el[0].y()) * s);
	// 	}
	// 	else
	// 	{
	// 		int i = m_el[0].x() < m_el[1].y() ? (m_el[1].y() < m_el[2].z() ? 2 : 1) : (m_el[0].x() < m_el[2].z() ? 2 : 0);
	// 		int j = (i + 1) % 3;
	// 		int k = (i + 2) % 3;

	// 		Real s = sqrtr(m_el[i][i] - m_el[j][j] - m_el[k][k] + Real(1.0));
	// 		temp[i] = s * Real(0.5);
	// 		s = Real(0.5) / s;

	// 		temp[3] = (m_el[k][j] - m_el[j][k]) * s;
	// 		temp[j] = (m_el[j][i] + m_el[i][j]) * s;
	// 		temp[k] = (m_el[k][i] + m_el[i][k]) * s;
	// 	}
	// 	q.setValue(temp[0], temp[1], temp[2], temp[3]);
	// }

	/**@brief Get the matrix represented as euler angles around YXZ, roundtrip with setEulerYPR
	* @param yaw Yaw around Y axis
	* @param pitch Pitch around X axis
	* @param roll around Z axis */
	void getEulerYPR(Real & yaw, Real & pitch, Real & roll) const
	{
		// first use the normal calculus
		yaw = Real(atan2r(m_el[1].x(), m_el[0].x()));
		pitch = Real(asinr(-m_el[2].x()));
		roll = Real(atan2r(m_el[2].y(), m_el[2].z()));

		// on pitch = +/-HalfPI
		if (fabsr(pitch) == REAL_HALF_PI)
		{
			if (yaw > 0)
				yaw -= REAL_PI;
			else
				yaw += REAL_PI;

			if (roll > 0)
				roll -= REAL_PI;
			else
				roll += REAL_PI;
		}
	};

	/**@brief Get the matrix represented as euler angles around ZYX
	* @param yaw Yaw around Z axis
	* @param pitch Pitch around Y axis
	* @param roll around X axis 
	* @param solution_number Which solution of two possible solutions ( 1 or 2) are possible values*/
	void getEulerZYX(Real & yaw, Real & pitch, Real & roll, unsigned int solution_number = 1) const
	{
		struct Euler
		{
			Real yaw;
			Real pitch;
			Real roll;
		};

		Euler euler_out;
		Euler euler_out2;  //second solution
		//get the pointer to the raw data

		// Check that pitch is not at a singularity
		if (fabsr(m_el[2].x()) >= 1)
		{
			euler_out.yaw = 0;
			euler_out2.yaw = 0;

			// From difference of angles formula
			Real delta = atan2r(m_el[0].x(), m_el[0].z());
			if (m_el[2].x() > 0)  //gimbal locked up
			{
				euler_out.pitch = REAL_PI / Real(2.0);
				euler_out2.pitch = REAL_PI / Real(2.0);
				euler_out.roll = euler_out.pitch + delta;
				euler_out2.roll = euler_out.pitch + delta;
			}
			else  // gimbal locked down
			{
				euler_out.pitch = -REAL_PI / Real(2.0);
				euler_out2.pitch = -REAL_PI / Real(2.0);
				euler_out.roll = -euler_out.pitch + delta;
				euler_out2.roll = -euler_out.pitch + delta;
			}
		}
		else
		{
			euler_out.pitch = -asinr(m_el[2].x());
			euler_out2.pitch = REAL_PI - euler_out.pitch;

			euler_out.roll = atan2r(m_el[2].y() / cosf(euler_out.pitch),
									 m_el[2].z() / cosf(euler_out.pitch));
			euler_out2.roll = atan2r(m_el[2].y() / cosf(euler_out2.pitch),
									  m_el[2].z() / cosf(euler_out2.pitch));

			euler_out.yaw = atan2r(m_el[1].x() / cosf(euler_out.pitch),
									m_el[0].x() / cosf(euler_out.pitch));
			euler_out2.yaw = atan2r(m_el[1].x() / cosf(euler_out2.pitch),
									 m_el[0].x() / cosf(euler_out2.pitch));
		}

		if (solution_number == 1)
		{
			yaw = euler_out.yaw;
			pitch = euler_out.pitch;
			roll = euler_out.roll;
		}
		else
		{
			yaw = euler_out2.yaw;
			pitch = euler_out2.pitch;
			roll = euler_out2.roll;
		}
	}

	/**@brief Create a scaled copy of the matrix 
	* @param s Scaling vector The elements of the vector will scale each column */

	Matrix3x3 scaled(const Vector3& s) const
	{
		return Matrix3x3(
			m_el[0].x() * s.x(), m_el[0].y() * s.y(), m_el[0].z() * s.z(),
			m_el[1].x() * s.x(), m_el[1].y() * s.y(), m_el[1].z() * s.z(),
			m_el[2].x() * s.x(), m_el[2].y() * s.y(), m_el[2].z() * s.z());
	}

	/**@brief Return the determinant of the matrix */
	Real determinant() const;
	/**@brief Return the adjoint of the matrix */
	Matrix3x3 adjoint() const;
	/**@brief Return the matrix with all values non negative */
	Matrix3x3 absolute() const;
	/**@brief Return the transpose of the matrix */
	Matrix3x3 transpose() const;
	/**@brief Return the inverse of the matrix */
	Matrix3x3 inverse() const;

	/// Solve A * x = b, where b is a column vector. This is more efficient
	/// than computing the inverse in one-shot cases.
	///Solve33 is from Box2d, thanks to Erin Catto,
	Vector3 solve33(const Vector3& b) const
	{
		Vector3 col1 = getColumn(0);
		Vector3 col2 = getColumn(1);
		Vector3 col3 = getColumn(2);

		Real det = dot(col1, cross(col2, col3));
		if (fabsr(det) > REAL_EPSILON)
		{
			det = 1.0f / det;
		}
		Vector3 x;
		x[0] = det * dot(b, cross(col2, col3));
		x[1] = det * dot(col1, cross(b, col3));
		x[2] = det * dot(col1, cross(col2, b));
		return x;
	}

	Matrix3x3 transposeTimes(const Matrix3x3& m) const;
	Matrix3x3 timesTranspose(const Matrix3x3& m) const;

	FORCE_INLINE Real tdotx(const Vector3& v) const
	{
		return m_el[0].x() * v.x() + m_el[1].x() * v.y() + m_el[2].x() * v.z();
	}
	FORCE_INLINE Real tdoty(const Vector3& v) const
	{
		return m_el[0].y() * v.x() + m_el[1].y() * v.y() + m_el[2].y() * v.z();
	}
	FORCE_INLINE Real tdotz(const Vector3& v) const
	{
		return m_el[0].z() * v.x() + m_el[1].z() * v.y() + m_el[2].z() * v.z();
	}

	// ///extractRotation is from "A robust method to extract the rotational part of deformations"
	// ///See http://dl.acm.org/citation.cfm?doid=2994258.2994269
	// ///decomposes a matrix A in a orthogonal matrix R and a
	// ///symmetric matrix S:
	// ///A = R*S.
	// ///note that R can include both rotation and scaling.
	// FORCE_INLINE void extractRotation(Quaternion & q, Real tolerance = 1.0e-9, int maxIter = 100)
	// {
	// 	int iter = 0;
	// 	Real w;
	// 	const Matrix3x3& A = *this;
	// 	for (iter = 0; iter < maxIter; iter++)
	// 	{
	// 		Matrix3x3 R(q);
	// 		Vector3 omega = (R.getColumn(0).cross(A.getColumn(0)) + R.getColumn(1).cross(A.getColumn(1)) + R.getColumn(2).cross(A.getColumn(2))) * (Real(1.0) / fabsr(R.getColumn(0).dot(A.getColumn(0)) + R.getColumn(1).dot(A.getColumn(1)) + R.getColumn(2).dot(A.getColumn(2))) +
	// 																																				  tolerance);
	// 		w = omega.norm();
	// 		if (w < tolerance)
	// 			break;
	// 		q = Quaternion(Vector3((Real(1.0) / w) * omega), w) *
	// 			q;
	// 		q.normalize();
	// 	}
	// }

	/**@brief diagonalizes this matrix by the Jacobi method.
	* @param rot stores the rotation from the coordinate system in which the matrix is diagonal to the original
	* coordinate system, i.e., old_this = rot * new_this * rot^T.
	* @param threshold See iteration
	* @param iteration The iteration stops when all off-diagonal elements are less than the threshold multiplied
	* by the sum of the absolute values of the diagonal, or when maxSteps have been executed.
	*
	* Note that this matrix is assumed to be symmetric.
	*/
	void diagonalize(Matrix3x3 & rot, Real threshold, int maxSteps)
	{
		rot.setIdentity();
		for (int step = maxSteps; step > 0; step--)
		{
			// find off-diagonal element [p][q] with largest magnitude
			int p = 0;
			int q = 1;
			int r = 2;
			Real max = fabsr(m_el[0][1]);
			Real v = fabsr(m_el[0][2]);
			if (v > max)
			{
				q = 2;
				r = 1;
				max = v;
			}
			v = fabsr(m_el[1][2]);
			if (v > max)
			{
				p = 1;
				q = 2;
				r = 0;
				max = v;
			}

			Real t = threshold * (fabsr(m_el[0][0]) + fabsr(m_el[1][1]) + fabsr(m_el[2][2]));
			if (max <= t)
			{
				if (max <= REAL_EPSILON * t)
				{
					return;
				}
				step = 1;
			}

			// compute Jacobi rotation J which leads to a zero for element [p][q]
			Real mpq = m_el[p][q];
			Real theta = (m_el[q][q] - m_el[p][p]) / (2 * mpq);
			Real theta2 = theta * theta;
			Real cos;
			Real sin;
			if (theta2 * theta2 < Real(10 / REAL_EPSILON))
			{
				t = (theta >= 0) ? 1 / (theta + sqrtr(1 + theta2))
								 : 1 / (theta - sqrtr(1 + theta2));
				cos = 1 / sqrtr(1 + t * t);
				sin = cos * t;
			}
			else
			{
				// approximation for large theta-value, i.e., a nearly diagonal matrix
				t = 1 / (theta * (2 + Real(0.5) / theta2));
				cos = 1 - Real(0.5) * t * t;
				sin = cos * t;
			}

			// apply rotation to matrix (this = J^T * this * J)
			m_el[p][q] = m_el[q][p] = 0;
			m_el[p][p] -= t * mpq;
			m_el[q][q] += t * mpq;
			Real mrp = m_el[r][p];
			Real mrq = m_el[r][q];
			m_el[r][p] = m_el[p][r] = cos * mrp - sin * mrq;
			m_el[r][q] = m_el[q][r] = cos * mrq + sin * mrp;

			// apply rotation to rot (rot = rot * J)
			for (int i = 0; i < 3; i++)
			{
				Vector3& row = rot[i];
				mrp = row[p];
				mrq = row[q];
				row[p] = cos * mrp - sin * mrq;
				row[q] = cos * mrq + sin * mrp;
			}
		}
	}

	/**@brief Calculate the matrix cofactor 
	* @param r1 The first row to use for calculating the cofactor
	* @param c1 The first column to use for calculating the cofactor
	* @param r1 The second row to use for calculating the cofactor
	* @param c1 The second column to use for calculating the cofactor
	* See http://en.wikipedia.org/wiki/Cofactor_(linear_algebra) for more details
	*/
	Real cofac(int r1, int c1, int r2, int c2) const
	{
		return m_el[r1][c1] * m_el[r2][c2] - m_el[r1][c2] * m_el[r2][c1];
	}

	FORCE_INLINE void serialize(struct Matrix3x3Data & dataOut) const;

	FORCE_INLINE void serializeFloat(struct Matrix3x3FloatData & dataOut) const;

	FORCE_INLINE void deSerialize(const struct Matrix3x3Data& dataIn);

	FORCE_INLINE void deSerializeFloat(const struct Matrix3x3FloatData& dataIn);

	FORCE_INLINE void deSerializeDouble(const struct Matrix3x3DoubleData& dataIn);


	friend std::ostream& operator<<(std::ostream&o,const Matrix3x3&m){
		o << "[" << m[0][0] <<", "<< m[0][1] <<", " << m[0][2]<<"\n"
			<< " " << m[1][0] <<", "<< m[1][1] <<", " << m[1][2]<<"\n"
			" " << m[2][0] <<", "<< m[2][1] <<", " << m[2][2]<< "]";
		return o;	
	}
};

FORCE_INLINE Matrix3x3&
Matrix3x3::operator*=(const Matrix3x3& m)
{	setValue(
		m.tdotx(m_el[0]), m.tdoty(m_el[0]), m.tdotz(m_el[0]),
		m.tdotx(m_el[1]), m.tdoty(m_el[1]), m.tdotz(m_el[1]),
		m.tdotx(m_el[2]), m.tdoty(m_el[2]), m.tdotz(m_el[2]));
	return *this;
}

FORCE_INLINE Matrix3x3&
Matrix3x3::operator+=(const Matrix3x3& m)
{
	setValue(
		m_el[0][0] + m.m_el[0][0],
		m_el[0][1] + m.m_el[0][1],
		m_el[0][2] + m.m_el[0][2],
		m_el[1][0] + m.m_el[1][0],
		m_el[1][1] + m.m_el[1][1],
		m_el[1][2] + m.m_el[1][2],
		m_el[2][0] + m.m_el[2][0],
		m_el[2][1] + m.m_el[2][1],
		m_el[2][2] + m.m_el[2][2]);
	return *this;
}

FORCE_INLINE Matrix3x3
operator*(const Matrix3x3& m, const Real& k)
{
	return Matrix3x3(
		m[0].x() * k, m[0].y() * k, m[0].z() * k,
		m[1].x() * k, m[1].y() * k, m[1].z() * k,
		m[2].x() * k, m[2].y() * k, m[2].z() * k);
}

FORCE_INLINE Matrix3x3
operator+(const Matrix3x3& m1, const Matrix3x3& m2)
{
	return Matrix3x3(
		m1[0][0] + m2[0][0],
		m1[0][1] + m2[0][1],
		m1[0][2] + m2[0][2],

		m1[1][0] + m2[1][0],
		m1[1][1] + m2[1][1],
		m1[1][2] + m2[1][2],

		m1[2][0] + m2[2][0],
		m1[2][1] + m2[2][1],
		m1[2][2] + m2[2][2]);
}

FORCE_INLINE Matrix3x3
operator-(const Matrix3x3& m1, const Matrix3x3& m2)
{
	return Matrix3x3(
		m1[0][0] - m2[0][0],
		m1[0][1] - m2[0][1],
		m1[0][2] - m2[0][2],

		m1[1][0] - m2[1][0],
		m1[1][1] - m2[1][1],
		m1[1][2] - m2[1][2],

		m1[2][0] - m2[2][0],
		m1[2][1] - m2[2][1],
		m1[2][2] - m2[2][2]);
}

FORCE_INLINE Matrix3x3&
Matrix3x3::operator-=(const Matrix3x3& m)
{
#if (defined(BT_USE_SSE_IN_API) && defined(BT_USE_SSE)) || defined(BT_USE_NEON)
	m_el[0].mVec128 = m_el[0].mVec128 - m.m_el[0].mVec128;
	m_el[1].mVec128 = m_el[1].mVec128 - m.m_el[1].mVec128;
	m_el[2].mVec128 = m_el[2].mVec128 - m.m_el[2].mVec128;
#else
	setValue(
		m_el[0][0] - m.m_el[0][0],
		m_el[0][1] - m.m_el[0][1],
		m_el[0][2] - m.m_el[0][2],
		m_el[1][0] - m.m_el[1][0],
		m_el[1][1] - m.m_el[1][1],
		m_el[1][2] - m.m_el[1][2],
		m_el[2][0] - m.m_el[2][0],
		m_el[2][1] - m.m_el[2][1],
		m_el[2][2] - m.m_el[2][2]);
#endif
	return *this;
}

FORCE_INLINE Real
Matrix3x3::determinant() const
{
	return triple((*this)[0], (*this)[1], (*this)[2]);
}

FORCE_INLINE Matrix3x3
Matrix3x3::absolute() const
{
	return Matrix3x3(
		fabsr(m_el[0].x()), fabsr(m_el[0].y()), fabsr(m_el[0].z()),
		fabsr(m_el[1].x()), fabsr(m_el[1].y()), fabsr(m_el[1].z()),
		fabsr(m_el[2].x()), fabsr(m_el[2].y()), fabsr(m_el[2].z()));
}

FORCE_INLINE Matrix3x3
Matrix3x3::transpose() const
{
	return Matrix3x3(m_el[0].x(), m_el[1].x(), m_el[2].x(),
					   m_el[0].y(), m_el[1].y(), m_el[2].y(),
					   m_el[0].z(), m_el[1].z(), m_el[2].z());
}

FORCE_INLINE Matrix3x3
Matrix3x3::adjoint() const
{
	return Matrix3x3(cofac(1, 1, 2, 2), cofac(0, 2, 2, 1), cofac(0, 1, 1, 2),
					   cofac(1, 2, 2, 0), cofac(0, 0, 2, 2), cofac(0, 2, 1, 0),
					   cofac(1, 0, 2, 1), cofac(0, 1, 2, 0), cofac(0, 0, 1, 1));
}

FORCE_INLINE Matrix3x3
Matrix3x3::inverse() const
{
	Vector3 co(cofac(1, 1, 2, 2), cofac(1, 2, 2, 0), cofac(1, 0, 2, 1));
	Real det = (*this)[0].dot(co);
	//DEBUG_ASSERT(det != Real(0.0));
	ASSERT(det != Real(0.0));
	Real s = Real(1.0) / det;
	return Matrix3x3(co.x() * s, cofac(0, 2, 2, 1) * s, cofac(0, 1, 1, 2) * s,
					   co.y() * s, cofac(0, 0, 2, 2) * s, cofac(0, 2, 1, 0) * s,
					   co.z() * s, cofac(0, 1, 2, 0) * s, cofac(0, 0, 1, 1) * s);
}

FORCE_INLINE Matrix3x3
Matrix3x3::transposeTimes(const Matrix3x3& m) const
{
	return Matrix3x3(
		m_el[0].x() * m[0].x() + m_el[1].x() * m[1].x() + m_el[2].x() * m[2].x(),
		m_el[0].x() * m[0].y() + m_el[1].x() * m[1].y() + m_el[2].x() * m[2].y(),
		m_el[0].x() * m[0].z() + m_el[1].x() * m[1].z() + m_el[2].x() * m[2].z(),
		m_el[0].y() * m[0].x() + m_el[1].y() * m[1].x() + m_el[2].y() * m[2].x(),
		m_el[0].y() * m[0].y() + m_el[1].y() * m[1].y() + m_el[2].y() * m[2].y(),
		m_el[0].y() * m[0].z() + m_el[1].y() * m[1].z() + m_el[2].y() * m[2].z(),
		m_el[0].z() * m[0].x() + m_el[1].z() * m[1].x() + m_el[2].z() * m[2].x(),
		m_el[0].z() * m[0].y() + m_el[1].z() * m[1].y() + m_el[2].z() * m[2].y(),
		m_el[0].z() * m[0].z() + m_el[1].z() * m[1].z() + m_el[2].z() * m[2].z());
}

FORCE_INLINE Matrix3x3
Matrix3x3::timesTranspose(const Matrix3x3& m) const
{
	return Matrix3x3(
		m_el[0].dot(m[0]), m_el[0].dot(m[1]), m_el[0].dot(m[2]),
		m_el[1].dot(m[0]), m_el[1].dot(m[1]), m_el[1].dot(m[2]),
		m_el[2].dot(m[0]), m_el[2].dot(m[1]), m_el[2].dot(m[2]));
}

FORCE_INLINE Vector3
operator*(const Matrix3x3& m, const Vector3& v)
{
	return Vector3(m[0].dot(v), m[1].dot(v), m[2].dot(v));
}

FORCE_INLINE Vector3
operator*(const Vector3& v, const Matrix3x3& m)
{
	return Vector3(m.tdotx(v), m.tdoty(v), m.tdotz(v));
}

FORCE_INLINE Matrix3x3
operator*(const Matrix3x3& m1, const Matrix3x3& m2)
{
	return Matrix3x3(
		m2.tdotx(m1[0]), m2.tdoty(m1[0]), m2.tdotz(m1[0]),
		m2.tdotx(m1[1]), m2.tdoty(m1[1]), m2.tdotz(m1[1]),
		m2.tdotx(m1[2]), m2.tdoty(m1[2]), m2.tdotz(m1[2]));
}

/*
FORCE_INLINE Matrix3x3 btMultTransposeLeft(const Matrix3x3& m1, const Matrix3x3& m2) {
return Matrix3x3(
m1[0][0] * m2[0][0] + m1[1][0] * m2[1][0] + m1[2][0] * m2[2][0],
m1[0][0] * m2[0][1] + m1[1][0] * m2[1][1] + m1[2][0] * m2[2][1],
m1[0][0] * m2[0][2] + m1[1][0] * m2[1][2] + m1[2][0] * m2[2][2],
m1[0][1] * m2[0][0] + m1[1][1] * m2[1][0] + m1[2][1] * m2[2][0],
m1[0][1] * m2[0][1] + m1[1][1] * m2[1][1] + m1[2][1] * m2[2][1],
m1[0][1] * m2[0][2] + m1[1][1] * m2[1][2] + m1[2][1] * m2[2][2],
m1[0][2] * m2[0][0] + m1[1][2] * m2[1][0] + m1[2][2] * m2[2][0],
m1[0][2] * m2[0][1] + m1[1][2] * m2[1][1] + m1[2][2] * m2[2][1],
m1[0][2] * m2[0][2] + m1[1][2] * m2[1][2] + m1[2][2] * m2[2][2]);
}
*/

/**@brief Equality operator between two matrices
* It will test all elements are equal.  */
FORCE_INLINE bool operator==(const Matrix3x3& m1, const Matrix3x3& m2)
{
	return (m1[0][0] == m2[0][0] && m1[1][0] == m2[1][0] && m1[2][0] == m2[2][0] &&
			m1[0][1] == m2[0][1] && m1[1][1] == m2[1][1] && m1[2][1] == m2[2][1] &&
			m1[0][2] == m2[0][2] && m1[1][2] == m2[1][2] && m1[2][2] == m2[2][2]);
}


///for serialization
struct Matrix3x3FloatData
{
	Vector3FloatData m_el[3];
};

///for serialization
struct Matrix3x3DoubleData
{
	Vector3DoubleData m_el[3];
};


FORCE_INLINE void Matrix3x3::serialize(struct Matrix3x3Data& dataOut) const
{
	for (int i = 0; i < 3; i++)
		m_el[i].serialize(dataOut.m_el[i]);
}

FORCE_INLINE void Matrix3x3::serializeFloat(struct Matrix3x3FloatData& dataOut) const
{
	for (int i = 0; i < 3; i++)
		m_el[i].serializeFloat(dataOut.m_el[i]);
}

FORCE_INLINE void Matrix3x3::deSerialize(const struct Matrix3x3Data& dataIn)
{
	for (int i = 0; i < 3; i++)
		m_el[i].deSerialize(dataIn.m_el[i]);
}

FORCE_INLINE void Matrix3x3::deSerializeFloat(const struct Matrix3x3FloatData& dataIn)
{
	for (int i = 0; i < 3; i++)
		m_el[i].deSerializeFloat(dataIn.m_el[i]);
}

FORCE_INLINE void Matrix3x3::deSerializeDouble(const struct Matrix3x3DoubleData& dataIn)
{
	for (int i = 0; i < 3; i++)
		m_el[i].deSerializeDouble(dataIn.m_el[i]);
}

PHYS_NAMESPACE_END
