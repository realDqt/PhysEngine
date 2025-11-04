#pragma once

#include "math/math.h"

PHYS_NAMESPACE_BEGIN
/** 
 * @brief Compute the tangent space, given a normal vector.
 * The implementation is mainly based on bullet3 now.
 * (bullet3/src/Bullet3Common/b3Vector.h/b3PlaneSpace1())
 */

FORCE_INLINE void getOrthoUnits(const Vector3 v, Vector3 &p, Vector3 &q){
    const Vector3 n = v.normalized();
    if (fabs(n[2]) > 0.707)
    {
        // choose p in y-z plane
        Real a = n[1] * n[1] + n[2] * n[2];
        Real k = (Real)1.0/sqrtr(a);
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
        Real k = (Real)1.0/sqrtr(a);
        p[0] = -n[1] * k;
        p[1] = n[0] * k;
        p[2] = 0;
        // set q = n x p
        q[0] = -n[2] * p[1];
        q[1] = n[2] * p[0];
        q[2] = a * k;
    }
}
PHYS_NAMESPACE_END