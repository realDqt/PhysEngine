#pragma once
#include "common/general.h"
#include "common/array.h"

PHYS_NAMESPACE_BEGIN

class SphKernel{
public:
    SphKernel(Real _h){
        Init(_h);
    }

    // Constructor for SphKernel class
    // Initializes the class with the given smoothing length (_h)
    void Init(Real _h){
        h=_h;
        h6=pow(h, 6);
        h9=pow(h, 9);
        poly6_const=315.0f / (64 * REAL_PI * h9);

        // Precompute constant value for Spiky gradient
        spiky_grad_const=-45.0f / (REAL_PI * h6);

        // Precompute constant value for Cubic spline kernel (k)
        cspline_k_const = 32.0f / (REAL_PI * pow(h, 9));

        // Precompute constant value for Cubic spline kernel (a)
        cspline_a_const = -pow(h, 6) / 64.0f;

        // Precompute constant value for Cubic kernel
        cubic_const = 1 / (REAL_PI * pow(h, 3));
    }

    Real h; // Smoothing length
    Real h6; // h^6
    Real h9; // h^9
    Real poly6_const; // Constant value for Poly6 kernel
    Real spiky_grad_const; // Constant value for Spiky gradient
    Real cspline_k_const; // Constant value for Cubic spline kernel (k)
    Real cspline_a_const; // Constant value for Cubic spline kernel (a)
    Real cubic_const; // Constant value for Cubic kernel

    // Calculates the Poly6 kernel value for the given deltaPos
    inline PE_CUDA_FUNC Real Poly6(vec3r deltaPos)
    {
        Real checkVal = h * h - dot(deltaPos,deltaPos);
        // real constant = 315.0f / (64 * REAL_PI * h9);
        return checkVal > 0 ? poly6_const * pow(checkVal, 3) : 0;
    }

    // Calculates the Spiky gradient for the given deltaPos
    inline PE_CUDA_FUNC vec3r SpikyGrad(vec3r deltaPos)
    {
        Real r = length(deltaPos);
        Real checkVal = h - r;
        // real constant = -45.0f / (REAL_PI * h6);

    #ifdef PE_USE_CUDA 
        return checkVal > 0 ? spiky_grad_const * pow(checkVal, 2) * deltaPos / max(r, 1e-8f) : make_vec3r(0, 0, 0);
    #else
        return checkVal > 0 ? spiky_grad_const * pow(checkVal, 2) * deltaPos / std::max(r, 1e-8f) : Vector3r(0, 0, 0);
    #endif
    }

    // Calculates the Cubic spline kernel value for the given deltaPos
    inline PE_CUDA_FUNC Real Cspline(vec3r deltaPos)
    {
        Real r = length(deltaPos);
        // real coefK = 32.0f / (REAL_PI * pow(h, 9));
        // real coefA = -pow(h, 6) / 64.0f;
        if (r > h)
            return 0; // Return 0 if the distance is greater than the smoothing length
        else if (2 * r > h)
            return cspline_k_const * pow((h - r) * r, 3); // Calculate the Cubic spline kernel value (k) for 2r > h
        else
            return cspline_k_const * (2 * pow((h - r) * r, 3) + cspline_a_const); // Calculate the Cubic spline kernel value (k) for 2r <= h
    }

    // Calculates the Cubic kernel value for the given deltaPos
    inline PE_CUDA_FUNC Real Cubic(vec3r deltaPos)
    {
        Real r = length(deltaPos);
        Real q = r / h;
        if (q < 1) {
            return cubic_const * (1 - 1.5 * q * q * (1 - 0.5 * q)); // Calculate the Cubic kernel value for q < 1
        }
        else if (q < 2) {
            return 0.25 * cubic_const * pow(2 - q, 3); // Calculate the Cubic kernel value for 1 <= q < 2
        }
        else
            return 0; // Return 0 if q >= 2
    }
};

PHYS_NAMESPACE_END