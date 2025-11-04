#pragma once
#include "object/particle_fluid.h"

PHYS_NAMESPACE_BEGIN



inline __host__ __device__ bool _d_isLiquid(uint ph){
    return ph==(uint)PhaseType::Liquid || ph==(uint)PhaseType::Oil;
}

// inline __host__ __device__ vec3r _d_enforceBoundaryLocal(vec3r pos){
//     vec3r npos=pos;
//     npos.x=(max(min(npos.x,params.worldMax.x), params.worldMin.x)+pos.x)*0.5f;
//     npos.y=(max(min(npos.y,params.worldMax.y), params.worldMin.y)+pos.y)*0.5f;
//     npos.z=(max(min(npos.z,params.worldMax.z), params.worldMin.z)+pos.z)*0.5f;
//     return npos;
// }

inline __host__ __device__ Real _d_poly6(Real d){
    Real tmp = params.h2 - d * d;
    return params.poly6Coef * tmp*tmp*tmp;
}

inline __host__ __device__ vec3r _d_poly6Grad(vec3r r, Real d){
    Real tmp = params.h2 - d*d;
    return -6 * params.poly6Coef * tmp*tmp*r;
}

inline __host__ __device__ Real _d_poly6(const Real d, const Real h2, const Real poly6Coef){
    Real tmp = params.h2 - d*d;
    return params.poly6Coef * tmp*tmp*tmp;
}

inline __host__ __device__ vec3r _d_spikyGrad(vec3r r){
    Real d = length(r);
    Real tmp = params.h - d;
    return (params.spikyGradCoef * tmp * tmp / max(d, 1e-8f)) * r;
}

inline __host__ __device__ vec3r _d_spikyGrad(vec3r r, Real d){
    Real tmp = params.h - d;
    return (params.spikyGradCoef / max(d, 1e-8f) * tmp * tmp) * r;
}

inline __host__ __device__ vec3r _d_spikyGrad(const vec3r& r, const Real& d, const Real& h, const Real& spikyGradCoef){
    Real tmp = params.h - d;
    // return (spikyGradCoef * tmp * tmp / max(d, 1e-8f)) * r;
    return (params.spikyGradCoef / (d+1e-8f) * tmp * tmp) * r;
}

//// for cohesion
//// ref to "2013-Versatile surface tension and adhesion for sph fluids"
inline __host__ __device__ Real _d_cSpline(Real d){
    Real hr3r3=pow((params.h-d)*d,3.0f);
    if(d<params.halfh) return (2*hr3r3-params.cohesionConstCoef)*params.cohesionCoef;
    else return hr3r3*params.cohesionCoef;
}

inline __host__ __device__ Real _d_rsWeight(Real d){
    if(d<params.h)
        return 1-d/params.h;
    else
        return 0;
}

inline __host__ __device__ Real _d_clampAndNormalize(Real x, Real Tmin, Real Tmax){
    return (min(x, Tmax) - min(x, Tmin))/(Tmax - Tmin);  
}

inline __host__ __device__ void _d_getOrthogonalVectors(vec3r vec, vec3r& e1, vec3r& e2) {
    vec3r v = make_vec3r(1, 0, 0);
    if (fabs(dot(v, vec)) > 0.999)
        v = make_vec3r(0, 1, 0);

    e1 = cross(vec, v);
    e2 = cross(vec, e1);
    e1 = normalize(e1);
    e2 = normalize(e2);
}

PHYS_NAMESPACE_END