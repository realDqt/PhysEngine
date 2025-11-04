#include "object/particle_system_util.h"
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
PHYS_NAMESPACE_BEGIN
// constant parameters on GPU
// defined in particle_system_util.h
__constant__ SimParams params;

void setParameters(SimParams *hostParams)
{
    // copy parameters to constant memory
    //// const hparams on CPU
    hparams = *hostParams;
    //// const params on GPU
    checkCudaErrors(cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)));
}

//// integrate device code
template <MemType MT>
__host__ __device__ void _k_integrate(int i, Real dt, vec3r *tmpPos, vec3r *pos, vec4r *vel)
{
    //// v'=v+g*dt
    vec3r v = make_vec3r(vel[i]);
    v += params.gravity * dt;
    v *= params.globalDamping;
    vel[i] = make_vec4r(v);
    //// x'=x+v*dt
    tmpPos[i] = pos[i] + v * dt;
}
DECLARE_KERNEL_TEMP(integrate, Real dt, vec3r *tmpPos, vec3r *pos, vec4r *vel);
// integrate kernel 
__global__ void _g_integrate(int size, Real dt, vec3r *tmpPos, vec3r *pos, vec4r *vel)
{
    IF_IDX_VALID(size)
    _k_integrate<MemType::GPU>(i, dt, tmpPos, pos, vel);
}
// integrate calling method 
template <MemType MT>
void callIntegrate(int size, Real dt, VecArray<vec3r, MT> &tmpPos, VecArray<vec3r, MT> &pos, VecArray<vec4r, MT> &vel)
{
    FILL_CALL_DEVICE_CODE(integrate, dt, tmpPos.m_data, pos.m_data, vel.m_data);
}
//implementation of the template
template void callIntegrate<MemType::CPU>(int, Real, VecArray<vec3r, MemType::CPU> &, VecArray<vec3r, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &);
template void callIntegrate<MemType::GPU>(int, Real, VecArray<vec3r, MemType::GPU> &, VecArray<vec3r, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &);

//// integrate for vec4r
template <MemType MT>
__host__ __device__ void _k_integrate_4(int i, Real dt, vec4r *vtxph, vec4r *vxph, vec4r *vv)
{
    vec4r xph = vxph[i];
    //// v'=v+g*dt
    vec3r v = make_vec3r(vv[i]);
    //update the velocity based on the gravity and damping
    v += params.gravity * dt;
    v *= params.globalDamping;
    if (uint(xph.w) == PhaseType::Fixed)
    {
        v = make_vec3r(0.0f);
    }
    vv[i] = make_vec4r(v);
    //// x'=x+v*dt
    vtxph[i] = make_vec4r(make_vec3r(xph) + v * dt, xph.w);
    // printf("integrate %d vec:%f %f %f txph:%f %f %f\n",v.x,v.y,v.z,vtxph[i].x,vtxph[i].y,vtxph[i].z);
}
DECLARE_KERNEL_TEMP(integrate_4, Real dt, vec4r *vtxph, vec4r *vxph, vec4r *vv);
// integrate for vec4r kernel
__global__ void _g_integrate_4(int size, Real dt, vec4r *vtxph, vec4r *vxph, vec4r *vv)
{
    IF_IDX_VALID(size)
    _k_integrate_4<MemType::GPU>(i, dt, vtxph, vxph, vv);
}
// integrate for vec4r calling method
template <MemType MT>
void callIntegrate(int size, Real dt, VecArray<vec4r, MT> &vtxph, VecArray<vec4r, MT> &vxph, VecArray<vec4r, MT> &vv)
{
    FILL_CALL_DEVICE_CODE(integrate_4, dt, vtxph.m_data, vxph.m_data, vv.m_data);
}
//implementation of the template
template void callIntegrate<MemType::CPU>(int, Real, VecArray<vec4r, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &);
template void callIntegrate<MemType::GPU>(int, Real, VecArray<vec4r, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &);

//// integrate with force
template <MemType MT>
__host__ __device__ void _k_integrate_4(int i, Real dt, vec4r *vtxph, vec4r *vxph, vec4r *vv, vec3r *vf)
{
    vec4r xph = vxph[i];
    //// v'=v+g*dt
    vec3r v = make_vec3r(vv[i]);
    v += vf[i] * dt;
    v *= params.globalDamping;
    if (uint(xph.w) == PhaseType::Fixed)
    {
        v = make_vec3r(0.0f);
    }
    vv[i] = make_vec4r(v);
    //// x'=x+v*dt
    vtxph[i] = make_vec4r(make_vec3r(xph) + v * dt, xph.w);
    vf[i] = make_vec3r(0.0f, 0.0f, 0.0f);
    // printf("integrate %d vec:%f %f %f txph:%f %f %f\n",v.x,v.y,v.z,vtxph[i].x,vtxph[i].y,vtxph[i].z);
}
DECLARE_KERNEL_TEMP(integrate_4, Real dt, vec4r *vtxph, vec4r *vxph, vec4r *vv, vec3r *vf);
// integrate with force kernel
__global__ void _g_integrate_4(int size, Real dt, vec4r *vtxph, vec4r *vxph, vec4r *vv, vec3r *vf)
{
    IF_IDX_VALID(size)
    _k_integrate_4<MemType::GPU>(i, dt, vtxph, vxph, vv, vf);
}
// integrate with force calling method
template <MemType MT>
void callIntegrate(int size, Real dt, VecArray<vec4r, MT> &vtxph, VecArray<vec4r, MT> &vxph, VecArray<vec4r, MT> &vv, VecArray<vec3r, MT> &vf)
{
    FILL_CALL_DEVICE_CODE(integrate_4, dt, vtxph.m_data, vxph.m_data, vv.m_data, vf.m_data);
}
//implementation of the template
template void callIntegrate<MemType::CPU>(int, Real, VecArray<vec4r, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &, VecArray<vec3r, MemType::CPU> &);
template void callIntegrate<MemType::GPU>(int, Real, VecArray<vec4r, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<vec3r, MemType::GPU> &);

//// integrate quaternion
template <MemType MT>
__host__ __device__ void _k_integrateQuaternion(int i, Real dt, quaternionf *vtq, quaternionf *vq, vec3r *vw, vec3r *vtorque, mat3r *vim)
{
    // printf("\nintegrate quaternion\n");
    quaternionf q = vq[i];
    //// v'=v+g*dt
    vec3r w = vw[i];
    mat3r im = vim[i];
    vec3r t = vtorque[i];
    w += im.inverse() * (t - cross(w, (im * w))) * dt;
    w *= params.globalDamping;
    // printf("integrate angular velocity %d: %f, %f, %f\n", i, w.x, w.y, w.z);

    quaternionf u = q + 0.5f * dt * quaternionf(w) * q;
    vtq[i] = normalize(u);
    vec3r axis;
    Real radians;
    vtq[i].get_value(axis, radians);
    // printf("integrate quaternion %d: axis: %f, %f, %f, radians: %f\n", i, axis.x, axis.y, axis.z, radians);

    vtorque[i] = make_vec3r(0.0, 0.0, 0.0);
}
DECLARE_KERNEL_TEMP(integrateQuaternion, Real dt, quaternionf *vtq, quaternionf *vq, vec3r *vw, vec3r *vtorque, mat3r *vim);
// integrate quaternion kernel
__global__ void _g_integrateQuaternion(int size, Real dt, quaternionf *vtq, quaternionf *vq, vec3r *vw, vec3r *vtorque, mat3r *vim)
{
    IF_IDX_VALID(size)
    _k_integrateQuaternion<MemType::GPU>(i, dt, vtq, vq, vw, vtorque, vim);
}
// integrate quaternion calling method
template <MemType MT>
void callIntegrateQuaternion(int size, Real dt, VecArray<quaternionf, MT> &tq, VecArray<quaternionf, MT> &q, VecArray<vec3r, MT> &w, VecArray<vec3r, MT> &torque, VecArray<mat3r, MT> &im)
{
    FILL_CALL_DEVICE_CODE(integrateQuaternion, dt, tq.m_data, q.m_data, w.m_data, torque.m_data, im.m_data);
}
//implementation of the template
template void callIntegrateQuaternion<MemType::CPU>(int, Real, VecArray<quaternionf, MemType::CPU> &, VecArray<quaternionf, MemType::CPU> &, VecArray<vec3r, MemType::CPU> &, VecArray<vec3r, MemType::CPU> &, VecArray<mat3r, MemType::CPU> &);
template void callIntegrateQuaternion<MemType::GPU>(int, Real, VecArray<quaternionf, MemType::GPU> &, VecArray<quaternionf, MemType::GPU> &, VecArray<vec3r, MemType::GPU> &, VecArray<vec3r, MemType::GPU> &, VecArray<mat3r, MemType::GPU> &);

//// merge position and phase into one vec4r array
template <MemType MT>
__host__ __device__ void _k_mergePositionPhase(int i, int start, vec4r *vxph, vec3r *vx, uint *vph)
{
    //// xph=(x, ph)
    if (i >= start)
        vxph[i] = make_vec4r(vx[i], (Real)vph[i]);
}
DECLARE_KERNEL_TEMP(mergePositionPhase, int start, vec4r *vxph, vec3r *vx, uint *vph);
// merge position and phase kernel
__global__ void _g_mergePositionPhase(int size, int start, vec4r *vxph, vec3r *vx, uint *vph)
{
    IF_IDX_VALID(size)
    _k_mergePositionPhase<MemType::GPU>(i, start, vxph, vx, vph);
}
// merge position and phase calling method
template <MemType MT>
void callMergePositionPhase(int size, int start, VecArray<vec4r, MT> &vxph, VecArray<vec3r, MT> &vx, VecArray<uint, MT> &vph)
{
    FILL_CALL_DEVICE_CODE(mergePositionPhase, start, vxph.m_data, vx.m_data, vph.m_data);
}
//implementation of the template
template void callMergePositionPhase<MemType::CPU>(int, int, VecArray<vec4r, MemType::CPU> &, VecArray<vec3r, MemType::CPU> &, VecArray<uint, MemType::CPU> &);
template void callMergePositionPhase<MemType::GPU>(int, int, VecArray<vec4r, MemType::GPU> &, VecArray<vec3r, MemType::GPU> &, VecArray<uint, MemType::GPU> &);

//// calculate hash on spatial grid
template <MemType MT>
__host__ __device__ void _k_updateHash(int i, uint *gridParticleHash, uint *S2OParticleIndex, vec3r *pos)
{
    //// store grid hash and particle index
    gridParticleHash[i] = _d_calcGridHash(_d_calcGridPos(pos[i]));
    S2OParticleIndex[i] = i;
}
DECLARE_KERNEL_TEMP(updateHash, uint *gridParticleHash, uint *S2OParticleIndex, vec3r *pos);
// calculate hash on spatial grid kernel
__global__ void _g_updateHash(int size, uint *gridParticleHash, uint *S2OParticleIndex, vec3r *pos)
{
    IF_IDX_VALID(size)
    _k_updateHash<MemType::GPU>(i, gridParticleHash, S2OParticleIndex, pos);
}
// calculate hash on spatial grid calling method
template <MemType MT>
void callUpdateHash(int size, VecArray<uint, MT> &gridParticleHash, VecArray<uint, MT> &S2OParticleIndex, VecArray<vec3r, MT> &pos)
{
    FILL_CALL_DEVICE_CODE(updateHash, gridParticleHash.m_data, S2OParticleIndex.m_data, pos.m_data);
}
//implementation of the template
template void callUpdateHash<MemType::CPU>(int, VecArray<uint, MemType::CPU> &, VecArray<uint, MemType::CPU> &, VecArray<vec3r, MemType::CPU> &);
template void callUpdateHash<MemType::GPU>(int, VecArray<uint, MemType::GPU> &, VecArray<uint, MemType::GPU> &, VecArray<vec3r, MemType::GPU> &);

//// update hash for vec4r
template <MemType MT>
__host__ __device__ void _k_updateHash_4(int i, uint *gridParticleHash, uint *S2OParticleIndex, vec4r *pos)
{
    //// store grid hash and particle index
    gridParticleHash[i] = _d_calcGridHash(_d_calcGridPos(make_vec3r(pos[i])));
    S2OParticleIndex[i] = i;
}
DECLARE_KERNEL_TEMP(updateHash_4, uint *gridParticleHash, uint *S2OParticleIndex, vec4r *pos);
// update hash for vec4r kernel
__global__ void _g_updateHash_4(int size, uint *gridParticleHash, uint *S2OParticleIndex, vec4r *pos)
{
    IF_IDX_VALID(size)
    _k_updateHash_4<MemType::GPU>(i, gridParticleHash, S2OParticleIndex, pos);
}
// update hash for vec4r calling method
template <MemType MT>
void callUpdateHash(int size, VecArray<uint, MT> &gridParticleHash, VecArray<uint, MT> &S2OParticleIndex, VecArray<vec4r, MT> &pos)
{
    FILL_CALL_DEVICE_CODE(updateHash_4, gridParticleHash.m_data, S2OParticleIndex.m_data, pos.m_data);
}
//implementation of the template
template void callUpdateHash<MemType::CPU>(int, VecArray<uint, MemType::CPU> &, VecArray<uint, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &);
template void callUpdateHash<MemType::GPU>(int, VecArray<uint, MemType::GPU> &, VecArray<uint, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &);

//// sort based on hash
template <MemType MT>
void sortParticles(int size, VecArray<uint, MT> &phash, VecArray<uint, MT> &pidx)
{
    // use GPU to sort
    thrust::sort_by_key(thrust::device,thrust::device_ptr<uint>(phash.m_data),thrust::device_ptr<uint>(phash.m_data + size),thrust::device_ptr<uint>(pidx.m_data));
}
template void sortParticles<MemType::GPU>(int size, VecArray<uint, MemType::GPU> &phash, VecArray<uint, MemType::GPU> &pidx);

//// collide
__host__ __device__ vec3r _d_collideShape(){return make_vec3r(0.0f);}
// collide with sphere
__host__ __device__ vec3r _d_collideSphere(vec3r c, Real r, vec3r x)
{
    //// sphere with center c and radius r
    vec3r nx = x;
    vec3r rel = x - c;
    Real l = length(rel);
    if (l < r)
        nx = c + (r / l) * rel;
    return nx;
}
// collide with column
__host__ __device__ vec3r _d_collideColumn(vec3r c, Real r, vec3r x)
{
    //// vertical column with center c and radius r
    vec3r nx = x;
    vec3r rel = x - c;
    rel.y = 0;
    Real l = length(rel);
    c.y = x.y;
    if (l < r)
        nx = c + (r / l) * rel;
    return nx;
}
// collide with obstacle
template <MemType MT>
__host__ __device__ void _k_collideStaticObstacle(int i, vec4r *sortedPositionPhase)
{
    vec4r xph = sortedPositionPhase[i];
    vec3r x = make_vec3r(xph);
    Real phr = xph.w;
    // choose column or sphere as obstacle
    x = _d_collideColumn(params.ocol_x[0], params.ocol_r[0], x);
    x = _d_collideColumn(params.ocol_x[1], params.ocol_r[1], x);
    x = _d_collideColumn(params.ocol_x[2], params.ocol_r[2], x);
    sortedPositionPhase[i] = make_vec4r(x, phr);
    // // nsortedPositionPhase=_d_collideSphere(params.osph_x[0],params.osph_r[0],nsortedPositionPhase);
    // return nsortedPositionPhase;
}
DECLARE_KERNEL_TEMP(collideStaticObstacle, vec4r *sortedPositionPhase);
// collide with obstacle kernel
__global__ void _g_collideStaticObstacle(int size, vec4r *sortedPositionPhase)
{
    IF_IDX_VALID(size) _k_collideStaticObstacle<MemType::GPU>(i, sortedPositionPhase);
}
// collide with obstacle calling method
template <MemType MT>
void callCollideStaticObstacle(int size, VecArray<vec4r, MT> &vx)
{
    FILL_CALL_DEVICE_CODE(collideStaticObstacle, vx.m_data);
}
template void callCollideStaticObstacle<MemType::CPU>(int, VecArray<vec4r, MemType::CPU> &);
template void callCollideStaticObstacle<MemType::GPU>(int, VecArray<vec4r, MemType::GPU> &);
// collide with sphere
template <MemType MT>
__host__ __device__ void _k_collideStaticSphere(int i, vec4r *sortedPositionPhase)
{
    vec4r xph = sortedPositionPhase[i];
    vec3r x = make_vec3r(xph);
    Real phr = xph.w;
    x = _d_collideSphere(params.osph_x[0], params.osph_r[0], x);
    // x=_d_collideSphere(params.osph_x[1],params.osph_r[1],x);
    // x=_d_collideSphere(params.osph_x[2],params.osph_r[2],x);
    sortedPositionPhase[i] = make_vec4r(x, phr);
    // // nsortedPositionPhase=_d_collideSphere(params.osph_x[0],params.osph_r[0],nsortedPositionPhase);
    // return nsortedPositionPhase;
}
DECLARE_KERNEL_TEMP(collideStaticSphere, vec4r *sortedPositionPhase);
// collide with sphere kernel
__global__ void _g_collideStaticSphere(int size, vec4r *sortedPositionPhase)
{
    IF_IDX_VALID(size) _k_collideStaticSphere<MemType::GPU>(i, sortedPositionPhase);
}
// collide with sphere calling method
template <MemType MT>
void callCollideStaticSphere(int size, VecArray<vec4r, MT> &vx)
{
    FILL_CALL_DEVICE_CODE(collideStaticSphere, vx.m_data);
}
//implementation of template
template void callCollideStaticSphere<MemType::CPU>(int, VecArray<vec4r, MemType::CPU> &);
template void callCollideStaticSphere<MemType::GPU>(int, VecArray<vec4r, MemType::GPU> &);

//// update position and velocity
template <MemType MT>
__device__ void _k_updatePositionVelocity(int idx, Real dt, vec4r *sortedPositionPhase, vec3r *oldPos, vec4r *oldVel)
{
    const vec4r xph = sortedPositionPhase[idx];
    vec3r x = make_vec3r(xph);
    const uint ph = (uint)xph.w;

    //// v'=x'-x
    const vec3r ox = oldPos[idx];
    vec3r v = (x - ox) / dt;
    const Real lv2 = dot(v, v);
    // if lv2 is too small, set v to 0
    if (lv2 < params.sleepVelocity2)
    {
        v = make_vec3r(0.f);
        x = ox;
    }
    // if lv2 is too large, set v to maxVelocity
    else if (lv2 > params.maxVelocity2)
    {
        v *= params.maxVelocity * __frsqrt_rn(lv2);
        x = ox + v * dt;
    }
    if (ph == PhaseType::Fixed)
        return;
    oldPos[idx] = x;
    oldVel[idx] = make_vec4r(v);
    // printf("%d tempPH:%f %f %f, oldPH:%f %f %f, vec:%f %f %f\n",idx,
    //x.x,x.y,x.z,
    //ox.x,ox.y,ox.z,
    //v.x,v.y,v.z);
}
DECLARE_GPU_KERNEL_TEMP(updatePositionVelocity, Real dt, vec4r *sortedPositionPhase, vec3r *pos, vec4r *vel);
// update position and velocity kernel
__global__ void _g_updatePositionVelocity(int size, Real dt, vec4r *sortedPositionPhase, vec3r *pos, vec4r *vel)
{
    IF_IDX_VALID(size) _k_updatePositionVelocity<MemType::GPU>(i, dt, sortedPositionPhase, pos, vel);
}
// update position and velocity calling method
template <MemType MT>
void callUpdatePositionVelocity(int size, Real dt, VecArray<vec4r, MT> &sortedPositionPhase, VecArray<vec3r, MT> &pos, VecArray<vec4r, MT> &vel)
{
    FILL_CALL_GPU_DEVICE_CODE(updatePositionVelocity, dt, sortedPositionPhase.m_data, pos.m_data, vel.m_data);
}
//implementation of template
template void callUpdatePositionVelocity<MemType::CPU>(int, Real, VecArray<vec4r, MemType::CPU> &, VecArray<vec3r, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &);
template void callUpdatePositionVelocity<MemType::GPU>(int, Real, VecArray<vec4r, MemType::GPU> &, VecArray<vec3r, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &);

//// update quaternion and angular velocity
// This function updates the quaternion and angular velocity for a given index.
// It takes the index, time step (dt), arrays of quaternion values (vu and vq), and an array of 
template <MemType MT>
__device__ void _k_updateQuaternionAndAngularVelocity(int idx, Real dt, quaternionf *vu, quaternionf *vq, vec3r *vw)
{
    // printf("update quaternion and angular velocity\n");
    quaternionf u = vu[idx];
    quaternionf q = vq[idx];
    vw[idx] = make_vec3r((2 * u * conjugate(q) / dt)._array);
    vq[idx] = u;
    // printf("update angular velocity %d: %f, %f, %f\n", idx, vw[idx].x, vw[idx].y, vw[idx].z);
    vec3r axis;
    Real radians;
    u.get_value(axis, radians);
    // printf("update quaternion %d: axis: %f, %f, %f, radians: %f\n", idx, axis.x, axis.y, axis.z, radians);
}
DECLARE_GPU_KERNEL_TEMP(updateQuaternionAndAngularVelocity, Real dt, quaternionf *vu, quaternionf *vq, vec3r *vw);

// Wrapper kernel function to update quaternion and angular velocity on GPU
__global__ void _g_updateQuaternionAndAngularVelocity(int size, Real dt, quaternionf *vu, quaternionf *vq, vec3r *vw)
{
    IF_IDX_VALID(size)
    _k_updateQuaternionAndAngularVelocity<MemType::GPU>(i, dt, vu, vq, vw);
}
// Function to call the kernel function for updating quaternion and angular velocity
template <MemType MT>
void callUpdateQuaternionAndAngularVelocity(int size, Real dt, VecArray<quaternionf, MT> &vu, VecArray<quaternionf, MT> &vq, VecArray<vec3r, MT> &vw)
{
    FILL_CALL_GPU_DEVICE_CODE(updateQuaternionAndAngularVelocity, dt, vu.m_data, vq.m_data, vw.m_data);
}
template void callUpdateQuaternionAndAngularVelocity<MemType::CPU>(int, Real, VecArray<quaternionf, MemType::CPU> &, VecArray<quaternionf, MemType::CPU> &, VecArray<vec3r, MemType::CPU> &);
template void callUpdateQuaternionAndAngularVelocity<MemType::GPU>(int, Real, VecArray<quaternionf, MemType::GPU> &, VecArray<quaternionf, MemType::GPU> &, VecArray<vec3r, MemType::GPU> &);
// Explanation:
// The function _k_updateQuaternionAndAngularVelocity updates the quaternion and angular velocity for a given index. It first retrieves the quaternion value u from the array vu and the quaternion value q from the array vq. Then, it updates the angular velocity at the given index idx using these quaternion values and the time step dt. Next, it updates the quaternion value at the given index with the quaternion value u. Finally, it extracts the axis and radians values from u and stores them in the variables axis and radians, respectively.

// The kernel function _g_updateQuaternionAndAngularVelocity is a wrapper function that calls _k_updateQuaternionAndAngularVelocity for each valid index in parallel on the GPU.

// The function callUpdateQuaternionAndAngularVelocity is a templated function that handles the calling of the kernel function based on the memory type MT. It takes the size of the arrays, the time step dt, and references to the arrays vu, vq, and vw. It fills the GPU device code with the appropriate arguments and calls the kernel function _g_updateQuaternionAndAngularVelocity.
// Templated instantiations for both CPU and GPU memory types are provided at the end of the code.

//// update position and velocity
template <MemType MT>
__device__ void _k_updateSortedPositionVelocity(int idx, Real dt, vec4r *sortedPositionPhase, vec3r *oldPos, vec4r *oldVel, uint *S2OParticleIndex)
{
    const vec4r xph = sortedPositionPhase[idx];
    const uint oidx = S2OParticleIndex[idx]; // for ps
    vec3r x = make_vec3r(xph);
    const uint ph = (uint)xph.w;

    //// v'=x'-x
    const vec3r ox = oldPos[oidx];
    vec3r v = (x - ox) / dt;
    const Real lv2 = dot(v, v);

    if (lv2 < params.sleepVelocity2)
    {
        v = make_vec3r(0.f);
        x = ox;
    }
    else if (lv2 > params.maxVelocity2)
    {
        v *= params.maxVelocity * __frsqrt_rn(lv2);
        x = ox + v * dt;
    }

    if (ph == PhaseType::Fixed)
        return;
    oldPos[oidx] = x;
    oldVel[oidx] = make_vec4r(v);
}
DECLARE_GPU_KERNEL_TEMP(updateSortedPositionVelocity, Real dt, vec4r *sortedPositionPhase, vec3r *pos, vec4r *vel, uint *S2OParticleIndex);

__global__ void _g_updateSortedPositionVelocity(int size, Real dt, vec4r *sortedPositionPhase, vec3r *pos, vec4r *vel, uint *S2OParticleIndex)
{
    IF_IDX_VALID(size)
    _k_updateSortedPositionVelocity<MemType::GPU>(i, dt, sortedPositionPhase, pos, vel, S2OParticleIndex);
}
// Function to call the kernel function for updating sorted position and velocity
template <MemType MT>
void callUpdateSortedPositionVelocity(int size, Real dt, VecArray<vec4r, MT> &sortedPositionPhase, VecArray<vec3r, MT> &pos, VecArray<vec4r, MT> &vel, VecArray<uint, MT> &S2OParticleIndex)
{
    FILL_CALL_GPU_DEVICE_CODE(updateSortedPositionVelocity, dt, sortedPositionPhase.m_data, pos.m_data, vel.m_data, S2OParticleIndex.m_data);
}
template void callUpdateSortedPositionVelocity<MemType::CPU>(int, Real, VecArray<vec4r, MemType::CPU> &, VecArray<vec3r, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &, VecArray<uint, MemType::CPU> &);
template void callUpdateSortedPositionVelocity<MemType::GPU>(int, Real, VecArray<vec4r, MemType::GPU> &, VecArray<vec3r, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<uint, MemType::GPU> &);
// This function updates the sorted position and velocity arrays for a given index.
// It takes the index, time step (dt), arrays of sorted position and phase values (sortedPositionPhase),
// position values (pos), velocity values (vel), and particle index values (S2OParticleIndex).
template <MemType MT>
__device__ void _k_correctPositionAverage(int idx, vec4r *deltaPosition, vec4r *positionPhase)
{
    const vec4r xph = positionPhase[idx];
    const vec4r dxn = deltaPosition[idx];

    vec3r x = make_vec3r(xph);
    const uint ph = (uint)xph.w;

    vec3r dx = make_vec3r(dxn);
    Real cnt = dxn.w;

    // if(ph!=PhaseType::Fixed && cnt>0){
    //     positionPhase[idx]=make_vec4r(x+dx*(params.collisionStiffness/cnt), xph.w);
    // }
    cnt = max(cnt, 1.0);
    // positionPhase[idx]=make_vec4r(x+dx*(params.collisionStiffness/cnt), xph.w);
    positionPhase[idx] = make_vec4r(x + dx, xph.w);

    deltaPosition[idx] = make_vec4r(0.0f, 0.0f, 0.0f, 0.0f);
}
DECLARE_GPU_KERNEL_TEMP(correctPositionAverage, vec4r *dx, vec4r *xph);
// Kernel function to correct the position average on GPU
__global__ void _g_correctPositionAverage(int size, vec4r *dx, vec4r *xph)
{
    IF_IDX_VALID(size)
    _k_correctPositionAverage<MemType::GPU>(i, dx, xph);
}
// Function to call the kernel function for correcting the position average
template <MemType MT>
void callCorrectPositionAverage(int size, VecArray<vec4r, MT> &dx, VecArray<vec4r, MT> &xph)
{
    FILL_CALL_GPU_DEVICE_CODE(correctPositionAverage, dx.m_data, xph.m_data);
}

template void callCorrectPositionAverage<MemType::CPU>(int size, VecArray<vec4r, MemType::CPU> &dx, VecArray<vec4r, MemType::CPU> &xph);
template void callCorrectPositionAverage<MemType::GPU>(int size, VecArray<vec4r, MemType::GPU> &dx, VecArray<vec4r, MemType::GPU> &xph);
// Explanation:
// The function _g_updateSortedPositionVelocity is a wrapper function that calls the kernel function _k_updateSortedPositionVelocity for each valid index in parallel on the GPU. It takes the size, time step (dt), arrays of sorted position and phase values (sortedPositionPhase), position values (pos), velocity values (vel), and particle index values (S2OParticleIndex).

// The function callUpdateSortedPositionVelocity is a templated function that handles the calling of the kernel function based on the memory type MT. It takes the size, time step (dt), and references to the arrays sortedPositionPhase, pos, vel, and S2OParticleIndex. It fills the GPU device code with the appropriate arguments and calls the kernel function _g_updateSortedPositionVelocity.

// The function _k_updateSortedPositionVelocity updates the sorted position and velocity arrays for a given index. It retrieves the sorted position and phase value xph and the velocity value xn at the given index. It updates the position value pos at the given index by adding the product of the velocity and time step. Finally, it updates the sorted position and phase value sortedPositionPhase at the given index with the updated position value.

// The function _k_correctPositionAverage corrects the position average for a given index. It retrieves the position and phase value xph and the deltaposition value dxn at the given index. It extracts the position vector x from xph and the phase value ph from xph. It also extracts the delta position vector dx from dxn and the count value cnt from dxn. The count value is then clamped to a minimum of 1 to avoid division by zero. Finally, it updates the position and phase value positionPhase at the given index by adding the corrected delta position, which is the sum of the position vector x and the delta position vector dx. The delta position value is reset to zero.

//// update triangle info
// This function updates the triangle information for a given index.
// It takes the index, arrays of triangle information (triangleInfo), triangle centers (triangleCenter),
// triangle normals (triangleNormal), particle position and phase values (particlePositionPhase),
// and triangle vertex indices (triangleVertex).
template <MemType MT>
__device__ void _k_updateTriangleInfo(int i, TriangleInfo *__restrict__ triangleInfo, vec4r *__restrict__ triangleCenter, vec4r *__restrict__ triangleNormal, const vec4r *__restrict__ particlePositionPhase, const int4 *__restrict__ triangleVertex)
{

    const int4 vertices = triangleVertex[i];
    const vec3r x0 = make_vec3r(particlePositionPhase[vertices.x]);
    const vec3r x1 = make_vec3r(particlePositionPhase[vertices.y]);
    const vec3r x2 = make_vec3r(particlePositionPhase[vertices.z]);
    //// normal
    const vec3r n = normalize(cross(x1 - x0, x2 - x0));
    //// center
    const vec3r c = 0.3333333333333333f * (x0 + x1 + x2); //// TODO:use circumcenter
    // Store the triangle center, normal, and information at the given index
    triangleCenter[i] = make_vec4r(c);
    triangleNormal[i] = make_vec4r(n);
    triangleInfo[i] = TriangleInfo(x0, x1, x2, c, n);
}
DECLARE_GPU_KERNEL_TEMP(updateTriangleInfo, TriangleInfo *__restrict__ triangleInfo, vec4r *__restrict__ triangleCenter, vec4r *__restrict__ triangleNormal, const vec4r *__restrict__ particlePositionPhase, const int4 *__restrict__ triangleVertex);
// Kernel function to update triangle info on GPU
__global__ void _g_updateTriangleInfo(int size, TriangleInfo *__restrict__ triangleInfo, vec4r *__restrict__ triangleCenter, vec4r *__restrict__ triangleNormal, const vec4r *__restrict__ particlePositionPhase, const int4 *__restrict__ triangleVertex)
{
    IF_IDX_VALID(size)
    _k_updateTriangleInfo<MemType::GPU>(i, triangleInfo, triangleCenter, triangleNormal, particlePositionPhase, triangleVertex);
}
// Function to call the kernel function for updating triangle info
template <MemType MT>
void callUpdateTriangleInfo(int size, VecArray<TriangleInfo, MT> &triangleInfo, VecArray<vec4r, MT> &triangleCenter, VecArray<vec4r, MT> &triangleNormal, VecArray<vec4r, MT> &particlePositionPhase, VecArray<int4, MT> &triangleVertex)
{
    FILL_CALL_GPU_DEVICE_CODE(updateTriangleInfo, triangleInfo.m_data, triangleCenter.m_data, triangleNormal.m_data, particlePositionPhase.m_data, triangleVertex.m_data);
}
template void callUpdateTriangleInfo<MemType::CPU>(int, VecArray<TriangleInfo, CPU> &, VecArray<vec4r, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &, VecArray<int4, MemType::CPU> &);
template void callUpdateTriangleInfo<MemType::GPU>(int, VecArray<TriangleInfo, GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<int4, MemType::GPU> &);
// Explanation:
// The function _k_updateTriangleInfo updates the triangle information for a given index. It retrieves the vertex indices for the triangle at the given index from the array triangleVertex. It then retrieves the position vectors of the three vertices from the array particlePositionPhase. Using these position vectors, it calculates the normalized triangle normal vector n by taking the cross product of two edge vectors. It also calculates the center of the triangle c as the average of the three vertices. Finally, it stores the triangle center, normal, and information at the given index in the arrays triangleCenter, triangleNormal, and triangleInfo, respectively.

// The kernel function _g_updateTriangleInfo is a wrapper function that calls the _k_updateTriangleInfo function for each valid index in parallel on the GPU.

// The function callUpdateTriangleInfo is a templated function that handles the calling of the kernel function based on the memory type MT. It takes the size of the arrays and references to the triangle info array triangleInfo, triangle center array triangleCenter, triangle normal array triangleNormal, particle position and phase array particlePositionPhase, and triangle vertex array triangleVertex. It fills the GPU device code with the appropriate arguments and calls the kernel function _g_updateTriangleInfo.

//// handle the particle-triangle collision
// This function handles the collision between particles and triangles.
// It takes the index, arrays of particle position and phase values (particlePositionPhase),
// sorted triangle information (sortedTriangleInfo), sorted triangle centers (sortedTriangleCenter),
// sorted triangle vertex indices (sortedTriangleVertex), cell start indices (cellStart), and cell end indices (cellEnd).
template <MemType MT>
__device__ void _k_updateParticleTriangleCollision(int i, vec4r *__restrict__ particlePositionPhase, const TriangleInfo *__restrict__ sortedTriangleInfo, const vec4r *__restrict__ sortedTriangleCenter, const int4 *__restrict__ sortedTriangleVertex, const uint *__restrict__ cellStart, const uint *__restrict__ cellEnd)
{

    const vec4r xph_i = particlePositionPhase[i];
    const uint ph_i = (uint)xph_i.w;
    const vec3r x_i = make_vec3r(xph_i);
    vec3r dx_i = make_vec3r(0.f);
    const Real thres = params.triCollisionThres;
    int cnt = 0;
    // If the particle is fixed, no collision handling is necessary
    if (ph_i == PhaseType::Fixed)
        return;

    //// iterate neighboring cells
    const int3 gridPos = _d_calcGridPos(x_i);
    for (int z = -1; z <= 1; z++)
        for (int y = -1; y <= 1; y++)
            for (int x = -1; x <= 1; x++)
            {
                //Get the grid position of the neighboring cell
                //and calculate the hash value of the cell
                const int3 neighbourPos = gridPos + make_int3(x, y, z);
                const uint gridHash = _d_calcGridHash(neighbourPos);
                const uint startIndex = cellStart[gridHash];
                const uint endIndex = cellEnd[gridHash];
                for (uint j = startIndex; j < endIndex; j++)
                {
                    //// find neighboring triangles
                    const TriangleInfo ti_j = (sortedTriangleInfo[j]);
                    const vec3r tc0 = ti_j.p[0];
                    const vec3r tc1 = ti_j.p[1];
                    const vec3r tc2 = ti_j.p[2];
                    const vec3r tc_j = ti_j.c;
                    const vec3r tn_j = ti_j.n;
                    const vec3r rel = x_i - tc0;

                    //// skip if vert is in the triangle
                    const int4 tverts = sortedTriangleVertex[j];
                    if (i == tverts.x || i == tverts.y || i == tverts.z)
                        continue;

                    //// skip far triangle
                    const Real nd = dot(rel, tn_j);
                    const Real rel2 = dot(rel, rel);
                    if ((ph_i == PhaseType::Deform && nd > 0) || abs(nd) > thres || rel2 > 2 * thres * thres)
                        continue;

                    //// project it in triangle
                    Real w0, w1, w2; //// the interpolation weights
                    vec3r n;         //// the project position and the face normal
                    _d_projectPointInTriangle(tc0, tc1, tc2, x_i, w0, w1, w2, n);
                    // vec3r dx_ij;
                    // {
                    //     Real nd2 = dot(n, n);
                    //     if (nd2 > thres* thres) continue;
                    //     //// for simplicity, we only apply the correction of collision constraint on the particle
                    //     dx_ij = (thres - sqrt(nd2)) * n;
                    // }
                    //// correct
                    // dx_i += dx_ij;
                    // cnt++;
                    Real nd2 = dot(n, n);
                    if (nd2 > thres * thres)
                        continue;

                    // printf("%f, %f, %f, n=%f\n", w0, w1, w2, length(n));

                    vec3r grad = n;
                    // vec3r grads[3] = {-n*w0,-n*w1,-n*w2};
                    Real grad2 = 1 + w0 * w0 + w1 * w1 + w2 * w2;

                    Real lambda = (sqrt(nd2) - thres) / grad2;

                    //// correct
                    dx_i += -lambda * grad;
                    cnt++;
                }
            }

    // Ensure that the collision counter is at least 1 to avoid division by zero
    cnt = max(cnt, 1);
    // particlePositionPhase[i] = make_vec4r(x_i + (1.0 / cnt) * dx_i*params.collisionStiffness, (Real)ph_i);
    particlePositionPhase[i] = make_vec4r(x_i + (1.0 / cnt) * dx_i, (Real)ph_i);
}

DECLARE_GPU_KERNEL_TEMP(updateParticleTriangleCollision, vec4r *__restrict__ particlePositionPhase, const TriangleInfo *__restrict__ sortedTriangleInfo, const vec4r *__restrict__ sortedTriangleCenter, const int4 *__restrict__ sortedTriangleVertex, const uint *__restrict__ cellStart, const uint *__restrict__ cellEnd);
// Kernel function to update particle-triangle collision on GPU
__global__ void _g_updateParticleTriangleCollision(int size, vec4r *__restrict__ particlePositionPhase, const TriangleInfo *__restrict__ sortedTriangleInfo, const vec4r *__restrict__ sortedTriangleCenter, const int4 *__restrict__ sortedTriangleVertex, const uint *__restrict__ cellStart, const uint *__restrict__ cellEnd)
{
    IF_IDX_VALID(size)
    _k_updateParticleTriangleCollision<MemType::GPU>(i, particlePositionPhase, sortedTriangleInfo, sortedTriangleCenter, sortedTriangleVertex, cellStart, cellEnd);
}
// Function to call the kernel function for updating particle-triangle collision
template <MemType MT>
void callUpdateParticleTriangleCollision(int size, VecArray<vec4r, MT> &particlePositionPhase, VecArray<TriangleInfo, MT> &sortedTriangleInfo, VecArray<vec4r, MT> &sortedTriangleCenter, VecArray<int4, MT> &sortedTriangleVertex, VecArray<uint, MT> &cellStart, VecArray<uint, MT> &cellEnd)
{
    FILL_CALL_GPU_DEVICE_CODE(updateParticleTriangleCollision, particlePositionPhase.m_data, sortedTriangleInfo.m_data, sortedTriangleCenter.m_data, sortedTriangleVertex.m_data, cellStart.m_data, cellEnd.m_data);
}
template void callUpdateParticleTriangleCollision<MemType::CPU>(int, VecArray<vec4r, MemType::CPU> &, VecArray<TriangleInfo, CPU> &, VecArray<vec4r, MemType::CPU> &, VecArray<int4, MemType::CPU> &, VecArray<uint, MemType::CPU> &, VecArray<uint, MemType::CPU> &);
template void callUpdateParticleTriangleCollision<MemType::GPU>(int, VecArray<vec4r, MemType::GPU> &, VecArray<TriangleInfo, GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<int4, MemType::GPU> &, VecArray<uint, MemType::GPU> &, VecArray<uint, MemType::GPU> &);
// The function _k_updateParticleTriangleCollision handles the collision between particles and triangles for a given index. It extracts the position and phase value of the particle at the given index from the array particlePositionPhase. It also retrieves the sorted triangle information, center, vertex indices, and cell start/end indices from their respective arrays. The function performs collision detection and correction by iterating over neighboring cells and neighboring triangles. For each neighboring triangle, it checks if the particle is inside the triangle, if the triangle is far from the particle, and if the particle is far from the triangle. If these conditions are not met, it projects the particle onto the triangle and calculates a collision correction based on the projection. Finally, it updates the particle position and phase value with the collision correction.
// The kernel function _g_updateParticleTriangleCollision is a wrapper function that calls the _k_updateParticleTriangleCollision function for each valid index in parallel on the GPU.
//The function callUpdateParticleTriangleCollision is a templated function that handles the calling of the kernel function based on the memory type MT. It takes the size of the arrays and references to the particle position and phase array particlePositionPhase, sorted triangle info array sortedTriangleInfo, sorted triangle center array sortedTriangleCenter, sorted triangle vertex array sortedTriangleVertex, and cell start/end arrays cellStart and cellEnd. It fills the GPU device code with the appropriate arguments and calls the kernel function _g_updateParticleTriangleCollision.

//// handle the particle-triangle collision with atomic function
// This function handles the collision between particles and triangles using an atomic update strategy for thread safety.
// It takes the index, arrays of delta particle position and phase values (deltaParticlePositionPhase), current particle position and phase values (particlePositionPhase),
// sorted triangle information (sortedTriangleInfo), sorted triangle centers (sortedTriangleCenter),
// sorted triangle vertex indices (sortedTriangleVertex), cell start indices (cellStart), and cell end indices (cellEnd).
template <MemType MT>
__device__ void _k_updateParticleTriangleCollisionAtom(int i, vec4r *__restrict__ deltaParticlePositionPhase, vec4r *__restrict__ particlePositionPhase, const TriangleInfo *__restrict__ sortedTriangleInfo, const vec4r *__restrict__ sortedTriangleCenter, const int4 *__restrict__ sortedTriangleVertex, const uint *__restrict__ cellStart, const uint *__restrict__ cellEnd)
{

    const vec4r xph_i = particlePositionPhase[i];
    const uint ph_i = (uint)xph_i.w;
    const vec3r x_i = make_vec3r(xph_i);
    vec3r dx_i = make_vec3r(0.f);
    const Real thres = params.triCollisionThres;
    int cnt = 0;
    const Real thres2 = (params.particleRadius + params.triLength) * (params.particleRadius + params.triLength);
    if (ph_i == PhaseType::Fixed)
        return;

    //// iterate neighboring cells
    const int3 gridPos = _d_calcGridPos(x_i);
    for (int z = -1; z <= 1; z++)
        for (int y = -1; y <= 1; y++)
            for (int x = -1; x <= 1; x++)
            {
                const int3 neighbourPos = gridPos + make_int3(x, y, z);
                const uint gridHash = _d_calcGridHash(neighbourPos);
                const uint startIndex = cellStart[gridHash];
                const uint endIndex = cellEnd[gridHash];
                for (uint j = startIndex; j < endIndex; j++)
                {
                    //// find neighboring triangles
                    const TriangleInfo ti_j = (sortedTriangleInfo[j]);
                    const vec3r tc0 = ti_j.p[0];
                    const vec3r tc1 = ti_j.p[1];
                    const vec3r tc2 = ti_j.p[2];
                    const vec3r tc_j = ti_j.c;
                    const vec3r tn_j = ti_j.n;
                    const vec3r rel = x_i - tc0;

                    //// skip if vert is in the triangle
                    const int4 tverts = sortedTriangleVertex[j];
                    if (i == tverts.x || i == tverts.y || i == tverts.z)
                        continue;

                    //// skip far triangle
                    const Real nd = dot(rel, tn_j);
                    const Real rel2 = dot(rel, rel);
                    // if ((ph_i== PhaseType::Deform && nd >0) || abs(nd) > thres || rel2 > 2 * thres * thres) continue;
                    if ((ph_i == PhaseType::Deform && nd > 0) || abs(nd) > thres || rel2 > thres2)
                        continue;

                    //// project it in triangle
                    Real w0, w1, w2; //// the interpolation weights
                    vec3r n;         //// the project position and the face normal
                    _d_projectPointInTriangle(tc0, tc1, tc2, x_i, w0, w1, w2, n);
                    // vec3r dx_ij;
                    // {
                    //     Real nd2 = dot(n, n);
                    //     if (nd2 > thres* thres) continue;
                    //     //// for simplicity, we only apply the correction of collision constraint on the particle
                    //     dx_ij = (thres - sqrt(nd2)) * n;
                    // }

                    Real nd2 = dot(n, n);
                    if (nd2 > thres * thres)
                        continue;

                    // printf("v=%d,(%f,%f,%f); fn=(%f,%f,%f); fc=(%f,%f,%f)\n", i, x_i.x, x_i.y, x_i.z, tn_j.x, tn_j.y, tn_j.z, tc_j.x, tc_j.y, tc_j.z);

                    vec3r grad = n;
                    vec3r grads[3] = {-n * w0, -n * w1, -n * w2};
                    Real grad2 = 1 + w0 * w0 + w1 * w1 + w2 * w2;

                    Real lambda = (sqrt(nd2) - thres) / grad2;

                    //// correct
                    vec3r corr = -lambda * grad;
                    vec3r corr0 = -lambda * grads[0];
                    vec3r corr1 = -lambda * grads[1];
                    vec3r corr2 = -lambda * grads[2];

                    //// corr0
                    atomicAdd(((Real *)deltaParticlePositionPhase) + 4 * tverts.x + 0, corr0.x);
                    atomicAdd(((Real *)deltaParticlePositionPhase) + 4 * tverts.x + 1, corr0.y);
                    atomicAdd(((Real *)deltaParticlePositionPhase) + 4 * tverts.x + 2, corr0.z);
                    atomicAdd(((Real *)deltaParticlePositionPhase) + 4 * tverts.x + 3, 1.0f);
                    //// corr1
                    atomicAdd(((Real *)deltaParticlePositionPhase) + 4 * tverts.y + 0, corr1.x);
                    atomicAdd(((Real *)deltaParticlePositionPhase) + 4 * tverts.y + 1, corr1.y);
                    atomicAdd(((Real *)deltaParticlePositionPhase) + 4 * tverts.y + 2, corr1.z);
                    atomicAdd(((Real *)deltaParticlePositionPhase) + 4 * tverts.y + 3, 1.0f);
                    //// corr2
                    atomicAdd(((Real *)deltaParticlePositionPhase) + 4 * tverts.z + 0, corr2.x);
                    atomicAdd(((Real *)deltaParticlePositionPhase) + 4 * tverts.z + 1, corr2.y);
                    atomicAdd(((Real *)deltaParticlePositionPhase) + 4 * tverts.z + 2, corr2.z);
                    atomicAdd(((Real *)deltaParticlePositionPhase) + 4 * tverts.z + 3, 1.0f);

                    dx_i += corr;
                    cnt++;
                }
            }

    atomicAdd(((Real *)deltaParticlePositionPhase) + 4 * i + 0, dx_i.x);
    atomicAdd(((Real *)deltaParticlePositionPhase) + 4 * i + 1, dx_i.y);
    atomicAdd(((Real *)deltaParticlePositionPhase) + 4 * i + 2, dx_i.z);
    atomicAdd(((Real *)deltaParticlePositionPhase) + 4 * i + 3, cnt);

    // deltaParticlePositionPhase[i]=make_vec4r(dx_i.x,dx_i.y,dx_i.z,cnt);
    // cnt = max(cnt, 1);
    // newParticlePositionPhase[i] += make_vec4r(x_i + (1.0 / cnt) * dx_i*params.collisionStiffness, (Real)ph_i);
}

DECLARE_GPU_KERNEL_TEMP(updateParticleTriangleCollisionAtom, vec4r *__restrict__ newParticlePositionPhase, vec4r *__restrict__ particlePositionPhase, const TriangleInfo *__restrict__ sortedTriangleInfo, const vec4r *__restrict__ sortedTriangleCenter, const int4 *__restrict__ sortedTriangleVertex, const uint *__restrict__ cellStart, const uint *__restrict__ cellEnd);

__global__ void _g_updateParticleTriangleCollisionAtom(int size, vec4r *__restrict__ newParticlePositionPhase, vec4r *__restrict__ particlePositionPhase, const TriangleInfo *__restrict__ sortedTriangleInfo, const vec4r *__restrict__ sortedTriangleCenter, const int4 *__restrict__ sortedTriangleVertex, const uint *__restrict__ cellStart, const uint *__restrict__ cellEnd)
{
    IF_IDX_VALID(size)
    _k_updateParticleTriangleCollisionAtom<MemType::GPU>(i, newParticlePositionPhase, particlePositionPhase, sortedTriangleInfo, sortedTriangleCenter, sortedTriangleVertex, cellStart, cellEnd);
}

template <MemType MT>
void callUpdateParticleTriangleCollisionAtom(int size, VecArray<vec4r, MT> &newParticlePositionPhase, VecArray<vec4r, MT> &particlePositionPhase, VecArray<TriangleInfo, MT> &sortedTriangleInfo, VecArray<vec4r, MT> &sortedTriangleCenter, VecArray<int4, MT> &sortedTriangleVertex, VecArray<uint, MT> &cellStart, VecArray<uint, MT> &cellEnd)
{
    FILL_CALL_GPU_DEVICE_CODE(updateParticleTriangleCollisionAtom, newParticlePositionPhase.m_data, particlePositionPhase.m_data, sortedTriangleInfo.m_data, sortedTriangleCenter.m_data, sortedTriangleVertex.m_data, cellStart.m_data, cellEnd.m_data);
}
template void callUpdateParticleTriangleCollisionAtom<MemType::CPU>(int, VecArray<vec4r, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &, VecArray<TriangleInfo, CPU> &, VecArray<vec4r, MemType::CPU> &, VecArray<int4, MemType::CPU> &, VecArray<uint, MemType::CPU> &, VecArray<uint, MemType::CPU> &);
template void callUpdateParticleTriangleCollisionAtom<MemType::GPU>(int, VecArray<vec4r, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<TriangleInfo, GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<int4, MemType::GPU> &, VecArray<uint, MemType::GPU> &, VecArray<uint, MemType::GPU> &);
// This set of functions handles the collision between particles and triangles using an atomic update strategy for thread safety.
// The function _k_updateParticleTriangleCollisionAtom is responsible for updating the particle position and phase values based on the collision with neighboring triangles.
// It follows a similar process as the _k_updateParticleTriangleCollision function but uses atomic operations to updatethe correction values (corr) and the counter (cnt) in a thread-safe manner.
//The kernel function _g_updateParticleTriangleCollisionAtom is the GPU implementation of _k_updateParticleTriangleCollisionAtom. It launches multiple threads to process different particle indices in parallel.
//The function callUpdateParticleTriangleCollisionAtom is used to call the GPU kernel function. It takes the size (number of particles) and the necessary arrays as input parameters.
//Overall, these functions ensure that particles and triangles interact correctly by detecting collisions and applying collision corrections atomically to avoid data races.

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__ void reorderDataAndFindCellStartD(uint *cellStart,uint *cellEnd,TriangleInfo *sortedTriangleInfo,vec4r *sortedTriangleCenter,vec4r *sortedTriangleNormal,int4 *sortedTriangleVertex,uint *gridTriangleHash,uint *O2STriangleIndex,uint *S2OTriangleIndex,TriangleInfo *triangleInfo,vec4r *triangleCenter,vec4r *triangleNormal,int4 *triangleVertex,
                                             uint numTriangles)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ uint sharedHash[]; // blockSize + 1 elements
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    uint hash;

    // handle case when no. of particles not multiple of block size
    if (index < numTriangles)
    {
        hash = gridTriangleHash[index];

        // Load hash data into shared memory so that we can look
        // at neighboring triangle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x + 1] = hash;

        if (index > 0 && threadIdx.x == 0)
        {
            // first thread in block must load neighbor triangle hash
            sharedHash[0] = gridTriangleHash[index - 1];
        }
    }

    cg::sync(cta);

    if (index < numTriangles)
    {
        // If this triangle has a different cell index to the previous
        // triangle then it must be the first triangle in the cell,
        // so store the index of this triangle in the cell.
        // As it isn't the first triangle, it must also be the cell end of
        // the previous triangle's cell

        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            cellStart[hash] = index;

            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
        }

        if (index == numTriangles - 1)
        {
            cellEnd[hash] = index + 1;
        }

        //// index mapping
        uint oidx = S2OTriangleIndex[index];
        O2STriangleIndex[oidx] = index;

        //// Use the sorted index to reorder the data
        sortedTriangleInfo[index] = triangleInfo[oidx];
        sortedTriangleCenter[index] = triangleCenter[oidx];
        sortedTriangleNormal[index] = triangleNormal[oidx];
        sortedTriangleVertex[index] = triangleVertex[oidx];
    }
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__ void reorderParticleDataAndFindCellStartD(uint *cellStart,  uint *cellEnd, vec4r *sortedPositionPhase, uint *gridParticleHash, uint *O2SParticleIndex,uint *S2OParticleIndex,vec4r *oldPos,uint numParticles)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ uint sharedHash[]; // blockSize + 1 elements
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    uint hash;

    // handle case when no. of particles not multiple of block size
    if (index < numParticles)
    {
        hash = gridParticleHash[index];

        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x + 1] = hash;

        if (index > 0 && threadIdx.x == 0)
        {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = gridParticleHash[index - 1];
        }
    }

    cg::sync(cta);

    if (index < numParticles)
    {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell

        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            cellStart[hash] = index;

            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
        }

        if (index == numParticles - 1)
        {
            cellEnd[hash] = index + 1;
        }

        // Now use the sorted index to reorder the pos and vel data
        uint oidx = S2OParticleIndex[index];
        O2SParticleIndex[oidx] = index;

        sortedPositionPhase[index] = oldPos[oidx];
    }
}

__global__ void reorderParticleObjectDataAndFindCellStartD(uint *cellStart,uint *cellEnd, vec4r *sortedPositionPhase, uint *sortedObjectIdx,uint *gridParticleHash, uint *O2SParticleIndex,uint *S2OParticleIndex,vec4r *oldPos, uint *oldObjectIdx,uint numParticles)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ uint sharedHash[]; // blockSize + 1 elements
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    uint hash;

    // handle case when no. of particles not multiple of block size
    if (index < numParticles)
    {
        hash = gridParticleHash[index];

        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x + 1] = hash;

        if (index > 0 && threadIdx.x == 0)
        {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = gridParticleHash[index - 1];
        }
    }

    cg::sync(cta);

    if (index < numParticles)
    {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell

        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            cellStart[hash] = index;

            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
        }

        if (index == numParticles - 1)
        {
            cellEnd[hash] = index + 1;
        }

        // Now use the sorted index to reorder the pos and vel data
        uint oidx = S2OParticleIndex[index];
        O2SParticleIndex[oidx] = index;

        sortedPositionPhase[index] = oldPos[oidx];
        sortedObjectIdx[index] = oldObjectIdx[oidx];
    }
}

void reorderDataAndFindCellStart(uint size,uint *cellStart,uint *cellEnd,TriangleInfo *sortedTriangleInfo,vec4r *sortedTriangleCenter,vec4r *sortedTriangleNormal,int4 *sortedTriangleVertex,uint *gridTriangleHash,uint *O2STriangleIndex,uint *S2OTriangleIndex,TriangleInfo *triangleInfo,vec4r *triangleCenter,vec4r *triangleNormal,int4 *triangleVertex,uint numCells)
{

    uint numThreads, numBlocks;
    computeCudaThread(size, PE_CUDA_BLOCKS, numBlocks, numThreads);
    // set all cells to empty
    checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells * sizeof(uint)));

    uint smemSize = sizeof(uint) * (numThreads + 1);
    reorderDataAndFindCellStartD<<<numBlocks, numThreads, smemSize>>>(
        cellStart,
        cellEnd,
        sortedTriangleInfo,
        sortedTriangleCenter,
        sortedTriangleNormal,
        sortedTriangleVertex,
        gridTriangleHash,
        O2STriangleIndex,
        S2OTriangleIndex,
        triangleInfo,
        triangleCenter,
        triangleNormal,
        triangleVertex,
        size);
    getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");
}

void reorderParticleDataAndFindCellStart(uint *cellStart,uint *cellEnd,vec4r *sortedPositionPhase,uint *gridParticleHash,uint *O2SParticleIndex,uint *S2OparticleIndex,vec4r *oldPos,uint numCells,uint numParticles)
{

    uint numThreads, numBlocks;
    computeCudaThread(numParticles, PE_CUDA_BLOCKS, numBlocks, numThreads);
    // set all cells to empty
    checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells * sizeof(uint)));

    uint smemSize = sizeof(uint) * (numThreads + 1);
    reorderParticleDataAndFindCellStartD<<<numBlocks, numThreads, smemSize>>>(
        cellStart,
        cellEnd,
        sortedPositionPhase,
        gridParticleHash,
        O2SParticleIndex,
        S2OparticleIndex,
        oldPos,
        numParticles);
    getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");
}

void reorderParticleObjectDataAndFindCellStart(uint *cellStart,uint *cellEnd,vec4r *sortedPositionPhase,uint *sortedObjectIdx,uint *gridParticleHash,uint *O2SParticleIndex,uint *S2OparticleIndex,vec4r *oldPos,uint *oldObjectIdx,uint numCells,uint numParticles)
{

    uint numThreads, numBlocks;
    computeCudaThread(numParticles, PE_CUDA_BLOCKS, numBlocks, numThreads);
    // set all cells to empty
    checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells * sizeof(uint)));

    uint smemSize = sizeof(uint) * (numThreads + 1);
    reorderParticleObjectDataAndFindCellStartD<<<numBlocks, numThreads, smemSize>>>(
        cellStart,
        cellEnd,
        sortedPositionPhase,
        sortedObjectIdx,
        gridParticleHash,
        O2SParticleIndex,
        S2OparticleIndex,
        oldPos,
        oldObjectIdx,
        numParticles);
    getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");
}

template <MemType MT>
__host__ __device__ void _k_resolvePenetration(int index, vec4r *tempPosition, vec4r *sortedPositionPhase, uint *S2OParticleIndex, uint *cellStart, uint *cellEnd)
{
    //C=|x_ij|-r>=0
    const vec4r xph_i = sortedPositionPhase[index];
    const vec3r x_i = make_vec3r(xph_i);
    const uint ph_i = (uint)xph_i.w;

    // vec3r x_i = sortedPositionPhase[index];
    tempPosition[index] = make_vec4r(x_i, ph_i);

    Real m_i = 1;
    uint oidx_i = S2OParticleIndex[index];
    vec3r deltax = make_vec3r(0.0f);
    int neighbourCount = 0;

    int3 gridPos = _d_calcGridPos(x_i);
    for (int z = -1; z <= 1; z++)
        for (int y = -1; y <= 1; y++)
            for (int x = -1; x <= 1; x++)
            {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                uint gridHash = _d_calcGridHash(neighbourPos);
                uint startIndex = cellStart[gridHash];
                uint endIndex = cellEnd[gridHash];
                for (uint j = startIndex; j < endIndex; j++)
                {
                    // uint ph_j = sortedPhase[j];
                    const vec4r xph_j = sortedPositionPhase[j];
                    // vec3r x_j = sortedPos[index];
                    const vec3r x_j = make_vec3r(xph_j);
                    const uint ph_j = (uint)xph_j.w;
                    uint oidx_j = S2OParticleIndex[j];

                    vec3r rel = x_i - x_j; //sortedPositionPhase[j];
                    Real dist2 = dot(rel, rel);
                    if (j == index || dist2 >= params.peneDist2 || neighbourCount > params.maxNeighbours)
                        continue;
                    Real dist = sqrt(dist2);

                    neighbourCount++;
                    Real m_j = 1;
                    // Real frac=1.0f/(m_i+m_j);
                    // Real frac_i=m_j*frac;
                    // Real frac_j=m_i*frac;
                    // vec3r delta=(dist-params.peneDist)/dist*rel;
                    // vec3r deltax_i=-frac_i*delta;
                    // Real frac=1.0f/(m_i+m_j);
                    // Real frac_i=m_j*frac;
                    deltax += -m_j * (dist - params.peneDist) / (dist * (m_i + m_j)) * rel;
                }
            }
    // if (neighbourCount > 0) tempPosition[index] = make_vec4r(x_i + deltax * (1.f / neighbourCount),ph_i);
    tempPosition[S2OParticleIndex[index]] = make_vec4r(x_i + deltax, ph_i);
}

DECLARE_KERNEL_TEMP(resolvePenetration, vec4r *tempPosition, vec4r *sortedPositionPhase, uint *S2OParticleIndex, uint *cellStart, uint *cellEnd);

__global__ void _g_resolvePenetration(int size, vec4r *tempPosition, vec4r *sortedPositionPhase, uint *S2OParticleIndex, uint *cellStart, uint *cellEnd)
{
    IF_IDX_VALID(size)
    _k_resolvePenetration<MemType::GPU>(i, tempPosition, sortedPositionPhase, S2OParticleIndex, cellStart, cellEnd);
}

template <MemType MT>
void callResolvePenetration(int size, VecArray<vec4r, MT> &tempPosition, VecArray<vec4r, MT> &sortedPositionPhase, VecArray<uint, MT> &S2OParticleIndex, VecArray<uint, MT> &cellStart, VecArray<uint, MT> &cellEnd)
{
    FILL_CALL_DEVICE_CODE(resolvePenetration, tempPosition.m_data, sortedPositionPhase.m_data, S2OParticleIndex.m_data, cellStart.m_data, cellEnd.m_data);
}
template void callResolvePenetration<MemType::CPU>(int, VecArray<vec4r, CPU> &, VecArray<vec4r, CPU> &, VecArray<uint, CPU> &, VecArray<uint, CPU> &, VecArray<uint, CPU> &);
template void callResolvePenetration<MemType::GPU>(int, VecArray<vec4r, GPU> &, VecArray<vec4r, GPU> &, VecArray<uint, GPU> &, VecArray<uint, GPU> &, VecArray<uint, GPU> &);

template <MemType MT>
__host__ __device__ void _k_resolvePenetrationDiffObj(int index, vec4r *tempPosition, vec4r *sortedPositionPhase, uint *sortedObjectIdx, uint *S2OParticleIndex, uint *cellStart, uint *cellEnd)
{
    //C=|x_ij|-r>=0
    const vec4r xph_i = sortedPositionPhase[index];
    const vec3r x_i = make_vec3r(xph_i);
    const uint ph_i = (uint)xph_i.w;
    const uint objid_i = sortedObjectIdx[index];

    // vec3r x_i = sortedPositionPhase[index];
    // tempPosition[index] = make_vec4r(x_i,ph_i);

    Real m_i = 1;
    uint oidx_i = S2OParticleIndex[index];
    vec3r deltax = make_vec3r(0.0f);
    int neighbourCount = 0;

    int3 gridPos = _d_calcGridPos(x_i);
    for (int z = -1; z <= 1; z++)
        for (int y = -1; y <= 1; y++)
            for (int x = -1; x <= 1; x++)
            {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                uint gridHash = _d_calcGridHash(neighbourPos);
                uint startIndex = cellStart[gridHash];
                uint endIndex = cellEnd[gridHash];
                for (uint j = startIndex; j < endIndex; j++)
                {
                    // uint ph_j = sortedPhase[j];
                    const vec4r xph_j = sortedPositionPhase[j];
                    // vec3r x_j = sortedPos[index];
                    const vec3r x_j = make_vec3r(xph_j);
                    const uint ph_j = (uint)xph_j.w;
                    const uint objid_j = sortedObjectIdx[j];
                    uint oidx_j = S2OParticleIndex[j];

                    vec3r rel = x_i - x_j; //sortedPositionPhase[j];
                    Real dist2 = dot(rel, rel);
                    if (objid_i == objid_j || j == index || dist2 >= params.peneDist2 || neighbourCount > params.maxNeighbours)
                        continue;
                    Real dist = sqrt(dist2);

                    neighbourCount++;
                    Real m_j = 1;
                    // Real frac=1.0f/(m_i+m_j);
                    // Real frac_i=m_j*frac;
                    // Real frac_j=m_i*frac;
                    // vec3r delta=(dist-params.peneDist)/dist*rel;
                    // vec3r deltax_i=-frac_i*delta;
                    // Real frac=1.0f/(m_i+m_j);
                    // Real frac_i=m_j*frac;
                    deltax += -m_j * (dist - params.peneDist) / (dist * (m_i + m_j)) * rel;
                }
            }
    // if (neighbourCount > 0) tempPosition[index] = make_vec4r(x_i + deltax * (1.f / neighbourCount),ph_i);
    tempPosition[S2OParticleIndex[index]] = make_vec4r(x_i + deltax, ph_i);
}

DECLARE_KERNEL_TEMP(resolvePenetrationDiffObj, vec4r *tempPosition, vec4r *sortedPositionPhase, uint *sortedObjectIdx, uint *S2OParticleIndex, uint *cellStart, uint *cellEnd);

__global__ void _g_resolvePenetrationDiffObj(int size, vec4r *tempPosition,
                                             vec4r *sortedPositionPhase, uint *sortedObjectIdx, uint *S2OParticleIndex, uint *cellStart, uint *cellEnd)
{
    IF_IDX_VALID(size)
    _k_resolvePenetrationDiffObj<MemType::GPU>(i, tempPosition, sortedPositionPhase, sortedObjectIdx, S2OParticleIndex, cellStart, cellEnd);
}

template <MemType MT>
void callResolvePenetrationDiffObj(int size, VecArray<vec4r, MT> &tempPosition, VecArray<vec4r, MT> &sortedPositionPhase, VecArray<uint, MT> &sortedObjectIdx, VecArray<uint, MT> &S2OParticleIndex, VecArray<uint, MT> &cellStart, VecArray<uint, MT> &cellEnd)
{
    FILL_CALL_DEVICE_CODE(resolvePenetrationDiffObj, tempPosition.m_data, sortedPositionPhase.m_data, sortedObjectIdx.m_data, S2OParticleIndex.m_data, cellStart.m_data, cellEnd.m_data);
}
template void callResolvePenetrationDiffObj<MemType::CPU>(int, VecArray<vec4r, CPU> &, VecArray<vec4r, CPU> &, VecArray<uint, CPU> &, VecArray<uint, CPU> &, VecArray<uint, CPU> &, VecArray<uint, CPU> &);
template void callResolvePenetrationDiffObj<MemType::GPU>(int, VecArray<vec4r, GPU> &, VecArray<vec4r, GPU> &, VecArray<uint, GPU> &, VecArray<uint, GPU> &, VecArray<uint, GPU> &, VecArray<uint, GPU> &);

PHYS_NAMESPACE_END