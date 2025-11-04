#include "object/particle_fluid.h"
// // #include "object/particle_fluid_util.h"
// #include "thrust/device_ptr.h"
// #include "thrust/for_each.h"
// #include "thrust/iterator/zip_iterator.h"
// #include "thrust/sort.h"
// #include <cooperative_groups.h>

// namespace cg = cooperative_groups;

PHYS_NAMESPACE_BEGIN



// template<MemType MT>
// __device__ void _k_updateNbrList(int index, NbrList* __restrict__ nbrLists, const vec4r* __restrict__ sortedPositionPhase, const uint* __restrict__ S2OParticleIndex, const uint* __restrict__ cellStart, const uint* __restrict__ cellEnd) {
//     // read particle data from sorted arrays
//     const vec4r xph_i = sortedPositionPhase[index];
//     const uint ph_i = (uint)xph_i.w;
//     const vec3r x_i = make_vec3r(xph_i);
//     const Real h=params.h;
//     const Real h2=params.h2;
//     NbrList nl;

//     const int3 gridPos = _d_calcGridPos(x_i);
//     for (int z = -1; z <= 1; z++)
//     for (int y = -1; y <= 1; y++)
//     for (int x = -1; x <= 1; x++) {
//         const int3 neighbourPos = gridPos + make_int3(x, y, z);
//         const uint gridHash = _d_calcGridHash(neighbourPos);
//         const uint startIndex = cellStart[gridHash];
//         const uint endIndex = cellEnd[gridHash];
//         for (uint j = startIndex; j < endIndex; j++) {
//             const vec4r xph_j = sortedPositionPhase[j];
//             const uint ph_j = (uint)xph_j.w;
//             const vec3r x_j = make_vec3r(xph_j);
//             // uint ph_j = sortedPhase[j];
//             // vec3r rel = x_i - sortedPos[j];
//             const vec3r rel = x_i - x_j;
//             const Real dist2 = dot(rel,rel);
//             if (dist2 > h2 || j == index)
//                 continue;
//             const Real dist = __fsqrt_rn(dist2);
//             nl.add(j,dist);
//         }
//     }
//     nbrLists[index]=nl;
// }

// DECLARE_GPU_KERNEL_TEMP(updateNbrList, NbrList* __restrict__ nbrLists, const vec4r* __restrict__ sortedPositionPhase, const uint* __restrict__ S2OParticleIndex, const uint* __restrict__ cellStart, const uint* __restrict__ cellEnd);

// __global__ void _g_updateNbrList(int size, NbrList* __restrict__ nbrLists, const vec4r* __restrict__ sortedPositionPhase, const uint* __restrict__ S2OParticleIndex, const uint* __restrict__ cellStart, const uint* __restrict__ cellEnd) {
//     IF_IDX_VALID(size) _k_updateNbrList<MemType::GPU>(i, nbrLists, sortedPositionPhase, S2OParticleIndex, cellStart, cellEnd);
// }

// template<MemType MT>
// void callUpdateNbrList(int size, VecArray<NbrList,MT>& nbrLists, VecArray<vec4r,MT>& sortedPositionPhase,VecArray<uint,MT>& S2OParticleIndex, VecArray<uint,MT>& cellStart, VecArray<uint,MT>& cellEnd) {
//     FILL_CALL_GPU_DEVICE_CODE(updateNbrList, nbrLists.m_data, sortedPositionPhase.m_data, S2OParticleIndex.m_data, cellStart.m_data, cellEnd.m_data);
// }
// template void callUpdateNbrList<MemType::CPU>(int, VecArray<NbrList,MemType::CPU>&, VecArray<vec4r,MemType::CPU>&, VecArray<uint,MemType::CPU>&, VecArray<uint,MemType::CPU>&, VecArray<uint,MemType::CPU>&);
// template void callUpdateNbrList<MemType::GPU>(int, VecArray<NbrList,MemType::GPU>&, VecArray<vec4r,MemType::GPU>&, VecArray<uint,MemType::GPU>&, VecArray<uint,MemType::GPU>&, VecArray<uint,MemType::GPU>&);



// template<MemType MT>
// __device__ void _k_updateLambdaFast(int index, Real* __restrict__ lambda, Real* __restrict__ invDensity, const NbrList* __restrict__ nbrLists, const vec4r* __restrict__ sortedPositionPhase) {
//     // read particle data from sorted arrays
//     const vec4r xph_i = sortedPositionPhase[index];
//     const uint ph_i = (uint)xph_i.w;
//     const vec3r x_i = make_vec3r(xph_i);
//     Real inv_rho0_i=params.invRho0[(uint)ph_i];

//     Real rho_i = params.zeroPoly6 * params.pmass[ph_i];
//     Real sumGradC2 = params.lambdaRelaxation;
//     const Real h=params.h;
//     const Real h2=params.h2;
//     const Real poly6Coef=params.poly6Coef;
//     const Real spikyGradCoef=params.spikyGradCoef;

//     vec3r gradC_i = make_vec3r(0.f);
//     Real curLambda = 0.f;

//     const NbrList& nbrs=nbrLists[index];
//     for (int jj=0;jj<nbrs.cnt;jj++){
//         int j=0;
//         Real dist=0;
//         nbrs.get(jj,j,dist);
//         if(!_d_isLiquid(ph_i)) return;

//         const vec4r xph_j = sortedPositionPhase[j];
//         const uint ph_j = (uint)xph_j.w;
//         const vec3r x_j = make_vec3r(xph_j);
//         const vec3r rel = x_i - x_j;
        
//         const Real m_j = params.pmass[ph_j];
//         rho_i += _d_poly6(dist, h2, poly6Coef) * m_j;

//         // cal gradCj
//         const Real tmp = h - dist;
//         const vec3r wSpiky = (__fdividef(spikyGradCoef, dist+1e-8f) * tmp * tmp) * rel;
//         // vec3r gradC_j = -m_j*inv_rho0_i*_d_spikyGrad(rel, dist, h, spikyGradCoef);
//         const vec3r gradC_j = -m_j*inv_rho0_i*wSpiky;
        
//         sumGradC2 += dot(gradC_j, gradC_j);
//         gradC_i -= gradC_j;        
//     }

//     const Real C = max(0.f, (rho_i * inv_rho0_i) - 1.f);
    
//     //output
//     lambda[index] = -__fdividef(C,(sumGradC2+dot(gradC_i, gradC_i)));
//     invDensity[index] = __fdividef(1.0f,rho_i);
// }

// DECLARE_GPU_KERNEL_TEMP(updateLambdaFast, Real* __restrict__ lambda, Real* __restrict__ invDensity, const NbrList* __restrict__ nbrLists, const vec4r* __restrict__ sortedPositionPhase);

// __global__ void _g_updateLambdaFast(int size, Real* __restrict__ lambda, Real* __restrict__ invDensity, const NbrList* __restrict__ nbrLists, const vec4r* __restrict__ sortedPositionPhase) {
//     IF_IDX_VALID(size) _k_updateLambdaFast<MemType::GPU>(i, lambda, invDensity, nbrLists, sortedPositionPhase);
// }

// template<MemType MT>
// void callUpdateLambdaFast(int size, VecArray<Real,MT>& lambda, VecArray<Real,MT>& invDensity, VecArray<NbrList,MT>& nbrLists, VecArray<vec4r,MT>& sortedPositionPhase) {
//     FILL_CALL_GPU_DEVICE_CODE(updateLambdaFast, lambda.m_data, invDensity.m_data, nbrLists.m_data, sortedPositionPhase.m_data);
// }
// template void callUpdateLambdaFast<MemType::CPU>(int, VecArray<Real,MemType::CPU>&, VecArray<Real,MemType::CPU>&, VecArray<NbrList,CPU>&, VecArray<vec4r,MemType::CPU>&);
// template void callUpdateLambdaFast<MemType::GPU>(int, VecArray<Real,MemType::GPU>&, VecArray<Real,MemType::GPU>&, VecArray<NbrList,GPU>&, VecArray<vec4r,MemType::GPU>&);

// //// solve fluid
// template<MemType MT>
// __device__ void _k_solveFluidFast(int index, vec4r* __restrict__ normal, vec4r* __restrict__ newPositionPhase, const Real* __restrict__ lambda, const Real* __restrict__ invDensity, const NbrList* __restrict__ nbrLists, const vec4r* __restrict__ sortedPositionPhase) {

//     const vec4r xph_i = sortedPositionPhase[index];
//     // vec3r x_i = sortedPos[index];
//     const vec3r x_i = make_vec3r(xph_i);
//     const uint ph_i = (uint)xph_i.w;
//     newPositionPhase[index]=xph_i;

//     // uint ph_i = sortedPhase[index];
//     if(!_d_isLiquid(ph_i)) return;

//     const Real lambda_i=lambda[index];
//     const Real inv_rho0_i=params.invRho0[(uint)ph_i];
//     const Real h=params.h;
//     const Real h2=params.h2;
//     const Real poly6Coef=params.poly6Coef;
//     const Real spikyGradCoef=params.spikyGradCoef;
//     vec3r deltap = make_vec3r(0.0f);

    
//     const NbrList& nbrs=nbrLists[index];
//     for (int jj=0;jj<nbrs.cnt;jj++){
//         int j=0;
//         Real dist=0;
//         nbrs.get(jj,j,dist);

//         if (j == index) continue;

//         const vec4r xph_j = sortedPositionPhase[j];
//         const uint ph_j = (uint)xph_j.w;
//         const vec3r x_j = make_vec3r(xph_j);
//         const vec3r rel = x_i - x_j;

//         const Real tmp = h - dist;
//         const vec3r wSpiky = (__fdividef(spikyGradCoef, dist+1e-8f) * tmp * tmp) * rel;

//         deltap += params.pmass[ph_j] * (lambda_i + lambda[j]) * wSpiky;
//     }
//     newPositionPhase[index]=make_vec4r(_d_enforceBoundaryLocal(x_i+deltap * inv_rho0_i),xph_i.w);
// }


// DECLARE_GPU_KERNEL_TEMP(solveFluidFast, vec4r* __restrict__ normal, vec4r* __restrict__ newPositionPhase, const Real* __restrict__ lambda, const Real* __restrict__ invDensity, const NbrList* __restrict__ nbrLists, const vec4r* __restrict__ sortedPositionPhase);

// __global__ void _g_solveFluidFast(int size, vec4r* __restrict__ normal, vec4r* __restrict__ newPositionPhase, const Real* __restrict__ lambda, const Real* __restrict__ invDensity, const NbrList* __restrict__ nbrLists, const vec4r* __restrict__ sortedPositionPhase) {
//     IF_IDX_VALID(size) _k_solveFluidFast<MemType::GPU>(i, normal, newPositionPhase, lambda, invDensity, nbrLists, sortedPositionPhase);
// }

// template<MemType MT>
// void callSolveFluidFast(int size, VecArray<vec4r,MT>& normal, VecArray<vec4r,MT>& newPositionPhase, VecArray<Real,MT>& lambda, VecArray<Real,MT>& invDensity, VecArray<NbrList,MT>& nbrLists, VecArray<vec4r,MT>& sortedPositionPhase) {
//     FILL_CALL_GPU_DEVICE_CODE(solveFluidFast, normal.m_data, newPositionPhase.m_data, lambda.m_data, invDensity.m_data, nbrLists.m_data, sortedPositionPhase.m_data);
// }
// template void callSolveFluidFast<MemType::CPU>(int, VecArray<vec4r,MemType::CPU>&, VecArray<vec4r,MemType::CPU>&, VecArray<Real,MemType::CPU>&, VecArray<Real,MemType::CPU>&, VecArray<NbrList,CPU>&, VecArray<vec4r,MemType::CPU>&);
// template void callSolveFluidFast<MemType::GPU>(int, VecArray<vec4r,MemType::GPU>&, VecArray<vec4r,MemType::GPU>&, VecArray<Real,MemType::GPU>&, VecArray<Real,MemType::GPU>&, VecArray<NbrList,GPU>&, VecArray<vec4r,MemType::GPU>&);


PHYS_NAMESPACE_END