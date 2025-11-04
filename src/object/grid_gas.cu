#include "object/grid_gas.h"
#include "curand_kernel.h"
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"
#include "object/particle_fluid_util.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

PHYS_NAMESPACE_BEGIN

__global__ void _g_generateSmoke(int size, int3 source, int radius, Real newDensity, Real* __restrict__ density, vec3r newVelocity, vec3r* __restrict__ velocity){
    IF_IDX_VALID(size) {
        int3 gridIdx = _d_getGridIdx(i);
        // Check if the grid index is valid
        if (_d_isValid(gridIdx))
        {
            // Calculate the squared distance between the grid index and the source
            int distance = length(make_vec3r(gridIdx - source));

            // Check if the distance is within the radius
            if (distance < radius) {
                // If within the radius, set the density to the new density
                density[i] = newDensity;
                velocity[i] = newVelocity;
            }
        }
    }
}

void GenerateSmoke(int size, int3 source, int radius, Real newDensity, VecArray<Real, MemType::GPU>& density, vec3r newVelocity, VecArray<vec3r, MemType::GPU> velocity){
    FILL_CALL_GPU_DEVICE_CODE(generateSmoke, source, radius, newDensity, density.m_data, newVelocity, velocity.m_data);
    cudaDeviceSynchronize();
}

__global__ void _g_addBuoyancy(int size, Real dt, vec3r buoyancyDirection, Real* __restrict__ density, vec3r* __restrict__ velocity){
    IF_IDX_VALID(size) {
        int3 gridIdx = _d_getGridIdx(i);
        // Check if the grid index is valid
        if (_d_isValid(gridIdx))
        {
            // add buoyancy force
            velocity[i] += buoyancyDirection * density[i] * gridParams.kbuoyancy * dt;
            return;
        }
    }
}

void AddBuoyancy(int size, Real dt, vec3r buoyancyDirection, VecArray<Real, MemType::GPU>& density, VecArray<vec3r, MemType::GPU>& velocity)
{
    FILL_CALL_GPU_DEVICE_CODE(addBuoyancy, dt, buoyancyDirection, density.m_data, velocity.m_data);
    cudaDeviceSynchronize();
}

__global__ void _g_addWind(int size, Real dt, vec3r windDirection, Real windStrength, Real* __restrict__ density, vec3r* __restrict__ velocity) {
    IF_IDX_VALID(size) {
        int3 gridIdx = _d_getGridIdx(i);
        // Check if the grid index is valid
        if (_d_isValid(gridIdx))
        {
            // add buoyancy force
            velocity[i] += windDirection * windStrength * density[i] * gridParams.kbuoyancy * dt;
            return;
        }
    }
}

void AddWind(int size, Real dt, vec3r windDirection, Real windStrength, VecArray<Real, MemType::GPU>& density, VecArray<vec3r, MemType::GPU>& velocity) {
    FILL_CALL_GPU_DEVICE_CODE(addWind, dt, windDirection, windStrength, density.m_data, velocity.m_data);
    cudaDeviceSynchronize();
}

__global__ void _g_lockRigid(int size, Real* __restrict__ density, vec3r* __restrict__ velocity, bool* __restrict__ rigidFlag) {
    IF_IDX_VALID(size) {
        int3 gridIdx = _d_getGridIdx(i);
        // Check if the grid index is valid
        if (_d_isValid(gridIdx) && rigidFlag[i])
        {
            density[i] = 0;
            velocity[i] = make_vec3r(0);
            return;
        }
    }
}

void LockRigid(int size, VecArray<Real, MemType::GPU>& density, VecArray<vec3r, MemType::GPU>& velocity, VecArray<bool, MemType::GPU>& rigidFlag) {
    FILL_CALL_GPU_DEVICE_CODE(lockRigid, density.m_data, velocity.m_data, rigidFlag.m_data);
    cudaDeviceSynchronize();
}

__global__ void _g_calculateCurl(int size, vec3r* __restrict__ velocity, vec3r* __restrict__ curl, bool* __restrict__ rigidFlag){
    IF_IDX_VALID(size) {
        int3 gridIdx = _d_getGridIdx(i);
        // Check if the grid index is valid
        if (_d_isValid(gridIdx)&&!rigidFlag[i])
        {
            // calculate curl
            curl[i] = make_vec3r(
                velocity[_d_getIdx(make_int3(gridIdx.x, gridIdx.y+1, gridIdx.z))].z - velocity[_d_getIdx(make_int3(gridIdx.x, gridIdx.y-1, gridIdx.z))].z - velocity[_d_getIdx(make_int3(gridIdx.x, gridIdx.y, gridIdx.z+1))].y + velocity[_d_getIdx(make_int3(gridIdx.x, gridIdx.y, gridIdx.z-1))].y,
                velocity[_d_getIdx(make_int3(gridIdx.x, gridIdx.y, gridIdx.z+1))].x - velocity[_d_getIdx(make_int3(gridIdx.x, gridIdx.y, gridIdx.z-1))].x - velocity[_d_getIdx(make_int3(gridIdx.x+1, gridIdx.y, gridIdx.z))].z + velocity[_d_getIdx(make_int3(gridIdx.x-1, gridIdx.y, gridIdx.z))].z,
                velocity[_d_getIdx(make_int3(gridIdx.x+1, gridIdx.y, gridIdx.z))].y - velocity[_d_getIdx(make_int3(gridIdx.x-1, gridIdx.y, gridIdx.z))].y - velocity[_d_getIdx(make_int3(gridIdx.x, gridIdx.y+1, gridIdx.z))].x + velocity[_d_getIdx(make_int3(gridIdx.x, gridIdx.y-1, gridIdx.z))].x
            ) * 0.5;
            return;
        }
        else {
            curl[i] = make_vec3r(0);
        }
    }
}

__global__ void _g_vorticityConfinement(int size, Real dt, vec3r* __restrict__ velocity, vec3r* __restrict__ curl, bool* __restrict__ rigidFlag){
    IF_IDX_VALID(size) {
        int3 gridIdx = _d_getGridIdx(i);
        // Check if the grid index is valid
        if (_d_isValid(gridIdx)&&!rigidFlag[i])
        {
            Real nX = (length(curl[_d_getIdx(make_int3(gridIdx.x+1, gridIdx.y, gridIdx.z))]) - length(curl[_d_getIdx(make_int3(gridIdx.x-1, gridIdx.y, gridIdx.z))])) * 0.5f;
            Real nY = (length(curl[_d_getIdx(make_int3(gridIdx.x, gridIdx.y+1, gridIdx.z))]) - length(curl[_d_getIdx(make_int3(gridIdx.x, gridIdx.y-1, gridIdx.z))])) * 0.5f;
            Real nZ = (length(curl[_d_getIdx(make_int3(gridIdx.x, gridIdx.y, gridIdx.z+1))]) - length(curl[_d_getIdx(make_int3(gridIdx.x, gridIdx.y, gridIdx.z-1))])) * 0.5f;
            Real len1 = 1.0f/(sqrtf(nX*nX + nY*nY + nZ*nZ)+1e-7f);
                
            nX *= len1;
            nY *= len1;
            nZ *= len1;

            velocity[i] += cross(make_vec3r(nX, nY, nZ), curl[i]) * dt * gridParams.vc_eps;
        }
    }
}

void VorticityConfinement(int size, Real dt, VecArray<vec3r, MemType::GPU>& velocity, VecArray<vec3r, MemType::GPU>& curl, VecArray<bool, MemType::GPU>& rigidFlag)
{
    FILL_CALL_GPU_DEVICE_CODE(calculateCurl, velocity.m_data, curl.m_data, rigidFlag.m_data);
    cudaDeviceSynchronize();


    FILL_CALL_GPU_DEVICE_CODE(vorticityConfinement, dt, velocity.m_data, curl.m_data, rigidFlag.m_data);
    cudaDeviceSynchronize();

}

__global__ void _g_diffuseVelocity(int size, Real a, vec3r* __restrict__ velocity, vec3r* __restrict__ tempVelocity, int parity, bool* __restrict__ rigidFlag){
    IF_IDX_VALID(size) {
        int3 gridIdx = _d_getGridIdx(i);
        // Check if the grid index is valid
        if(_d_isValid(gridIdx) && (gridIdx.x + gridIdx.y + gridIdx.z) % 2 == parity&&!rigidFlag[i]){
            tempVelocity[i] = (velocity[i] + a * (tempVelocity[_d_getIdx(make_int3(gridIdx.x+1, gridIdx.y, gridIdx.z))] + tempVelocity[_d_getIdx(make_int3(gridIdx.x-1, gridIdx.y, gridIdx.z))] + tempVelocity[_d_getIdx(make_int3(gridIdx.x, gridIdx.y+1, gridIdx.z))] + tempVelocity[_d_getIdx(make_int3(gridIdx.x, gridIdx.y-1, gridIdx.z))] + tempVelocity[_d_getIdx(make_int3(gridIdx.x, gridIdx.y, gridIdx.z+1))] + tempVelocity[_d_getIdx(make_int3(gridIdx.x, gridIdx.y, gridIdx.z-1))])) / (1 + 6.0f * a);
        }
    }
}

void DiffuseVelocity(int size, Real dt, VecArray<vec3r, MemType::GPU>& velocity, VecArray<vec3r, MemType::GPU>& tempVelocity, VecArray<bool, MemType::GPU>& rigidFlag){
    Real coef = dt * hGridParams.kvorticity * (hGridParams.gridSize.x-2) * (hGridParams.gridSize.y-2) * (hGridParams.gridSize.z-2);
    //printf("%f\n", coef);
    for(int i = 0; i < 20; i++){
        FILL_CALL_GPU_DEVICE_CODE(diffuseVelocity, coef, velocity.m_data, tempVelocity.m_data, 0, rigidFlag.m_data);
        cudaDeviceSynchronize();

        FILL_CALL_GPU_DEVICE_CODE(diffuseVelocity, coef, velocity.m_data, tempVelocity.m_data, 1, rigidFlag.m_data);
        cudaDeviceSynchronize();
    }
    velocity.swap(tempVelocity);
}

__global__ void _g_calculateDivergence(int size, vec3r* __restrict__ velocity, Real* __restrict__ divergence, bool* __restrict__ rigidFlag){
    IF_IDX_VALID(size) {
        int3 gridIdx = _d_getGridIdx(i);
        // Check if the grid index is valid
        if(_d_isValid(gridIdx)&&!rigidFlag[i]){
            divergence[i] = -(velocity[_d_getIdx(make_int3(gridIdx.x+1, gridIdx.y, gridIdx.z))].x - velocity[_d_getIdx(make_int3(gridIdx.x-1, gridIdx.y, gridIdx.z))].x + velocity[_d_getIdx(make_int3(gridIdx.x, gridIdx.y+1, gridIdx.z))].y - velocity[_d_getIdx(make_int3(gridIdx.x, gridIdx.y-1, gridIdx.z))].y + velocity[_d_getIdx(make_int3(gridIdx.x, gridIdx.y, gridIdx.z+1))].z - velocity[_d_getIdx(make_int3(gridIdx.x, gridIdx.y, gridIdx.z-1))].z) / 3.0f / (gridParams.gridSize.x-2);
        }
    }
}

__global__ void _g_calculatePressure(int size, Real* __restrict__ divergence, Real* __restrict__ pressure, bool* __restrict__ rigidFlag){
    IF_IDX_VALID(size) {
        int3 gridIdx = _d_getGridIdx(i);
        // Check if the grid index is valid
        if(_d_isValid(gridIdx)&&!rigidFlag[i]){
            pressure[i] = (divergence[i] + 
            pressure[_d_getIdx(make_int3(gridIdx.x-1, gridIdx.y, gridIdx.z))] + pressure[_d_getIdx(make_int3(gridIdx.x+1, gridIdx.y, gridIdx.z))] + 
            pressure[_d_getIdx(make_int3(gridIdx.x, gridIdx.y-1, gridIdx.z))] + pressure[_d_getIdx(make_int3(gridIdx.x, gridIdx.y+1, gridIdx.z))] +
            pressure[_d_getIdx(make_int3(gridIdx.x, gridIdx.y, gridIdx.z-1))] + pressure[_d_getIdx(make_int3(gridIdx.x, gridIdx.y, gridIdx.z+1))]
            )/6;
        }
    }
}

__global__ void _g_project(int size, Real* __restrict__ pressure, vec3r* __restrict__ velocity, bool* __restrict__ rigidFlag){
    IF_IDX_VALID(size) {
        int3 gridIdx = _d_getGridIdx(i);
        // Check if the grid index is valid
        if(_d_isValid(gridIdx)&&!rigidFlag[i]){
            velocity[i] -= make_vec3r(
                pressure[_d_getIdx(make_int3(gridIdx.x+1, gridIdx.y, gridIdx.z))] - pressure[_d_getIdx(make_int3(gridIdx.x-1, gridIdx.y, gridIdx.z))],
                pressure[_d_getIdx(make_int3(gridIdx.x, gridIdx.y+1, gridIdx.z))] - pressure[_d_getIdx(make_int3(gridIdx.x, gridIdx.y-1, gridIdx.z))],
                pressure[_d_getIdx(make_int3(gridIdx.x, gridIdx.y, gridIdx.z+1))] - pressure[_d_getIdx(make_int3(gridIdx.x, gridIdx.y, gridIdx.z-1))]
            ) /3.0f * (gridParams.gridSize.x-2);
        }
    }
}
void Project(int size, VecArray<vec3r, MemType::GPU>& velocity, VecArray<Real, MemType::GPU>& divergence, VecArray<Real, MemType::GPU>& pressure, VecArray<bool, MemType::GPU>& rigidFlag){
    {
        FILL_CALL_GPU_DEVICE_CODE(calculateDivergence, velocity.m_data, divergence.m_data, rigidFlag.m_data);
        cudaDeviceSynchronize();
        pressure.fill(size, 0.0f);
    }

    for(int i = 0; i < 20; i++){
        FILL_CALL_GPU_DEVICE_CODE(calculatePressure, divergence.m_data, pressure.m_data, rigidFlag.m_data);
        cudaDeviceSynchronize();
    }

    {
        FILL_CALL_GPU_DEVICE_CODE(project, pressure.m_data, velocity.m_data, rigidFlag.m_data);
        cudaDeviceSynchronize();
    }

}

__global__ void _g_advectVelocity1(int size, Real dt, vec3r* __restrict__ velocity, vec3r* __restrict__ tempVelocity, bool* __restrict__ rigidFlag){
    IF_IDX_VALID(size) {
        int3 gridIdx = _d_getGridIdx(i);
        // Check if the grid index is valid
        if(_d_isValid(gridIdx)&&!rigidFlag[i]){
            Real xx = gridIdx.x - gridParams.gridSize.x * dt * velocity[i].x;
            Real yy = gridIdx.y - gridParams.gridSize.y * dt * velocity[i].y;
            Real zz = gridIdx.z - gridParams.gridSize.z * dt * velocity[i].z;
            if(xx < 0.5f) xx = 0.5f; if(xx > gridParams.gridSize.x - 1.5f) xx = gridParams.gridSize.x - 1.5f;
            if(yy < 0.5f) yy = 0.5f; if(yy > gridParams.gridSize.y - 1.5f) yy = gridParams.gridSize.y - 1.5f;
            if(zz < 0.5f) zz = 0.5f; if(zz > gridParams.gridSize.z - 1.5f) zz = gridParams.gridSize.z - 1.5f;
            int x0 = (int)xx, y0 = (int)yy, z0 = (int)zz;
            int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
            Real sx1 = xx - x0, sx0 = 1.0f - sx1;
            Real sy1 = yy - y0, sy0 = 1.0f - sy1;
            Real sz1 = zz - z0, sz0 = 1.0f - sz1;
            vec3r v0 = sx0 * (sy0  * velocity[_d_getIdx(make_int3(x0, y0, z0))] + sy1 * velocity[_d_getIdx(make_int3(x0, y1, z0))]) + sx1 * (sy0 * velocity[_d_getIdx(make_int3(x1, y0, z0))] + sy1 * velocity[_d_getIdx(make_int3(x1, y1, z0))]);
            vec3r v1 = sx0 * (sy0  * velocity[_d_getIdx(make_int3(x0, y0, z1))] + sy1 * velocity[_d_getIdx(make_int3(x0, y1, z1))]) + sx1 * (sy0 * velocity[_d_getIdx(make_int3(x1, y0, z1))] + sy1 * velocity[_d_getIdx(make_int3(x1, y1, z1))]);
            tempVelocity[i] = sz0 * v0 + sz1 * v1;
        }
    }
}

void AdvectVelocity(int size, Real dt, VecArray<vec3r, MemType::GPU>& velocity, VecArray<vec3r, MemType::GPU>& tempVelocity, VecArray<bool, MemType::GPU>& rigidFlag){
    FILL_CALL_GPU_DEVICE_CODE(advectVelocity1, dt, velocity.m_data, tempVelocity.m_data, rigidFlag.m_data);
    cudaDeviceSynchronize();

    tempVelocity.swap(velocity);
}

__global__ void _g_diffuseDensity(int size, Real a, Real* __restrict__ density, Real* __restrict__ tempDensity, int parity, bool* __restrict__ rigidFlag){
    IF_IDX_VALID(size) {
        int3 gridIdx = _d_getGridIdx(i);
        // Check if the grid index is valid
        if(_d_isValid(gridIdx)&&(gridIdx.x+gridIdx.y+gridIdx.z)%2==parity && !rigidFlag[i]){
            tempDensity[i] = (density[i] + a * (tempDensity[_d_getIdx(make_int3(gridIdx.x+1, gridIdx.y, gridIdx.z))] + tempDensity[_d_getIdx(make_int3(gridIdx.x-1, gridIdx.y, gridIdx.z))] + tempDensity[_d_getIdx(make_int3(gridIdx.x, gridIdx.y+1, gridIdx.z))] + tempDensity[_d_getIdx(make_int3(gridIdx.x, gridIdx.y-1, gridIdx.z))] + tempDensity[_d_getIdx(make_int3(gridIdx.x, gridIdx.y, gridIdx.z+1))] + tempDensity[_d_getIdx(make_int3(gridIdx.x, gridIdx.y, gridIdx.z-1))])) / (1 + 6.0f * a);
        }
    }
}

void DiffuseDensity(int size, Real dt, VecArray<Real, MemType::GPU>& density, VecArray<Real, MemType::GPU>& tempDensity, VecArray<bool, MemType::GPU>& rigidFlag){
    
    Real coef = dt * hGridParams.kdiffusion * (hGridParams.gridSize.x-2) * (hGridParams.gridSize.y-2) * (hGridParams.gridSize.z-2);
    for(int i = 0; i < 20; i++){
        FILL_CALL_GPU_DEVICE_CODE(diffuseDensity, coef, density.m_data, tempDensity.m_data, 0, rigidFlag.m_data);
        cudaDeviceSynchronize();

        FILL_CALL_GPU_DEVICE_CODE(diffuseDensity, coef, density.m_data, tempDensity.m_data, 1, rigidFlag.m_data);
        cudaDeviceSynchronize();
    }

    tempDensity.swap(density);
}

__global__ void _g_advectDensity(int size, Real dt, Real* __restrict__ density, Real* __restrict__ tempDensity, vec3r* __restrict__ velocity, bool* __restrict__ rigidFlag){
        IF_IDX_VALID(size) {
        int3 gridIdx = _d_getGridIdx(i);
        // Check if the grid index is valid
        if(_d_isValid(gridIdx) && !rigidFlag[i]){
            Real xx = gridIdx.x - gridParams.gridSize.x * dt * velocity[i].x;
            Real yy = gridIdx.y - gridParams.gridSize.y * dt * velocity[i].y;
            Real zz = gridIdx.z - gridParams.gridSize.z * dt * velocity[i].z;
            if(xx < 0.5f) xx = 0.5f; if(xx > gridParams.gridSize.x - 1.5f) xx = gridParams.gridSize.x - 1.5f;
            if(yy < 0.5f) yy = 0.5f; if(yy > gridParams.gridSize.y - 1.5f) yy = gridParams.gridSize.y - 1.5f;
            if(zz < 0.5f) zz = 0.5f; if(zz > gridParams.gridSize.z - 1.5f) zz = gridParams.gridSize.z - 1.5f;
            int x0 = (int)xx, y0 = (int)yy, z0 = (int)zz;
            int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
            Real sx1 = xx - x0, sx0 = 1.0f - sx1;
            Real sy1 = yy - y0, sy0 = 1.0f - sy1;
            Real sz1 = zz - z0, sz0 = 1.0f - sz1;
            Real v0 = sx0 * (sy0  * density[_d_getIdx(make_int3(x0, y0, z0))] + sy1 * density[_d_getIdx(make_int3(x0, y1, z0))]) + sx1 * (sy0 * density[_d_getIdx(make_int3(x1, y0, z0))] + sy1 * density[_d_getIdx(make_int3(x1, y1, z0))]);
            Real v1 = sx0 * (sy0  * density[_d_getIdx(make_int3(x0, y0, z1))] + sy1 * density[_d_getIdx(make_int3(x0, y1, z1))]) + sx1 * (sy0 * density[_d_getIdx(make_int3(x1, y0, z1))] + sy1 * density[_d_getIdx(make_int3(x1, y1, z1))]);
            tempDensity[i] = sz0 * v0 + sz1 * v1;
        }
    }
}

void AdvectDensity(int size, Real dt, VecArray<Real, MemType::GPU>& density, VecArray<Real, MemType::GPU>& tempDensity, VecArray<vec3r, MemType::GPU>& velocity, VecArray<bool, MemType::GPU>& rigidFlag){
    FILL_CALL_GPU_DEVICE_CODE(advectDensity, dt, density.m_data, tempDensity.m_data, velocity.m_data, rigidFlag.m_data);
    cudaDeviceSynchronize();

    tempDensity.swap(density);
}

__global__ void _g_decreaseDensity(int size, Real disappearRate, Real* __restrict__ density){
    IF_IDX_VALID(size) {
        int3 gridIdx = _d_getGridIdx(i);
        // Check if the grid index is valid
        if(_d_isValid(gridIdx)){
            density[i] -= disappearRate;
            if(density[i] < 0.0f) density[i] = 0.0f;
        }
    }
}

void DecreaseDensity(int size, Real disappearRate, VecArray<Real, MemType::GPU>& density){
    FILL_CALL_GPU_DEVICE_CODE(decreaseDensity, disappearRate, density.m_data);
    cudaDeviceSynchronize();
}
PHYS_NAMESPACE_END
