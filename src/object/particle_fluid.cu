#include "object/particle_fluid.h"
#include "object/particle_fluid_util.h"
#include "curand_kernel.h"
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"
// #include "object/particle_system_util.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

PHYS_NAMESPACE_BEGIN

// __host__ __device__ vec3r _d_enforceBoundaryLocal(vec3r pos){
//     vec3r npos=pos;
//     npos.x=(max(min(npos.x,params.worldMax.x), params.worldMin.x)+pos.x)*0.5f;
//     npos.y=(max(min(npos.y,params.worldMax.y), params.worldMin.y)+pos.y)*0.5f;
//     npos.z=(max(min(npos.z,params.worldMax.z), params.worldMin.z)+pos.z)*0.5f;
//     return npos;
// }

// //// advect dt
// template<MemType MT>
// __host__ __device__ void _k_advect(int i, Real dt, vec3r* pos, vec3r* vel){
//     pos[i].x+=vel[i].x;
//     pos[i].y+=vel[i].y;
//     pos[i].z+=vel[i].z;
// }
// DECLARE_KERNEL_TEMP(advect, Real dt, vec3r* pos, vec3r* vel);

// __global__ void _g_advect(int size, Real dt, vec3r* pos, vec3r* vel){
//     IF_IDX_VALID(size) _k_advect<MemType::GPU>(i, dt, pos, vel);
// }

// template<MemType MT>
// void callAdvect(int size, Real dt, VecArray<vec3r,MT>& pos, VecArray<vec3r,MT>& vel){
//     FILL_CALL_DEVICE_CODE(advect, dt, pos.m_data, vel.m_data);
// }
// template void callAdvect<MemType::CPU>(int, Real, VecArray<vec3r,MemType::CPU>&, VecArray<vec3r,MemType::CPU>&);
// template void callAdvect<MemType::GPU>(int, Real, VecArray<vec3r,MemType::GPU>&, VecArray<vec3r,MemType::GPU>&);

// //// init particles in a grid
// template<MemType MT>
// __host__ __device__ void _k_stackParticle(int i, int x, int y, int z, Real gap, vec3r* pos){

//     int px=i%x;
//     int yz=i/x;
//     int py=yz%y;
//     int pz=yz/y;
//     vec3r p=make_vec3r(px*gap,py*gap,pz*gap);
//     pos[i].x=p.x;
//     pos[i].y=p.y;
//     pos[i].z=p.z;
// }
// DECLARE_KERNEL_TEMP(stackParticle, int x, int y, int z, Real gap, vec3r* pos);

// __global__ void _g_stackParticle(int size, int x, int y, int z, Real gap, vec3r* pos){
//     IF_IDX_VALID(size) _k_stackParticle<MemType::GPU>(i, x, y, z, gap, pos);
// }

// template<MemType MT>
// void callStackParticle(int size, int x, int y, int z, Real gap, VecArray<vec3r,MT>& pos){
//     FILL_CALL_DEVICE_CODE(stackParticle, x, y, z, gap, pos.m_data);
// }
// template void callStackParticle<MemType::CPU>(int, int, int, int, Real, VecArray<vec3r,MemType::CPU>&);
// template void callStackParticle<MemType::GPU>(int, int, int, int, Real, VecArray<vec3r,MemType::GPU>&);

// //// compute a virtual attract force
// template<MemType MT>
// __host__ __device__ void _k_computeAttractForce(int i, vec3r center, Real scale, vec3r* force, vec3r* pos){
//     // vec3r rel=pos[i]-center;
//     force[i]=(center-pos[i])*scale;
// }
// DECLARE_KERNEL_TEMP(computeAttractForce, vec3r center, Real scale, vec3r* force, vec3r* pos);

// __global__ void _g_computeAttractForce(int size, vec3r center, Real scale, vec3r* force, vec3r* pos){
//     IF_IDX_VALID(size) _k_computeAttractForce<MemType::GPU>(i, center, scale, force, pos);
// }

// template<MemType MT>
// void callComputeAttractForce(int size, vec3r center, Real scale, VecArray<vec3r,MT>& vf, VecArray<vec3r,MT>& vx){
//     FILL_CALL_DEVICE_CODE(computeAttractForce, center, scale, vf.m_data, vx.m_data);
// }
// template void callComputeAttractForce<MemType::CPU>(int, vec3r, Real, VecArray<vec3r,MemType::CPU>&, VecArray<vec3r,MemType::CPU>&);
// template void callComputeAttractForce<MemType::GPU>(int, vec3r, Real, VecArray<vec3r,MemType::GPU>&, VecArray<vec3r,MemType::GPU>&);

/**
 * Iterates over the neighboring cells of a target point in a grid.
 *
 * @param target_point The target point in the grid.
 */
#define NEIGHBOR_BEGIN(target_point)                                    \
    const int3 gridPos = _d_calcGridPos(target_point);                  \
    for (int z = -1; z <= 1; z++)                                       \
        for (int y = -1; y <= 1; y++)                                   \
            for (int x = -1; x <= 1; x++)                               \
            {                                                           \
                const int3 neighbourPos = gridPos + make_int3(x, y, z); \
                const uint gridHash = _d_calcGridHash(neighbourPos);    \
                const uint startIndex = cellStart[gridHash];            \
                const uint endIndex = cellEnd[gridHash];                \
                for (uint j = startIndex; j < endIndex; j++)            \
                {

#define NEIGHBOR_END \
    }                \
    }

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
/**
 * Reorders data and finds cell start indices.
 *
 * @param cellStart - output: cell start index
 * @param cellEnd - output: cell end index
 * @param sortedPositionPhase - output: sorted phase
 * @param sortedVel - output: sorted velocity
 * @param sortedMass - output: sorted invmass
 * @param sortedPhase - output: sorted phase
 * @param gridParticleHash - input: sorted grid hashes
 * @param O2SParticleIndex - input: O2S particle index
 * @param S2OParticleIndex - input: S2O particle index
 * @param oldPos - input: sorted position array
 * @param oldVel - input: sorted velocity array
 * @param oldMass - input: sorted W array
 * @param oldPhase - input: sorted Phase array
 * @param numParticles - number of particles
 */
__global__ void reorderDataAndFindCellStartD(uint *cellStart, uint *cellEnd, vec4r *sortedPositionPhase, vec4r *sortedVel, Real *sortedMass, uint *sortedPhase, uint *gridParticleHash, uint *O2SParticleIndex,uint *S2OParticleIndex,vec3r *oldPos,vec4r *oldVel,Real *oldMass,uint *oldPhase,uint numParticles)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();

    // Shared memory for storing hash values
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

        sortedVel[index] = oldVel[oidx];
        sortedMass[index] = oldMass[oidx];
        sortedPhase[index] = oldPhase[oidx];
        sortedPositionPhase[index] = make_vec4r(oldPos[oidx].x, oldPos[oidx].y, oldPos[oidx].z, (Real)oldPhase[oidx]);
    }
}

/**
 * Reorders the data and finds the start of each cell.
 *
 * @param size The size of the data.
 * @param cellStart The array to store the start indices of each cell.
 * @param cellEnd The array to store the end indices of each cell.
 * @param sortedPositionPhase The sorted position and phase data.
 * @param sortedVel The sorted velocity data.
 * @param sortedMass The sorted mass data.
 * @param sortedPhase The sorted phase data.
 * @param gridParticleHash The grid particle hash data.
 * @param O2SParticleIndex The index of particles after the order-to-species transformation.
 * @param S2OparticleIndex The index of particles after the species-to-order transformation.
 * @param oldPos The old position data.
 * @param oldVel The old velocity data.
 * @param oldMass The old mass data.
 * @param oldPhase The old phase data.
 * @param numCells The number of cells.
 * @param numParticles The number of particles.
 */
void reorderDataAndFindCellStart(uint size, uint *cellStart, uint *cellEnd, vec4r *sortedPositionPhase, vec4r *sortedVel, Real *sortedMass, uint *sortedPhase, uint *gridParticleHash, uint *O2SParticleIndex, uint *S2OparticleIndex, vec3r *oldPos, vec4r *oldVel, Real *oldMass, uint *oldPhase, uint numCells, uint numParticles)
{
    // Compute the number of CUDA threads and blocks
    uint numThreads, numBlocks;
    computeCudaThread(numParticles, PE_CUDA_BLOCKS, numBlocks, numThreads);

    // Set all cells to empty
    checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells * sizeof(uint)));

    // Calculate the shared memory size
    uint smemSize = sizeof(uint) * (numThreads + 1);

    // Invoke the kernel to reorder the data and find cell start indices
    reorderDataAndFindCellStartD<<<numBlocks, numThreads, smemSize>>>(cellStart, cellEnd, sortedPositionPhase, sortedVel, sortedMass, sortedPhase, gridParticleHash, O2SParticleIndex, S2OparticleIndex, oldPos, oldVel, oldMass, oldPhase, numParticles);
    
    // Check for kernel execution errors
    getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");
}

// Resolves penetration for particles in a given grid cell
// index: index of the particle
// tempPosition: array of temporary positions for particles
// sortedPositionPhase: array of sorted positions and phases of particles
// sortedMass: array of sorted masses of particles
// S2OParticleIndex: array of sorted to original particle indices
// cellStart: array of indices indicating the start of each grid cell
// cellEnd: array of indices indicating the end of each grid cell
// rigidParticleSign: array of signs indicating if a particle is rigid or not
template <MemType MT>
__host__ __device__ void _k_resolvePenetration(int index, vec3r *tempPosition, vec4r *sortedPositionPhase, Real *sortedMass, uint *S2OParticleIndex, uint *cellStart, uint *cellEnd, uint *rigidParticleSign)
{
    // C=|x_ij|-r>=0
    const vec4r xph_i = sortedPositionPhase[index];
    const vec3r x_i = make_vec3r(xph_i);
    const uint ph_i = (uint)xph_i.w;

    // Store the current position in the temporary position array
    tempPosition[index] = x_i;

    // Skip the resolution for non-sand and non-rigid particles
    if (ph_i != PhaseType::Sand && ph_i != PhaseType::Rigid)
        return;

    Real m_i = params.pmass[ph_i];
    uint oidx_i = S2OParticleIndex[index];
    vec3r deltax = make_vec3r(0.0f);
    int neighbourCount = 0;

    int3 gridPos = _d_calcGridPos(x_i);

    // Iterate over neighbouring grid cells
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
                    // Get the position and phase of the j-th particle
                    const vec4r xph_j = sortedPositionPhase[j];
                    const vec3r x_j = make_vec3r(xph_j);
                    const uint ph_j = (uint)xph_j.w;
                    uint oidx_j = S2OParticleIndex[j];

                    // Skip if both particles are rigid and have the same sign
                    if (ph_i == PhaseType::Rigid && ph_j == PhaseType::Rigid && rigidParticleSign[oidx_i] == rigidParticleSign[oidx_j])
                        continue;

                    // Calculate the relative position and distance between particles i and j
                    vec3r rel = x_i - x_j;
                    Real dist2 = dot(rel, rel);
                    Real dist = sqrt(dist2);

                    // Skip if the particles are the same or the distance is greater than the threshold or there are already too many neighbors
                    if (j == index || dist2 >= params.peneDist2 || neighbourCount > params.maxNeighbours)
                        continue;

                    neighbourCount++;
                    Real m_j = params.pmass[ph_j];

                    // Calculate the displacement for particle i
                    deltax += -m_j * (dist - params.peneDist) / (dist * (m_i + m_j)) * rel;
                }
            }
    // Update the position of particle i based on the average displacement from its neighbors
    if (neighbourCount > 0)
        tempPosition[index] = x_i + deltax * (1.f / neighbourCount);
}

// Declare the kernel function resolvePenetration with its input parameters
DECLARE_KERNEL_TEMP(resolvePenetration, vec3r *tempPosition, vec4r *sortedPositionPhase, Real *sortedMass, uint *S2OParticleIndex, uint *cellStart, uint *cellEnd, uint *rigidParticleSign);

/**
 * Kernel function to resolve penetration
 *
 * @param size                The size of the input arrays
 * @param tempPosition        The temporary position array
 * @param sortedPositionPhase The sorted position phase array
 * @param sortedMass          The sorted mass array
 * @param S2OParticleIndex    The S2O particle index array
 * @param cellStart           The cell start array
 * @param cellEnd             The cell end array
 * @param rigidParticleSign   The rigid particle sign array
 */
__global__ void _g_resolvePenetration(int size, vec3r *tempPosition, vec4r *sortedPositionPhase, Real *sortedMass, uint *S2OParticleIndex, uint *cellStart, uint *cellEnd, uint *rigidParticleSign)
{
    // Check if the current thread index is valid
    IF_IDX_VALID(size)
    
    // Call the kernel function _k_resolvePenetration with GPU memory type
    _k_resolvePenetration<MemType::GPU>(i, tempPosition, sortedPositionPhase, sortedMass, S2OParticleIndex, cellStart, cellEnd, rigidParticleSign);
}

// Resolves penetration between particles using the given parameters
// 
// Parameters:
// - size: The size of the particle arrays
// - tempPosition: The temporary position array
// - sortedPositionPhase: The sorted position-phase array
// - sortedMass: The sorted mass array
// - S2OParticleIndex: The index array for S2O particles
// - cellStart: The array containing the starting indices of each cell
// - cellEnd: The array containing the ending indices of each cell
// - rigidParticleSign: The array containing the signs of rigid particles
template <MemType MT>
void callResolvePenetration(int size, VecArray<vec3r, MT> &tempPosition, VecArray<vec4r, MT> &sortedPositionPhase, VecArray<Real, MT> &sortedMass, VecArray<uint, MT> &S2OParticleIndex, VecArray<uint, MT> &cellStart, VecArray<uint, MT> &cellEnd, VecArray<uint, MT> &rigidParticleSign)
{
    // Call the resolvePenetration device code with the given parameters
    FILL_CALL_DEVICE_CODE(resolvePenetration, tempPosition.m_data, sortedPositionPhase.m_data, sortedMass.m_data, S2OParticleIndex.m_data, cellStart.m_data, cellEnd.m_data, rigidParticleSign.m_data);
}
// Template specialization for CPU memory type
template void callResolvePenetration<MemType::CPU>(int, VecArray<vec3r, CPU> &, VecArray<vec4r, CPU> &, VecArray<Real, CPU> &, VecArray<uint, CPU> &, VecArray<uint, CPU> &, VecArray<uint, CPU> &, VecArray<uint, CPU> &);
// Template specialization for GPU memory type
template void callResolvePenetration<MemType::GPU>(int, VecArray<vec3r, GPU> &, VecArray<vec4r, GPU> &, VecArray<Real, GPU> &, VecArray<uint, GPU> &, VecArray<uint, GPU> &, VecArray<uint, GPU> &, VecArray<uint, GPU> &);

/**
 * Resolves friction for a given particle index.
 *
 * @tparam MT The memory type.
 * @param index The index of the particle.
 * @param tempStarPositionPhase The temporary star position phase array.
 * @param tempPosition The temporary position array.
 * @param oldPosition The old position array.
 * @param sortedPositionPhase The sorted position phase array.
 * @param sortedMass The sorted mass array.
 * @param S2OParticleIndex The S2O particle index array.
 * @param cellStart The cell start array.
 * @param cellEnd The cell end array.
 * @param rigidParticleSign The rigid particle sign array.
 */
template <MemType MT>
__host__ __device__ void _k_resolveFriction(int index, vec4r *tempStarPositionPhase, vec3r *tempPosition, vec3r *oldPosition, vec4r *sortedPositionPhase, Real *sortedMass, uint *S2OParticleIndex, uint *cellStart, uint *cellEnd, uint *rigidParticleSign)
{

    // Get the sorted position and phase values for the current index
    const vec4r xph_i = sortedPositionPhase[index];

    // Extract the position vector from the sorted position and phase values
    const vec3r x_i = make_vec3r(xph_i);

    // Extract the phase value as an unsigned integer
    const uint ph_i = (uint)xph_i.w;

    // Store the current position and phase values in a temporary array
    vec3r nx_i = tempPosition[index];
    tempStarPositionPhase[index] = make_vec4r(nx_i, ph_i);

    // Check if the phase value is not Sand or Rigid
    if (ph_i != PhaseType::Sand && ph_i != PhaseType::Rigid)
        return;

    // Get the mass value for the current phase
    Real m_i = params.pmass[ph_i];
    // Get the old position index for the current particle index
    uint oidx_i = S2OParticleIndex[index];
    // Get the old position vector for the old position index
    vec3r ox_i = oldPosition[oidx_i];

    // Calculate the difference vector between the new and old positions
    vec3r dx_i = nx_i - ox_i;
    // Initialize the deltax vector with all zeros
    vec3r deltax = make_vec3r(0.0f);

    //// neighbor friction
    int neighbourCount = 0;
    int3 gridPos = _d_calcGridPos(x_i);

    // Iterate over neighboring cells
    for (int z = -1; z <= 1; z++)
        for (int y = -1; y <= 1; y++)
            for (int x = -1; x <= 1; x++)
            {
                // Calculate the position of the neighboring cell
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                // Calculate the hash value for the neighboring cell
                uint gridHash = _d_calcGridHash(neighbourPos);
                // Get the start and end indices for particles in the neighboring cell
                uint startIndex = cellStart[gridHash];
                uint endIndex = cellEnd[gridHash];

                // Iterate over particles in the current cell
                for (uint j = startIndex; j < endIndex; j++)
                {
                    // Get the index of the j-th particle
                    uint oidx_j = S2OParticleIndex[j];
                    // Get the position and phase of the j-th particle
                    const vec4r xph_j = sortedPositionPhase[j];
                    const vec3r x_j = make_vec3r(xph_j);
                    const uint ph_j = (uint)xph_j.w;

                    // Skip rigid particles that have the same sign
                    if (ph_i == PhaseType::Rigid && ph_j == PhaseType::Rigid && rigidParticleSign[oidx_i] == rigidParticleSign[oidx_j])
                        continue;

                    // Calculate the relative position between the i-th and j-th particles
                    vec3r rel = x_i - x_j;
                    Real dist2 = dot(rel, rel);

                    // Skip if the particle is the same or the distance is greater than the penetration distance squared
                    if (j == index || dist2 >= params.peneDist2 || neighbourCount > params.maxNeighbours)
                        continue;

                    // Calculate the Euclidean distance between the particles
                    Real dist = sqrt(dist2);
                    neighbourCount++;

                    // Compute the normalized vector pointing from particle i to particle j
                    vec3r n = rel / dist;

                    // Get the position and mass of particle j
                    vec3r nx_j = tempPosition[j];
                    Real m_j = params.pmass[ph_j];

                    // Compute the relative velocity tangent to the contact normal
                    vec3r ddx = dx_i - nx_j + oldPosition[oidx_j];
                    vec3r dx_tan = ddx - n * dot(ddx, n);
                    Real l_dx_tan = length(dx_tan);

                    // Compute the displacement of particle i due to the collision with particle j
                    Real d = params.peneDist - dist;
                    vec3r deltax_i = m_j / (m_i + m_j) * dx_tan;

                    // Apply dynamic friction if the relative velocity tangent is greater than the static friction threshold
                    if (l_dx_tan > d * params.staticFriction)
                        deltax_i *= params.dynamicFriction * d / l_dx_tan;

                    deltax -= deltax_i;
                }
            }
    // if(neighbourCount>0) tempStarPositionPhase[index]=_d_enforceBoundaryLocal(nx_i+deltax/neighbourCount);
    if (neighbourCount > 0)
        nx_i += deltax * (1.f / neighbourCount);
    // tempStarPositionPhase[index]=nx_i;
    // x_i=nx_i;

    //// boundary friction
    {
        // Initialize variables
        vec3r n = make_vec3r(0.f); // Normal vector
        Real d = 0.f; // Distance
        bool isCollide = false; // Flag for collision detection

        // Check if the x-coordinate is greater than the maximum allowed value
        if (nx_i.x > params.worldMax.x)
        {
            d = nx_i.x - params.worldMax.x;
            nx_i.x = params.worldMax.x;
            n = make_vec3r(-1.f, 0.f, 0.f); // Set the normal vector to point in the negative x-direction
            isCollide = true; // Collision detected
        }

        // Check if the x-coordinate is less than the minimum allowed value
        if (nx_i.x < params.worldMin.x)
        {
            d = params.worldMin.x - nx_i.x;
            nx_i.x = params.worldMin.x;
            n = make_vec3r(1.f, 0.f, 0.f); // Set the normal vector to point in the positive x-direction
            isCollide = true; // Collision detected
        }

        // Check if the y-coordinate is greater than the maximum allowed value
        if (nx_i.y > params.worldMax.y)
        {
            d = nx_i.y - params.worldMax.y;
            nx_i.y = params.worldMax.y;
            n = make_vec3r(0.f, -1.f, 0.f); // Set the normal vector to point in the negative y-direction
            isCollide = true; // Collision detected
        }

        // Check if the y-coordinate is less than the minimum allowed value
        if (nx_i.y < params.worldMin.y)
        {
            d = params.worldMin.y - nx_i.y;
            nx_i.y = params.worldMin.y;
            n = make_vec3r(0.f, 1.f, 0.f); // Set the normal vector to point in the positive y-direction
            isCollide = true; // Collision detected
        }

        // Check if the z-coordinate is greater than the maximum allowed value
        if (nx_i.z > params.worldMax.z)
        {
            d = nx_i.z - params.worldMax.z;
            nx_i.z = params.worldMax.z;
            n = make_vec3r(0.f, 0.f, -1.f); // Set the normal vector to point in the negative z-direction
            isCollide = true; // Collision detected
        }

        // Check if the z-coordinate is less than the minimum allowed value
        if (nx_i.z < params.worldMin.z)
        {
            d = params.worldMin.z - nx_i.z;
            nx_i.z = params.worldMin.z;
            n = make_vec3r(0.f, 0.f, 1.f); // Set the normal vector to point in the positive z-direction
            isCollide = true; // Collision detected
        }

        if (!isCollide)
        {
            tempStarPositionPhase[index] = make_vec4r(nx_i, ph_i);
            return;
        }

        // Calculate the difference vector between nx_i and ox_i
        vec3r dx = nx_i - ox_i;
        // Calculate the tangential component of dx
        vec3r dx_tan = dx - n * dot(dx, n);
        // Calculate the length of dx_tan
        Real l_dx_tan = length(dx_tan);
        // Store dx_tan in deltax for further calculations
        vec3r deltax = dx_tan;

        // Check if the length of dx_tan is greater than d * params.staticFriction
        if (l_dx_tan > d * params.staticFriction)
        {
            // Scale deltax by params.dynamicFriction * d / l_dx_tan
            deltax *= params.dynamicFriction * d / l_dx_tan;
        }

        // Update tempStarPositionPhase with nx_i - deltax and ph_i
        tempStarPositionPhase[index] = make_vec4r(nx_i - deltax, ph_i);
    }
}
// Declare the kernel function resolveFriction with its input arguments
// tempStarPositionPhase: pointer to vec4r
// tempPosition: pointer to vec3r
// oldPosition: pointer to vec3r
// sortedPositionPhase: pointer to vec4r
// sortedMass: pointer to Real
// S2OParticleIndex: pointer to uint
// cellStart: pointer to uint
// cellEnd: pointer to uint
// rigidParticleSign: pointer to uint
DECLARE_KERNEL_TEMP(resolveFriction, vec4r *tempStarPositionPhase, vec3r *tempPosition, vec3r *oldPosition, vec4r *sortedPositionPhase, Real *sortedMass, uint *S2OParticleIndex, uint *cellStart, uint *cellEnd, uint *rigidParticleSign);

// Define the global kernel function _g_resolveFriction
__global__ void _g_resolveFriction(int size, vec4r *tempStarPositionPhase, vec3r *tempPosition, vec3r *oldPosition, vec4r *sortedPositionPhase, Real *sortedMass, uint *S2OParticleIndex, uint *cellStart, uint *cellEnd, uint *rigidParticleSign)
{
    // Check if the thread index is valid
    IF_IDX_VALID(size)
    // Call the kernel function _k_resolveFriction with the GPU memory type
    _k_resolveFriction<MemType::GPU>(i, tempStarPositionPhase, tempPosition, oldPosition, sortedPositionPhase, sortedMass, S2OParticleIndex, cellStart, cellEnd, rigidParticleSign);
}

// Define the template function callResolveFriction with the memory type template parameter MT
template <MemType MT>
void callResolveFriction(int size, VecArray<vec4r, MT> &tempStarPositionPhase, VecArray<vec3r, MT> &tempPosition, VecArray<vec3r, MT> &oldPosition, VecArray<vec4r, MT> &sortedPositionPhase, VecArray<Real, MT> &sortedMass, VecArray<uint, MT> &S2OParticleIndex, VecArray<uint, MT> &cellStart, VecArray<uint, MT> &cellEnd, VecArray<uint, MT> &rigidParticleSign)
{
    // Fill the call device code for resolveFriction with the data from the input arrays
    FILL_CALL_DEVICE_CODE(resolveFriction, tempStarPositionPhase.m_data, tempPosition.m_data, oldPosition.m_data, sortedPositionPhase.m_data, sortedMass.m_data, S2OParticleIndex.m_data, cellStart.m_data, cellEnd.m_data, rigidParticleSign.m_data);
}
// Explicitly instantiate the callResolveFriction function for CPU memory type
template void callResolveFriction<MemType::CPU>(int, VecArray<vec4r, CPU> &, VecArray<vec3r, CPU> &, VecArray<vec3r, CPU> &, VecArray<vec4r, CPU> &, VecArray<Real, CPU> &, VecArray<uint, CPU> &, VecArray<uint, CPU> &, VecArray<uint, CPU> &, VecArray<uint, CPU> &);
// Explicitly instantiate the callResolveFriction function for GPU memory type
template void callResolveFriction<MemType::GPU>(int, VecArray<vec4r, GPU> &, VecArray<vec3r, GPU> &, VecArray<vec3r, GPU> &, VecArray<vec4r, GPU> &, VecArray<Real, GPU> &, VecArray<uint, GPU> &, VecArray<uint, GPU> &, VecArray<uint, GPU> &, VecArray<uint, GPU> &);

//// update lambda
/**
 * Updates the lambda value for a particle at the given index.
 *
 * @tparam MT The memory type.
 * @param index The index of the particle.
 * @param lambda The array of lambda values.
 * @param invDensity The array of inverse densities.
 * @param sortedPositionPhase The array of sorted position and phase values.
 * @param sortedMass The array of sorted mass values.
 * @param S2OParticleIndex The array of particle indices.
 * @param cellStart The array of cell start indices.
 * @param cellEnd The array of cell end indices.
 */
template <MemType MT>
__device__ void _k_updateLambda(int index, Real *__restrict__ lambda, Real *__restrict__ invDensity, const vec4r *__restrict__ sortedPositionPhase, const Real *__restrict__ sortedMass, const uint *__restrict__ S2OParticleIndex, const uint *__restrict__ cellStart, const uint *__restrict__ cellEnd)
{
    // read particle data from sorted arrays
    const vec4r xph_i = sortedPositionPhase[index]; // Get the position and phase information of the particle at the given index
    const uint ph_i = (uint)xph_i.w; // Extract the phase value from the position-phase vector
    const vec3r x_i = make_vec3r(xph_i); // Extract the position vector from the position-phase vector
    Real inv_rho0_i = params.invRho0[(uint)ph_i]; // Get the inverse of the reference density for the given phase

    Real rho_i = params.zeroPoly6 * params.pmass[ph_i]; // Calculate the density of the particle using the poly6 kernel and the particle mass
    Real sumGradC2 = params.lambdaRelaxation; // Initialize the sum of gradient squared with the relaxation factor
    const Real h = params.h; // Get the smoothing length
    const Real h2 = params.h2; // Get the square of the smoothing length
    const Real poly6Coef = params.poly6Coef; // Get the coefficient for the poly6 kernel
    const Real spikyGradCoef = params.spikyGradCoef; // Get the coefficient for the spiky gradient kernel

    vec3r gradC_i = make_vec3r(0.f); // Initialize the gradient of the concentration with zero
    Real curLambda = 0.f; // Initialize the current lambda value with zero

    // Calculate the grid position of a given point
    const int3 gridPos = _d_calcGridPos(x_i);
    // Iterate over the 3D neighborhood of the point
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                // Calculate the position of the neighboring cell
                const int3 neighbourPos = gridPos + make_int3(x, y, z);
                // Calculate the hash value of the neighboring cell
                const uint gridHash = _d_calcGridHash(neighbourPos);
                // Find the start and end indices of particles in the cell
                const uint startIndex = cellStart[gridHash];
                const uint endIndex = cellEnd[gridHash];
    
                // Iterate over the particles in the cell
                for (uint j = startIndex; j < endIndex; j++) {
                    // Get the position and phase of the neighboring particle
                    const vec4r xph_j = sortedPositionPhase[j];
                    const uint ph_j = (uint)xph_j.w;
                    const vec3r x_j = make_vec3r(xph_j);
    
                    // Calculate the relative distance between the particles
                    const vec3r rel = x_i - x_j;
                    const Real dist2 = dot(rel, rel);
    
                    // Check conditions for interaction between particles
                    if (dist2 > h2 || (!_d_isLiquid(ph_i) && !_d_isLiquid(ph_j)) || j == index)
                        continue;
    
                    // Calculate the distance and mass of the neighboring particle
                    const Real dist = __fsqrt_rn(dist2);
                    const Real m_j = params.pmass[ph_j];
    
                    // Calculate the spiky kernel and gradient of the neighboring particle
                    const Real tmp_spiky = h - dist;
                    const vec3r wSpiky = __fdividef(spikyGradCoef, dist + 1e-8f) * tmp_spiky * tmp_spiky * rel;
                    Real tmp_poly6 = h2 - dist * dist;
                    const vec3r gradC_j = -m_j * inv_rho0_i * wSpiky;
    
                    // Calculate the density of the particle
                    rho_i += (poly6Coef * tmp_poly6) * (tmp_poly6 * tmp_poly6) * m_j;
    
                    // Update the sum of squared gradients and the gradient of the current particle
                    sumGradC2 += dot(gradC_j, gradC_j);
                    gradC_i -= gradC_j;
                }
            }
        }
    }
    
    // Calculate the constraint C
    const Real C = max(0.f, (rho_i * inv_rho0_i) - 1.f);
    // Calculate the lambda value for the current particle
    lambda[index] = -__fdividef(C, (sumGradC2 + dot(gradC_i, gradC_i)));
    // Calculate the inverse density of the current particle
    invDensity[index] = __fdividef(1.0f, rho_i);
}
// Declare GPU kernel template
DECLARE_GPU_KERNEL_TEMP(updateLambda, Real *__restrict__ lambda, Real *__restrict__ invDensity, const vec4r *__restrict__ sortedPositionPhase, const Real *__restrict__ sortedMass, const uint *__restrict__ S2OParticleIndex, const uint *__restrict__ cellStart, const uint *__restrict__ cellEnd);

// GPU kernel function
__global__ void _g_updateLambda(int size, Real *__restrict__ lambda, Real *__restrict__ invDensity, const vec4r *__restrict__ sortedPositionPhase, const Real *__restrict__ sortedMass, const uint *__restrict__ S2OParticleIndex, const uint *__restrict__ cellStart, const uint *__restrict__ cellEnd)
{
    // Check if the thread index is valid
    IF_IDX_VALID(size)

    // Call the GPU kernel template function
    _k_updateLambda<MemType::GPU>(i, lambda, invDensity, sortedPositionPhase, sortedMass, S2OParticleIndex, cellStart, cellEnd);
}

// Template function to call the updateLambda GPU kernel
template <MemType MT>
void callUpdateLambda(int size, VecArray<Real, MT> &lambda, VecArray<Real, MT> &invDensity, VecArray<vec4r, MT> &sortedPositionPhase, VecArray<Real, MT> &sortedMass, VecArray<uint, MT> &S2OParticleIndex, VecArray<uint, MT> &cellStart, VecArray<uint, MT> &cellEnd)
{
    // Fill the call to the GPU device code for the updateLambda kernel
    FILL_CALL_GPU_DEVICE_CODE(updateLambda, lambda.m_data, invDensity.m_data, sortedPositionPhase.m_data, sortedMass.m_data, S2OParticleIndex.m_data, cellStart.m_data, cellEnd.m_data);
}

// Template instantiations for the callUpdateLambda function
template void callUpdateLambda<MemType::CPU>(int, VecArray<Real, MemType::CPU> &, VecArray<Real, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &, VecArray<Real, MemType::CPU> &, VecArray<uint, MemType::CPU> &, VecArray<uint, MemType::CPU> &, VecArray<uint, MemType::CPU> &);
template void callUpdateLambda<MemType::GPU>(int, VecArray<Real, MemType::GPU> &, VecArray<Real, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<Real, MemType::GPU> &, VecArray<uint, MemType::GPU> &, VecArray<uint, MemType::GPU> &, VecArray<uint, MemType::GPU> &);

//// solve fluid
template <MemType MT>
__device__ void _k_solveFluid(int index, vec4r *__restrict__ normal, vec4r *__restrict__ newPositionPhase, const Real *__restrict__ lambda, const Real *__restrict__ invDensity, const vec4r *__restrict__ sortedPositionPhase, const Real *__restrict__ sortedMass, const uint *__restrict__ S2OParticleIndex, const uint *__restrict__ cellStart, const uint *__restrict__ cellEnd)
{

    const vec4r xph_i = sortedPositionPhase[index];
    // vec3r x_i = sortedPos[index];
    const vec3r x_i = make_vec3r(xph_i);
    const uint ph_i = (uint)xph_i.w;
    newPositionPhase[index] = xph_i;

    // uint ph_i = sortedPhase[index];
    if (!_d_isLiquid(ph_i))
        return;

    const Real lambda_i = lambda[index];
    const Real inv_rho0_i = params.invRho0[(uint)ph_i];
    const Real h = params.h;
    const Real h2 = params.h2;
    const Real poly6Coef = params.poly6Coef;
    const Real spikyGradCoef = params.spikyGradCoef;
    vec3r deltap = make_vec3r(0.0f);

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
                    const vec4r xph_j = sortedPositionPhase[j];
                    // vec3r x_j = sortedPos[index];
                    const vec3r x_j = make_vec3r(xph_j);
                    const uint ph_j = (uint)xph_j.w;
                    // vec3r rel = x_i - sortedPos[j];
                    const vec3r rel = x_i - x_j;
                    const Real dist2 = dot(rel, rel);
                    if (dist2 > h2 || j == index)
                        continue;

                    // uint ph_j = sortedPhase[j];
                    // Real dist = _d_fastsqrt(dist2);
                    const Real dist = __fsqrt_rn(dist2);
                    // vec3r wSpiky=_d_spikyGrad(rel, dist, h, spikyGradCoef);
                    const Real tmp = h - dist;
                    const Real m_j = params.pmass[ph_j];
                    const Real lambda_j = lambda[j];
                    // const vec3r wSpiky = (__fdividef(spikyGradCoef, dist+1e-8f) * tmp * tmp) * rel;
                    // deltap += m_j * (lambda_i+lambda_j) * wSpiky;

                    const Real tmp2 = tmp * tmp;
                    // const Real lambda_sum=lambda_i+lambda_j;
                    const Real lambda_sum = __fadd_rn(lambda_i, lambda_j);
                    const Real spiky_div = __fdividef(spikyGradCoef, __fadd_rn(dist, 1e-8f));
                    const Real mlambda = m_j * lambda_sum;
                    vec3r wSpiky = spiky_div * tmp2 * rel;
                    deltap += mlambda * wSpiky;
                    // const vec3r wSpiky = (__fdividef(spikyGradCoef, dist+1e-8f) * tmp * tmp) * rel;
                    // deltap += m_j * (lambda_i+lambda_j) * wSpiky;
                }
            }
    newPositionPhase[index] = make_vec4r(_d_enforceBoundaryLocal(x_i + deltap * inv_rho0_i), xph_i.w);
}

DECLARE_GPU_KERNEL_TEMP(solveFluid, vec4r *__restrict__ normal, vec4r *__restrict__ newPositionPhase, const Real *__restrict__ lambda, const Real *__restrict__ invDensity, const vec4r *__restrict__ sortedPositionPhase, const Real *__restrict__ sortedMass, const uint *__restrict__ S2OParticleIndex, const uint *__restrict__ cellStart, const uint *__restrict__ cellEnd);

__global__ void _g_solveFluid(int size, vec4r *__restrict__ normal, vec4r *__restrict__ newPositionPhase, const Real *__restrict__ lambda, const Real *__restrict__ invDensity, const vec4r *__restrict__ sortedPositionPhase, const Real *__restrict__ sortedMass, const uint *__restrict__ S2OParticleIndex, const uint *__restrict__ cellStart, const uint *__restrict__ cellEnd)
{
    IF_IDX_VALID(size)
    _k_solveFluid<MemType::GPU>(i, normal, newPositionPhase, lambda, invDensity, sortedPositionPhase, sortedMass, S2OParticleIndex, cellStart, cellEnd);
}

template <MemType MT>
void callSolveFluid(int size, VecArray<vec4r, MT> &normal, VecArray<vec4r, MT> &newPositionPhase, VecArray<Real, MT> &lambda, VecArray<Real, MT> &invDensity, VecArray<vec4r, MT> &sortedPositionPhase, VecArray<Real, MT> &sortedMass, VecArray<uint, MT> &S2OParticleIndex, VecArray<uint, MT> &cellStart, VecArray<uint, MT> &cellEnd)
{
    FILL_CALL_GPU_DEVICE_CODE(solveFluid, normal.m_data, newPositionPhase.m_data, lambda.m_data, invDensity.m_data, sortedPositionPhase.m_data, sortedMass.m_data, S2OParticleIndex.m_data, cellStart.m_data, cellEnd.m_data);
}
template void callSolveFluid<MemType::CPU>(int, VecArray<vec4r, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &, VecArray<Real, MemType::CPU> &, VecArray<Real, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &, VecArray<Real, MemType::CPU> &, VecArray<uint, MemType::CPU> &, VecArray<uint, MemType::CPU> &, VecArray<uint, MemType::CPU> &);
template void callSolveFluid<MemType::GPU>(int, VecArray<vec4r, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<Real, MemType::GPU> &, VecArray<Real, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<Real, MemType::GPU> &, VecArray<uint, MemType::GPU> &, VecArray<uint, MemType::GPU> &, VecArray<uint, MemType::GPU> &);

template <MemType MT>
__device__ void _k_solveFluidAndViscosity(int index, vec4r *__restrict__ normal, vec4r *__restrict__ dv, vec4r *__restrict__ newPositionPhase, const Real *__restrict__ lambda, const Real *__restrict__ invDensity, const vec4r *__restrict__ sortedPositionPhase, const vec4r *__restrict__ sortedVel, const Real *__restrict__ sortedMass, const uint *__restrict__ S2OParticleIndex, const uint *__restrict__ cellStart, const uint *__restrict__ cellEnd)
{
    const vec4r xph_i = sortedPositionPhase[index];
    // vec3r x_i = sortedPos[index];
    const vec3r x_i = make_vec3r(xph_i);
    const uint ph_i = (uint)xph_i.w;

    newPositionPhase[index] = xph_i;
    dv[index] = make_vec4r(0.0f);

    if (!_d_isLiquid(ph_i))
        return;

    const vec3r v_i = make_vec3r(sortedVel[index]);
    const Real lambda_i = lambda[index];
    const Real inv_rho0_i = params.invRho0[(uint)ph_i];
    const Real h = params.h;
    const Real h2 = params.h2;
    const Real poly6Coef = params.poly6Coef;
    const Real spikyGradCoef = params.spikyGradCoef;

    vec3r deltap = make_vec3r(0.0f);
    vec3r deltav = make_vec3r(0.0f);

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
                    const vec4r xph_j = sortedPositionPhase[j];
                    // vec3r x_j = sortedPos[index];
                    const vec3r x_j = make_vec3r(xph_j);
                    const uint ph_j = (uint)xph_j.w;
                    // uint ph_j = sortedPhase[j];
                    const vec3r rel = x_i - x_j;
                    const Real dist2 = dot(rel, rel);
                    if (dist2 > h2 || j == index)
                        continue;

                    const Real dist = __fsqrt_rn(dist2);
                    // Real dist = _d_fastsqrt(dist2);
                    const Real m_j = params.pmass[ph_j];
                    // const Real lambda_j=lambda[j];
                    const Real lambda_sum = lambda_i + lambda[j];
                    const Real inv_rho_j = invDensity[j];
                    // const vec3r v_j=make_vec3r(sortedVel[j]);
                    const Real tmp_spiky = h - dist;
                    // const Real tmp_poly6 = h2-dist*dist;
                    const vec3r wSpiky = (__fdividef(spikyGradCoef, dist + 1e-8f) * tmp_spiky * tmp_spiky) * rel;
                    // const Real wPoly6=(poly6Coef*tmp_poly6)*(tmp_poly6*tmp_poly6);

                    // deltap += m_j*(lambda_i + lambda[j] + corr) * wSpiky * params.invRho0[(uint)ph_i];
                    deltap += m_j * (lambda_i + lambda[j]) * wSpiky;
                    deltav -= (m_j * inv_rho_j * _d_poly6(dist, h2, poly6Coef)) * (v_i - make_vec3r(sortedVel[j]));
                }
            }
    const vec3r newPosition = _d_enforceBoundaryLocal(x_i + deltap * inv_rho0_i);
    newPositionPhase[index] = make_vec4r(newPosition, xph_i.w);
    dv[index] = make_vec4r(deltav * params.kviscosity);
}

DECLARE_GPU_KERNEL_TEMP(solveFluidAndViscosity, vec4r *__restrict__ normal, vec4r *__restrict__ dv, vec4r *__restrict__ newPositionPhase, const Real *__restrict__ lambda, const Real *__restrict__ invDensity, const vec4r *__restrict__ sortedPositionPhase, const vec4r *__restrict__ sortedVel, const Real *__restrict__ sortedMass, const uint *__restrict__ S2OParticleIndex, const uint *__restrict__ cellStart, const uint *__restrict__ cellEnd);

__global__ void _g_solveFluidAndViscosity(int size, vec4r *__restrict__ normal, vec4r *__restrict__ dv, vec4r *__restrict__ newPositionPhase, const Real *__restrict__ lambda, const Real *__restrict__ invDensity, const vec4r *__restrict__ sortedPositionPhase, const vec4r *__restrict__ sortedVel, const Real *__restrict__ sortedMass, const uint *__restrict__ S2OParticleIndex, const uint *__restrict__ cellStart, const uint *__restrict__ cellEnd)
{
    IF_IDX_VALID(size)
    _k_solveFluidAndViscosity<MemType::GPU>(i, normal, dv, newPositionPhase, lambda, invDensity, sortedPositionPhase, sortedVel, sortedMass, S2OParticleIndex, cellStart, cellEnd);
}

template <MemType MT>
void callSolveFluidAndViscosity(int size, VecArray<vec4r, MT> &normal, VecArray<vec4r, MT> &dv, VecArray<vec4r, MT> &newPositionPhase, VecArray<Real, MT> &lambda, VecArray<Real, MT> &invDensity, VecArray<vec4r, MT> &sortedPositionPhase, VecArray<vec4r, MT> &sortedVel, VecArray<Real, MT> &sortedMass, VecArray<uint, MT> &S2OParticleIndex, VecArray<uint, MT> &cellStart, VecArray<uint, MT> &cellEnd)
{
    FILL_CALL_GPU_DEVICE_CODE(solveFluidAndViscosity, normal.m_data, dv.m_data, newPositionPhase.m_data, lambda.m_data, invDensity.m_data, sortedPositionPhase.m_data, sortedVel.m_data, sortedMass.m_data, S2OParticleIndex.m_data, cellStart.m_data, cellEnd.m_data);
}
template void callSolveFluidAndViscosity<MemType::CPU>(int, VecArray<vec4r, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &, VecArray<Real, MemType::CPU> &, VecArray<Real, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &, VecArray<Real, MemType::CPU> &, VecArray<uint, MemType::CPU> &, VecArray<uint, MemType::CPU> &, VecArray<uint, MemType::CPU> &);
template void callSolveFluidAndViscosity<MemType::GPU>(int, VecArray<vec4r, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<Real, MemType::GPU> &, VecArray<Real, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<Real, MemType::GPU> &, VecArray<uint, MemType::GPU> &, VecArray<uint, MemType::GPU> &, VecArray<uint, MemType::GPU> &);

//// compute a virtual attract force
template <MemType MT>
__host__ __device__ void _k_updateSurfaceTension(int index, vec4r *force, vec4r *normal, Real *invDensity, vec4r *sortedPositionPhase, Real *sortedMass, uint *S2OParticleIndex, uint *cellStart, uint *cellEnd)
{

    // uint ph_i = sortedPhase[index];
    const vec4r xph_i = sortedPositionPhase[index];
    const vec3r x_i = make_vec3r(xph_i);
    const uint ph_i = (uint)xph_i.w;

    if (!_d_isLiquid(ph_i))
        return;

    // vec3r x_i = sortedPos[index];
    uint inv_rho_i = invDensity[index];
    // vec3r n_i = normal[index];
    vec3r cohesion = make_vec3r(0.0f);
    vec3r curvature = make_vec3r(0.0f);

    Real h = params.h;
    Real h2 = params.h2;
    Real spikyGradCoef = params.spikyGradCoef;

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
                    const vec4r xph_j = sortedPositionPhase[j];
                    const vec3r x_j = make_vec3r(xph_j);
                    const uint ph_j = (uint)xph_j.w;
                    // uint ph_j = sortedPhase[j];
                    // vec3r x_j = sortedPos[j];
                    vec3r rel = x_i - x_j;
                    // vec3r n_j = normal[j];
                    Real dist2 = dot(rel, rel);
                    if (dist2 > h2 || ph_j != ph_i || j == index)
                        continue;
                    Real dist = sqrt(dist2);

                    Real m_j = params.pmass[ph_j];
                    Real inv_rho_j = invDensity[j];

                    Real Kij = 2.0f * params.invRho0[(uint)ph_j] / (inv_rho_i + inv_rho_j);
                    cohesion -= m_j * _d_cSpline(dist) * Kij / dist * rel;
                    // curvature -= (n_i-n_j)*Kij;
                }
            }
    force[index] = make_vec4r(cohesion * params.kcohesion); //+curvature*params.kcurvature;
}

DECLARE_KERNEL_TEMP(updateSurfaceTension, vec4r *force, vec4r *normal, Real *invDensity, vec4r *sortedPositionPhase, Real *sortedMass, uint *S2OParticleIndex, uint *cellStart, uint *cellEnd);

__global__ void _g_updateSurfaceTension(int size, vec4r *force, vec4r *normal, Real *invDensity, vec4r *sortedPositionPhase, Real *sortedMass, uint *S2OParticleIndex, uint *cellStart, uint *cellEnd)
{
    IF_IDX_VALID(size)
    _k_updateSurfaceTension<MemType::GPU>(i, force, normal, invDensity, sortedPositionPhase, sortedMass, S2OParticleIndex, cellStart, cellEnd);
}

template <MemType MT>
void callUpdateSurfaceTension(int size, VecArray<vec4r, MT> &force, VecArray<vec4r, MT> &normal, VecArray<Real, MT> &invDensity, VecArray<vec4r, MT> &sortedPositionPhase, VecArray<Real, MT> &sortedMass, VecArray<uint, MT> &S2OParticleIndex, VecArray<uint, MT> &cellStart, VecArray<uint, MT> &cellEnd)
{
    FILL_CALL_DEVICE_CODE(updateSurfaceTension, force.m_data, normal.m_data, invDensity.m_data, sortedPositionPhase.m_data, sortedMass.m_data, S2OParticleIndex.m_data, cellStart.m_data, cellEnd.m_data);
}
template void callUpdateSurfaceTension<MemType::CPU>(int, VecArray<vec4r, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &, VecArray<Real, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &, VecArray<Real, MemType::CPU> &, VecArray<uint, MemType::CPU> &, VecArray<uint, MemType::CPU> &, VecArray<uint, MemType::CPU> &);
template void callUpdateSurfaceTension<MemType::GPU>(int, VecArray<vec4r, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<Real, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<Real, MemType::GPU> &, VecArray<uint, MemType::GPU> &, VecArray<uint, MemType::GPU> &, VecArray<uint, MemType::GPU> &);

template <MemType MT>
__device__ void _k_updatePositionVelocity(int idx, Real dt, vec4r *sortedPositionPhase, vec3r *oldPos, vec4r *oldVel, vec4r *dvel, vec4r *force, uint *S2OParticleIndex, uint *cellStart, uint *cellEnd)
{

    vec3r x = make_vec3r(sortedPositionPhase[idx]);
    const uint oidx = S2OParticleIndex[idx]; // for ps
    const vec3r ox = oldPos[oidx];

    vec3r v = (x - ox) / dt + make_vec3r(force[idx]) * dt + make_vec3r(dvel[idx]);
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

    oldPos[oidx] = x;
    oldVel[oidx] = make_vec4r(v);
}
DECLARE_GPU_KERNEL_TEMP(updatePositionVelocity, Real dt, vec4r *sortedPositionPhase, vec3r *pos, vec4r *vel, vec4r *dvel, vec4r *force, uint *S2OParticleIndex, uint *cellStart, uint *cellEnd);

__global__ void _g_updatePositionVelocity(int size, Real dt, vec4r *sortedPositionPhase, vec3r *pos, vec4r *vel, vec4r *dvel, vec4r *force, uint *S2OParticleIndex, uint *cellStart, uint *cellEnd)
{
    IF_IDX_VALID(size)
    _k_updatePositionVelocity<MemType::GPU>(i, dt, sortedPositionPhase, pos, vel, dvel, force, S2OParticleIndex, cellStart, cellEnd);
}

template <MemType MT>
void callUpdatePositionVelocity(int size, Real dt, VecArray<vec4r, MT> &sortedPositionPhase, VecArray<vec3r, MT> &pos, VecArray<vec4r, MT> &vel, VecArray<vec4r, MT> &dvel, VecArray<vec4r, MT> &force, VecArray<uint, MT> &S2OParticleIndex, VecArray<uint, MT> &cellStart, VecArray<uint, MT> &cellEnd)
{
    FILL_CALL_GPU_DEVICE_CODE(updatePositionVelocity, dt, sortedPositionPhase.m_data, pos.m_data, vel.m_data, dvel.m_data, force.m_data, S2OParticleIndex.m_data, cellStart.m_data, cellEnd.m_data);
}
template void callUpdatePositionVelocity<MemType::CPU>(int, Real, VecArray<vec4r, MemType::CPU> &, VecArray<vec3r, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &, VecArray<uint, MemType::CPU> &, VecArray<uint, MemType::CPU> &, VecArray<uint, MemType::CPU> &);
template void callUpdatePositionVelocity<MemType::GPU>(int, Real, VecArray<vec4r, MemType::GPU> &, VecArray<vec3r, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<uint, MemType::GPU> &, VecArray<uint, MemType::GPU> &, VecArray<uint, MemType::GPU> &);

//// compute a virtual attract force
template <MemType MT>
__host__ __device__ void _k_updateVorticity(int index, vec4r *vorticity, Real *invDensity, vec4r *sortedPositionPhase, vec4r *vel, Real *sortedMass, uint *sortedPhase, uint *S2OParticleIndex, uint *cellStart, uint *cellEnd)
{

    const vec4r xph_i = sortedPositionPhase[index];
    const vec3r x_i = make_vec3r(xph_i);
    const uint ph_i = (uint)xph_i.w;

    if (!_d_isLiquid(ph_i))
        return;

    // get address in grid
    const uint oidx_i = S2OParticleIndex[index]; // for velocity
    const vec3r v_i = make_vec3r(vel[oidx_i]);
    vec3r vort = make_vec3r(0, 0, 0);
    const Real h = params.h;
    const Real h2 = params.h2;
    const Real spikyGradCoef = params.spikyGradCoef;

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
                    const vec4r xph_j = sortedPositionPhase[j];
                    const vec3r x_j = make_vec3r(xph_j);
                    const uint ph_j = (uint)xph_j.w;

                    const vec3r rel = x_i - x_j;
                    const Real dist2 = dot(rel, rel);
                    if (dist2 > h2 || j == index || !_d_isLiquid(ph_j))
                        continue;
                    const Real dist = sqrt(dist2);

                    const Real m_j = params.pmass[ph_j];
                    const Real inv_rho_j = invDensity[j];
                    const uint oidx_j = S2OParticleIndex[j];
                    const vec3r v_j = make_vec3r(vel[oidx_j]);

                    // vort += params.volume*cross(v_j-v_i, _d_spikyGrad(rel,dist));
                    vort += m_j * inv_rho_j * cross(v_j - v_i, _d_spikyGrad(rel, dist, h, spikyGradCoef));
                }
            }
    vorticity[index] = make_float4(vort, length(vort));
    // omega[index] = length(vort);
}
DECLARE_KERNEL_TEMP(updateVorticity, vec4r *vorticity, Real *invDensity, vec4r *sortedPositionPhase, vec4r *vel, Real *sortedMass, uint *sortedPhase, uint *S2OParticleIndex, uint *cellStart, uint *cellEnd);

__global__ void _g_updateVorticity(int size, vec4r *vorticity, Real *invDensity, vec4r *sortedPositionPhase, vec4r *vel, Real *sortedMass, uint *sortedPhase, uint *S2OParticleIndex, uint *cellStart, uint *cellEnd)
{
    IF_IDX_VALID(size)
    _k_updateVorticity<MemType::GPU>(i, vorticity, invDensity, sortedPositionPhase, vel, sortedMass, sortedPhase, S2OParticleIndex, cellStart, cellEnd);
}

template <MemType MT>
void callUpdateVorticity(int size, VecArray<vec4r, MT> &vorticity, VecArray<Real, MT> &invDensity, VecArray<vec4r, MT> &sortedPositionPhase, VecArray<vec4r, MT> &vel, VecArray<Real, MT> &sortedMass, VecArray<uint, MT> &sortedPhase, VecArray<uint, MT> &S2OParticleIndex, VecArray<uint, MT> &cellStart, VecArray<uint, MT> &cellEnd)
{
    FILL_CALL_DEVICE_CODE(updateVorticity, vorticity.m_data, invDensity.m_data, sortedPositionPhase.m_data, vel.m_data, sortedMass.m_data, sortedPhase.m_data, S2OParticleIndex.m_data, cellStart.m_data, cellEnd.m_data);
}
template void callUpdateVorticity<MemType::CPU>(int, VecArray<vec4r, MemType::CPU> &, VecArray<Real, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &, VecArray<Real, MemType::CPU> &, VecArray<uint, MemType::CPU> &, VecArray<uint, MemType::CPU> &, VecArray<uint, MemType::CPU> &, VecArray<uint, MemType::CPU> &);
template void callUpdateVorticity<MemType::GPU>(int, VecArray<vec4r, MemType::GPU> &, VecArray<Real, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<Real, MemType::GPU> &, VecArray<uint, MemType::GPU> &, VecArray<uint, MemType::GPU> &, VecArray<uint, MemType::GPU> &, VecArray<uint, MemType::GPU> &);

//// compute a virtual attract force
template <MemType MT>
__host__ __device__ void _k_applyVorticity(int index, Real dt, vec4r *vel, vec4r *vorticity, Real *invDensity, vec4r *sortedPositionPhase, Real *sortedMass, uint *sortedPhase, uint *S2OParticleIndex, uint *cellStart, uint *cellEnd)
{

    const vec4r xph_i = sortedPositionPhase[index];
    const vec3r x_i = make_vec3r(xph_i);
    const uint ph_i = (uint)xph_i.w;
    if (!_d_isLiquid(ph_i))
        return;

    const uint oidx_i = S2OParticleIndex[index]; // for velocity
    const vec3r v_i = make_vec3r(vel[oidx_i]);
    const vec3r vort_i = make_vec3r(vorticity[index]);

    vec3r eta = make_vec3r(0.0f);

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
                    const const vec4r xph_j = sortedPositionPhase[j];
                    const const vec3r x_j = make_vec3r(xph_j);
                    const const uint ph_j = (uint)xph_j.w;

                    const vec3r rel = x_i - x_j;
                    const Real dist2 = dot(rel, rel);
                    if (dist2 > params.h2 || j == index || !_d_isLiquid(ph_j))
                        continue;
                    const Real dist = sqrt(dist2);

                    const Real omega_j = vorticity[j].w;
                    const Real inv_rho_j = invDensity[j];
                    eta += omega_j * inv_rho_j * _d_spikyGrad(rel, dist);
                }
            }

    Real l_eta = length(eta);
    if (l_eta > 0)
    {
        const vec3r N = eta / l_eta;
        const vec3r f = params.kvorticity * cross(N, vort_i);
        vel[oidx_i] = make_vec4r(v_i + f * dt);
    }
}
DECLARE_KERNEL_TEMP(applyVorticity, Real dt, vec4r *vel, vec4r *vorticity, Real *invDensity, vec4r *sortedPositionPhase, Real *sortedMass, uint *sortedPhase, uint *S2OParticleIndex, uint *cellStart, uint *cellEnd);

__global__ void _g_applyVorticity(int size, Real dt, vec4r *vel, vec4r *vorticity, Real *invDensity, vec4r *sortedPositionPhase, Real *sortedMass, uint *sortedPhase, uint *S2OParticleIndex, uint *cellStart, uint *cellEnd)
{
    IF_IDX_VALID(size)
    _k_applyVorticity<MemType::GPU>(i, dt, vel, vorticity, invDensity, sortedPositionPhase, sortedMass, sortedPhase, S2OParticleIndex, cellStart, cellEnd);
}

template <MemType MT>
void callApplyVorticity(int size, Real dt, VecArray<vec4r, MT> &vel, VecArray<vec4r, MT> &voriticity, VecArray<Real, MT> &invDensity, VecArray<vec4r, MT> &sortedPositionPhase, VecArray<Real, MT> &sortedMass, VecArray<uint, MT> &sortedPhase, VecArray<uint, MT> &S2OParticleIndex, VecArray<uint, MT> &cellStart, VecArray<uint, MT> &cellEnd)
{
    FILL_CALL_DEVICE_CODE(applyVorticity, dt, vel.m_data, voriticity.m_data, invDensity.m_data, sortedPositionPhase.m_data, sortedMass.m_data, sortedPhase.m_data, S2OParticleIndex.m_data, cellStart.m_data, cellEnd.m_data);
}
template void callApplyVorticity<MemType::CPU>(int, Real, VecArray<vec4r, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &, VecArray<Real, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &, VecArray<Real, MemType::CPU> &, VecArray<uint, MemType::CPU> &, VecArray<uint, MemType::CPU> &, VecArray<uint, MemType::CPU> &, VecArray<uint, MemType::CPU> &);
template void callApplyVorticity<MemType::GPU>(int, Real, VecArray<vec4r, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<Real, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<Real, MemType::GPU> &, VecArray<uint, MemType::GPU> &, VecArray<uint, MemType::GPU> &, VecArray<uint, MemType::GPU> &, VecArray<uint, MemType::GPU> &);

//// single thread
template <MemType MT>
__host__ __device__ void _k_solveShapeMatching(int index, uint rigidConstraintCount, vec4r *tempStarPositionPhase, vec4r *sortedPositionPhase, uint *constraintStartIndex, uint *constraintParticleCount, vec3r *R, uint *constraintParticleMap, vec3r *q, Real *mass, uint *O2SParticleIndex, uint *S2OParticleIndex)
{
    // if (index >= rigidConstraintCount) return;
    int oidx = index;

    uint startIndex = constraintStartIndex[oidx];
    uint pcnt = constraintParticleCount[oidx];

    vec3r mcenter = make_vec3r(0.0f);
    Real msum = 0.f;
    //// mass center
    for (uint i = 0; i < pcnt; i++)
    {
        uint curIndex = startIndex + i;
        uint coidx = constraintParticleMap[curIndex];
        int csidx = O2SParticleIndex[coidx];

        // examine phase not rigid then return
        //  if (phase[csidx] > 0)
        //      return;

        Real m = params.pmass[PhaseType::Rigid]; /*mass[csidx]*/
        ;
        mcenter += make_vec3r(sortedPositionPhase[csidx]) * m;
        msum += m;
    }

    mcenter /= msum;

    vec3r ACol0 = make_vec3r(0.f);
    vec3r ACol1 = make_vec3r(0.f);
    vec3r ACol2 = make_vec3r(0.f);

    //// A
    for (uint i = 0; i < pcnt; i++)
    {
        uint curIndex = startIndex + i;
        uint coidx = constraintParticleMap[curIndex];
        int csidx = O2SParticleIndex[coidx];

        Real m = params.pmass[PhaseType::Rigid]; /*mass[csidx]*/
        ;
        vec3r qi = q[curIndex];
        vec3r pi = make_vec3r(sortedPositionPhase[csidx]) - mcenter;

        ACol0.x += m * pi.x * qi.x;
        ACol1.x += m * pi.x * qi.y;
        ACol2.x += m * pi.x * qi.z;

        ACol0.y += m * pi.y * qi.x;
        ACol1.y += m * pi.y * qi.y;
        ACol2.y += m * pi.y * qi.z;

        ACol0.z += m * pi.z * qi.x;
        ACol1.z += m * pi.z * qi.y;
        ACol2.z += m * pi.z * qi.z;
    }

    // polar decomposition
    vec3r R0 = R[oidx * 3];
    vec3r R1 = R[oidx * 3 + 1];
    vec3r R2 = R[oidx * 3 + 2];

    vec3r RCol0 = make_vec3r(R0.x, R1.x, R2.x);
    vec3r RCol1 = make_vec3r(R0.y, R1.y, R2.y);
    vec3r RCol2 = make_vec3r(R0.z, R1.z, R2.z);

    uint maxIter = 10;
    for (uint iter = 0; iter < maxIter; iter++)
    {
        vec3r omega = ((cross(RCol0, ACol0) + cross(RCol1, ACol1) + cross(RCol2, ACol2)) *
                       (1.f / fabs(dot(RCol0, ACol0) + dot(RCol1, ACol1) + dot(RCol2, ACol2)) + 1.0e-7f));

        Real w = length(omega);

        if (w < 1.0e-7f)
            break;

        vec3r axis = (1.f / w) * omega;

        vec3r R0Left;
        vec3r R1Left;
        vec3r R2Left;
        vec3r RCol0Right;
        vec3r RCol1Right;
        vec3r RCol2Right;
        // cal R angle=w axis=axis
        Real sina = sin(w);
        Real cosa = cos(w);
        Real oneMinusCosa = 1 - cosa;
        R0Left.x = cosa + axis.x * axis.x * oneMinusCosa;
        R0Left.y = axis.x * axis.y * oneMinusCosa - axis.z * sina;
        R0Left.z = axis.x * axis.z * oneMinusCosa + axis.y * sina;
        R1Left.x = axis.y * axis.x * oneMinusCosa + axis.z * sina;
        R1Left.y = cosa + axis.y * axis.y * oneMinusCosa;
        R1Left.z = axis.y * axis.z * oneMinusCosa - axis.x * sina;
        R2Left.x = axis.z * axis.x * oneMinusCosa - axis.y * sina;
        R2Left.y = axis.z * axis.y * oneMinusCosa + axis.x * sina;
        R2Left.z = cosa + axis.z * axis.z * oneMinusCosa;

        RCol0Right = RCol0;
        RCol1Right = RCol1;
        RCol2Right = RCol2;

        R0.x = dot(R0Left, RCol0Right);
        R0.y = dot(R0Left, RCol1Right);
        R0.z = dot(R0Left, RCol2Right);
        R1.x = dot(R1Left, RCol0Right);
        R1.y = dot(R1Left, RCol1Right);
        R1.z = dot(R1Left, RCol2Right);
        R2.x = dot(R2Left, RCol0Right);
        R2.y = dot(R2Left, RCol1Right);
        R2.z = dot(R2Left, RCol2Right);

        RCol0 = make_vec3r(R0.x, R1.x, R2.x);
        RCol1 = make_vec3r(R0.y, R1.y, R2.y);
        RCol2 = make_vec3r(R0.z, R1.z, R2.z);
    }

    // write back
    R[oidx * 3] = R0;
    R[oidx * 3 + 1] = R1;
    R[oidx * 3 + 2] = R2;

    for (uint i = 0; i < pcnt; i++)
    {
        uint curIndex = startIndex + i;
        uint coidx = constraintParticleMap[curIndex];
        int csidx = O2SParticleIndex[coidx];

        vec3r qi = q[curIndex];

        vec3r goaldelta;
        goaldelta.x = R0.x * qi.x + R0.y * qi.y + R0.z * qi.z;
        goaldelta.y = R1.x * qi.x + R1.y * qi.y + R1.z * qi.z;
        goaldelta.z = R2.x * qi.x + R2.y * qi.y + R2.z * qi.z;
        vec3r goal = mcenter + goaldelta;
        vec3r deltap = (goal - make_vec3r(sortedPositionPhase[csidx]));
        const vec4r xph = tempStarPositionPhase[csidx];
        tempStarPositionPhase[csidx] = make_vec4r(make_vec3r(xph) + deltap, xph.w);
    }
}
DECLARE_KERNEL_TEMP(solveShapeMatching, uint rigidConstraintCount, vec4r *tempStarPositionPhase, vec4r *sortedPositionPhase, uint *constraintStartIndex, uint *constraintParticleCount, vec3r *R, uint *constraintParticleMap, vec3r *q, Real *Mass, uint *O2SParticleIndex, uint *S2OParticleIndex);

__global__ void _g_solveShapeMatching(int size, uint rigidConstraintCount, vec4r *tempStarPositionPhase, vec4r *sortedPositionPhase, uint *constraintStartIndex, uint *constraintParticleCount, vec3r *R, uint *constraintParticleMap, vec3r *q, Real *Mass, uint *O2SParticleIndex, uint *S2OParticleIndex)
{
    IF_IDX_VALID(rigidConstraintCount)
    _k_solveShapeMatching<MemType::GPU>(i, rigidConstraintCount, tempStarPositionPhase, sortedPositionPhase, constraintStartIndex, constraintParticleCount, R, constraintParticleMap, q, Mass, O2SParticleIndex, S2OParticleIndex);
}

template <MemType MT>
void callSolveShapeMatching(int size, uint rigidConstraintCount, VecArray<vec4r, MT> &tempStarPositionPhase, VecArray<vec4r, MT> &sortedPositionPhase, VecArray<uint, MT> &constraintStartIndex, VecArray<uint, MT> &constraintParticleCount, VecArray<vec3r, MT> &r, VecArray<uint, MT> &constraintParticleMap, VecArray<vec3r, MT> &q, VecArray<Real, MT> &mass, VecArray<uint, MT> &O2SParticleIndex, VecArray<uint, MT> &S2OParticleIndex)
{
    FILL_CALL_DEVICE_CODE(solveShapeMatching, rigidConstraintCount, tempStarPositionPhase.m_data, sortedPositionPhase.m_data, constraintStartIndex.m_data, constraintParticleCount.m_data, r.m_data, constraintParticleMap.m_data, q.m_data, mass.m_data, O2SParticleIndex.m_data, S2OParticleIndex.m_data);
}
template void callSolveShapeMatching<MemType::CPU>(int, uint, VecArray<vec4r, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &, VecArray<uint, MemType::CPU> &, VecArray<uint, MemType::CPU> &, VecArray<vec3r, MemType::CPU> &, VecArray<uint, MemType::CPU> &, VecArray<vec3r, MemType::CPU> &, VecArray<Real, MemType::CPU> &, VecArray<uint, MemType::CPU> &, VecArray<uint, MemType::CPU> &);
template void callSolveShapeMatching<MemType::GPU>(int, uint, VecArray<vec4r, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<uint, MemType::GPU> &, VecArray<uint, MemType::GPU> &, VecArray<vec3r, MemType::GPU> &, VecArray<uint, MemType::GPU> &, VecArray<vec3r, MemType::GPU> &, VecArray<Real, MemType::GPU> &, VecArray<uint, MemType::GPU> &, VecArray<uint, MemType::GPU> &);

template <MemType MT>
__device__ void _k_updateNbrList(int index, NbrList *__restrict__ nbrLists, const vec4r *__restrict__ sortedPositionPhase, const uint *__restrict__ S2OParticleIndex, const uint *__restrict__ cellStart, const uint *__restrict__ cellEnd)
{
    // read particle data from sorted arrays
    const vec4r xph_i = sortedPositionPhase[index];
    const uint ph_i = (uint)xph_i.w;
    const vec3r x_i = make_vec3r(xph_i);
    const Real h = params.h;
    const Real h2 = params.h2;
    NbrList nl;

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
                    const vec4r xph_j = sortedPositionPhase[j];
                    const uint ph_j = (uint)xph_j.w;
                    const vec3r x_j = make_vec3r(xph_j);
                    // uint ph_j = sortedPhase[j];
                    // vec3r rel = x_i - sortedPos[j];
                    const vec3r rel = x_i - x_j;
                    const Real dist2 = dot(rel, rel);
                    if (dist2 > h2 || j == index)
                        continue;
                    const Real dist = __fsqrt_rn(dist2);
                    nl.add(j, dist);
                }
            }
    nbrLists[index] = nl;
}

DECLARE_GPU_KERNEL_TEMP(updateNbrList, NbrList *__restrict__ nbrLists, const vec4r *__restrict__ sortedPositionPhase, const uint *__restrict__ S2OParticleIndex, const uint *__restrict__ cellStart, const uint *__restrict__ cellEnd);

__global__ void _g_updateNbrList(int size, NbrList *__restrict__ nbrLists, const vec4r *__restrict__ sortedPositionPhase, const uint *__restrict__ S2OParticleIndex, const uint *__restrict__ cellStart, const uint *__restrict__ cellEnd)
{
    IF_IDX_VALID(size)
    _k_updateNbrList<MemType::GPU>(i, nbrLists, sortedPositionPhase, S2OParticleIndex, cellStart, cellEnd);
}

template <MemType MT>
void callUpdateNbrList(int size, VecArray<NbrList, MT> &nbrLists, VecArray<vec4r, MT> &sortedPositionPhase, VecArray<uint, MT> &S2OParticleIndex, VecArray<uint, MT> &cellStart, VecArray<uint, MT> &cellEnd)
{
    FILL_CALL_GPU_DEVICE_CODE(updateNbrList, nbrLists.m_data, sortedPositionPhase.m_data, S2OParticleIndex.m_data, cellStart.m_data, cellEnd.m_data);
}
template void callUpdateNbrList<MemType::CPU>(int, VecArray<NbrList, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &, VecArray<uint, MemType::CPU> &, VecArray<uint, MemType::CPU> &, VecArray<uint, MemType::CPU> &);
template void callUpdateNbrList<MemType::GPU>(int, VecArray<NbrList, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<uint, MemType::GPU> &, VecArray<uint, MemType::GPU> &, VecArray<uint, MemType::GPU> &);

template <MemType MT>
__device__ void _k_updateLambdaFast(int index, Real *__restrict__ lambda, Real *__restrict__ invDensity, const NbrList *__restrict__ nbrLists, const vec4r *__restrict__ sortedPositionPhase)
{
    // read particle data from sorted arrays
    const vec4r xph_i = sortedPositionPhase[index];
    const uint ph_i = (uint)xph_i.w;
    const vec3r x_i = make_vec3r(xph_i);
    Real inv_rho0_i = params.invRho0[(uint)ph_i];

    Real rho_i = params.zeroPoly6 * params.pmass[ph_i];
    Real sumGradC2 = params.lambdaRelaxation;
    const Real h = params.h;
    const Real h2 = params.h2;
    const Real poly6Coef = params.poly6Coef;
    const Real spikyGradCoef = params.spikyGradCoef;

    vec3r gradC_i = make_vec3r(0.f);
    Real curLambda = 0.f;

    const NbrList &nbrs = nbrLists[index];
    for (int jj = 0; jj < nbrs.cnt; jj++)
    {
        int j = 0;
        Real dist = 0;
        nbrs.get(jj, j, dist);
        if (!_d_isLiquid(ph_i))
            return;

        const vec4r xph_j = sortedPositionPhase[j];
        const uint ph_j = (uint)xph_j.w;
        const vec3r x_j = make_vec3r(xph_j);
        const vec3r rel = x_i - x_j;

        const Real m_j = params.pmass[ph_j];
        rho_i += _d_poly6(dist, h2, poly6Coef) * m_j;

        // cal gradCj
        const Real tmp = h - dist;
        const vec3r wSpiky = (__fdividef(spikyGradCoef, dist + 1e-8f) * tmp * tmp) * rel;
        // vec3r gradC_j = -m_j*inv_rho0_i*_d_spikyGrad(rel, dist, h, spikyGradCoef);
        const vec3r gradC_j = -m_j * inv_rho0_i * wSpiky;

        sumGradC2 += dot(gradC_j, gradC_j);
        gradC_i -= gradC_j;
    }

    const Real C = max(0.f, (rho_i * inv_rho0_i) - 1.f);

    // output
    lambda[index] = -__fdividef(C, (sumGradC2 + dot(gradC_i, gradC_i)));
    invDensity[index] = __fdividef(1.0f, rho_i);
}

DECLARE_GPU_KERNEL_TEMP(updateLambdaFast, Real *__restrict__ lambda, Real *__restrict__ invDensity, const NbrList *__restrict__ nbrLists, const vec4r *__restrict__ sortedPositionPhase);

__global__ void _g_updateLambdaFast(int size, Real *__restrict__ lambda, Real *__restrict__ invDensity, const NbrList *__restrict__ nbrLists, const vec4r *__restrict__ sortedPositionPhase)
{
    IF_IDX_VALID(size)
    _k_updateLambdaFast<MemType::GPU>(i, lambda, invDensity, nbrLists, sortedPositionPhase);
}

template <MemType MT>
void callUpdateLambdaFast(int size, VecArray<Real, MT> &lambda, VecArray<Real, MT> &invDensity, VecArray<NbrList, MT> &nbrLists, VecArray<vec4r, MT> &sortedPositionPhase)
{
    FILL_CALL_GPU_DEVICE_CODE(updateLambdaFast, lambda.m_data, invDensity.m_data, nbrLists.m_data, sortedPositionPhase.m_data);
}
template void callUpdateLambdaFast<MemType::CPU>(int, VecArray<Real, MemType::CPU> &, VecArray<Real, MemType::CPU> &, VecArray<NbrList, CPU> &, VecArray<vec4r, MemType::CPU> &);
template void callUpdateLambdaFast<MemType::GPU>(int, VecArray<Real, MemType::GPU> &, VecArray<Real, MemType::GPU> &, VecArray<NbrList, GPU> &, VecArray<vec4r, MemType::GPU> &);

//// solve fluid
template <MemType MT>
__device__ void _k_solveFluidFast(int index, vec4r *__restrict__ normal, vec4r *__restrict__ newPositionPhase, const Real *__restrict__ lambda, const Real *__restrict__ invDensity, const NbrList *__restrict__ nbrLists, const vec4r *__restrict__ sortedPositionPhase)
{

    const vec4r xph_i = sortedPositionPhase[index];
    // vec3r x_i = sortedPos[index];
    const vec3r x_i = make_vec3r(xph_i);
    const uint ph_i = (uint)xph_i.w;
    newPositionPhase[index] = xph_i;

    // uint ph_i = sortedPhase[index];
    if (!_d_isLiquid(ph_i))
        return;

    const Real lambda_i = lambda[index];
    const Real inv_rho0_i = params.invRho0[(uint)ph_i];
    const Real h = params.h;
    const Real h2 = params.h2;
    const Real poly6Coef = params.poly6Coef;
    const Real spikyGradCoef = params.spikyGradCoef;
    vec3r deltap = make_vec3r(0.0f);

    const NbrList &nbrs = nbrLists[index];
    for (int jj = 0; jj < nbrs.cnt; jj++)
    {
        int j = 0;
        Real dist = 0;
        nbrs.get(jj, j, dist);

        if (j == index)
            continue;

        const vec4r xph_j = sortedPositionPhase[j];
        const uint ph_j = (uint)xph_j.w;
        const vec3r x_j = make_vec3r(xph_j);
        const vec3r rel = x_i - x_j;

        const Real tmp = h - dist;
        const vec3r wSpiky = (__fdividef(spikyGradCoef, dist + 1e-8f) * tmp * tmp) * rel;

        deltap += params.pmass[ph_j] * (lambda_i + lambda[j]) * wSpiky;
    }
    newPositionPhase[index] = make_vec4r(_d_enforceBoundaryLocal(x_i + deltap * inv_rho0_i), xph_i.w);
}

DECLARE_GPU_KERNEL_TEMP(solveFluidFast, vec4r *__restrict__ normal, vec4r *__restrict__ newPositionPhase, const Real *__restrict__ lambda, const Real *__restrict__ invDensity, const NbrList *__restrict__ nbrLists, const vec4r *__restrict__ sortedPositionPhase);

__global__ void _g_solveFluidFast(int size, vec4r *__restrict__ normal, vec4r *__restrict__ newPositionPhase, const Real *__restrict__ lambda, const Real *__restrict__ invDensity, const NbrList *__restrict__ nbrLists, const vec4r *__restrict__ sortedPositionPhase)
{
    IF_IDX_VALID(size)
    _k_solveFluidFast<MemType::GPU>(i, normal, newPositionPhase, lambda, invDensity, nbrLists, sortedPositionPhase);
}

template <MemType MT>
void callSolveFluidFast(int size, VecArray<vec4r, MT> &normal, VecArray<vec4r, MT> &newPositionPhase, VecArray<Real, MT> &lambda, VecArray<Real, MT> &invDensity, VecArray<NbrList, MT> &nbrLists, VecArray<vec4r, MT> &sortedPositionPhase)
{
    FILL_CALL_GPU_DEVICE_CODE(solveFluidFast, normal.m_data, newPositionPhase.m_data, lambda.m_data, invDensity.m_data, nbrLists.m_data, sortedPositionPhase.m_data);
}
template void callSolveFluidFast<MemType::CPU>(int, VecArray<vec4r, MemType::CPU> &, VecArray<vec4r, MemType::CPU> &, VecArray<Real, MemType::CPU> &, VecArray<Real, MemType::CPU> &, VecArray<NbrList, CPU> &, VecArray<vec4r, MemType::CPU> &);
template void callSolveFluidFast<MemType::GPU>(int, VecArray<vec4r, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<Real, MemType::GPU> &, VecArray<Real, MemType::GPU> &, VecArray<NbrList, GPU> &, VecArray<vec4r, MemType::GPU> &);

template <MemType MT>
__device__ void _k_updateColorField(int index, Real *__restrict__ colorField, Real *__restrict__ invDensity, vec4r *__restrict__ sortedPositionPhase, const uint *__restrict__ cellStart, const uint *__restrict__ cellEnd)
{
    const vec4r xph_i = sortedPositionPhase[index];
    const vec3r x_i = make_vec3r(xph_i);
    const uint ph_i = (uint)xph_i.w;

    Real c = params.pmass[ph_i] * invDensity[index] * _d_poly6(0.0f);
    NEIGHBOR_BEGIN(x_i)

    const const vec4r xph_j = sortedPositionPhase[j];
    const const vec3r x_j = make_vec3r(xph_j);
    const const uint ph_j = (uint)xph_j.w;

    const vec3r rel = x_i - x_j;
    const Real dist2 = dot(rel, rel);
    if (dist2 > params.h2 || j == index || !_d_isLiquid(ph_j))
        continue;
    const Real dist = sqrt(dist2);

    c += params.pmass[ph_j] * invDensity[j] * _d_poly6(dist);

    NEIGHBOR_END

    colorField[index] = c;
}

__global__ void _g_updateColorField(int size, Real *__restrict__ colorField, Real *__restrict__ invDensity, vec4r *__restrict__ sortedPositionPhase, const uint *__restrict__ cellStart, const uint *__restrict__ cellEnd)
{
    IF_IDX_VALID(size)
    _k_updateColorField<MemType::GPU>(i, colorField, invDensity, sortedPositionPhase, cellStart, cellEnd);
}

template <MemType MT>
void callUpdateColorField(int size, VecArray<Real, MT> &colorField, VecArray<Real, MT> &invDensity, VecArray<vec4r, MT> &sortedPositionPhase, VecArray<uint, MT> &cellStart, VecArray<uint, MT> &cellEnd)
{
    FILL_CALL_GPU_DEVICE_CODE(updateColorField, colorField.m_data, invDensity.m_data, sortedPositionPhase.m_data, cellStart.m_data, cellEnd.m_data);
}
template void callUpdateColorField<MemType::GPU>(int, VecArray<Real, MemType::GPU> &, VecArray<Real, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<uint, MemType::GPU> &, VecArray<uint, MemType::GPU> &);

template <MemType MT>
__device__ void _k_updateNormal(int index, vec4r *__restrict__ normal, Real *__restrict__ colorField, vec4r *__restrict__ sortedPositionPhase, const uint *__restrict__ cellStart, const uint *__restrict__ cellEnd)
{
    const vec4r xph_i = sortedPositionPhase[index];
    const vec3r x_i = make_vec3r(xph_i);
    const uint ph_i = (uint)xph_i.w;

    vec3r n = make_vec3r(0.0f);
    uint count = 0;

    NEIGHBOR_BEGIN(x_i)

    const const vec4r xph_j = sortedPositionPhase[j];
    const const vec3r x_j = make_vec3r(xph_j);
    const const uint ph_j = (uint)xph_j.w;

    const vec3r rel = x_i - x_j;
    const Real dist2 = dot(rel, rel);
    if (dist2 > params.h2 || j == index || !_d_isLiquid(ph_j))
        continue;
    const Real dist = sqrt(dist2);
    count += 1;

    n += colorField[j] * _d_poly6Grad(rel, dist);

    NEIGHBOR_END

    if (dot(n, n) > 1e-2 && count < 30)
        n = normalize(n);
    else
        n = make_vec3r(0.0f);
    normal[index] = -make_vec4r(n, 0.0f);
}

__global__ void _g_updateNormal(int size, vec4r *__restrict__ normal, Real *__restrict__ colorField, vec4r *__restrict__ sortedPositionPhase, const uint *__restrict__ cellStart, const uint *__restrict__ cellEnd)
{
    IF_IDX_VALID(size)
    _k_updateNormal<MemType::GPU>(i, normal, colorField, sortedPositionPhase, cellStart, cellEnd);
}

template <MemType MT>
void callUpdateNormal(int size, VecArray<vec4r, MT> &normal, VecArray<Real, MT> &colorField, VecArray<vec4r, MT> &sortedPositionPhase, VecArray<uint, MT> &cellStart, VecArray<uint, MT> &cellEnd)
{
    FILL_CALL_GPU_DEVICE_CODE(updateNormal, normal.m_data, colorField.m_data, sortedPositionPhase.m_data, cellStart.m_data, cellEnd.m_data);
}
template void callUpdateNormal<MemType::GPU>(int, VecArray<vec4r, MemType::GPU> &, VecArray<Real, MemType::GPU> &, VecArray<vec4r, MemType::GPU> &, VecArray<uint, MemType::GPU> &, VecArray<uint, MemType::GPU> &);

template <MemType MT>
__device__ void _k_generateFoamParticle(int index, int size, int rigidbodySize, Real dt, int *foamParticleCount, vec3r *__restrict__ foamParticlePositionPhase, vec4r *__restrict__ foamParticleVelocity, Real *__restrict__ foamParticleLifetime, vec4r *__restrict__ sortedPositionPhase, vec4r *__restrict__ normal, vec4r *__restrict__ vel, uint *__restrict__ S2OParticleIndex, const uint *__restrict__ cellStart, const uint *__restrict__ cellEnd, vec3r *__restrict__ color)
{
    // if (size>= rigidbodySize + index) return;
    const vec4r xph_i = sortedPositionPhase[index];
    const vec3r x_i = make_vec3r(xph_i);
    const uint ph_i = (uint)xph_i.w;

    uint oidi = S2OParticleIndex[index];
    vec3r veli = make_vec3r(vel[oidi]);

    // calculate potential of trapping air
    Real Vdiff = 0.0f;
    NEIGHBOR_BEGIN(x_i)
    const vec4r xph_j = sortedPositionPhase[j];
    const vec3r x_j = make_vec3r(xph_j);
    const uint ph_j = (uint)xph_j.w;

    const vec3r rel = x_i - x_j;
    const Real dist2 = dot(rel, rel);
    if (dist2 > params.h2 || j == index || !_d_isLiquid(ph_j))
        continue;
    const Real dist = sqrt(dist2);

    uint oidj = S2OParticleIndex[j];
    vec3r relV = veli - make_vec3r(vel[oidj]);
    if (length(relV) > 1e-4)
    {
        Vdiff += length(relV) * (1 - dot(normalize(relV), normalize(rel))) * _d_rsWeight(dist);
    }
    NEIGHBOR_END

    Real Ita = _d_clampAndNormalize(Vdiff, 50.0f, 500.0f);

    // calculate potential of wave crest
    Real curvature = 0.0f;

    vec3r normi = make_vec3r(normal[index]);
    if (dot(veli, normi) >= 0.6f)
    {
        NEIGHBOR_BEGIN(x_i)

        const vec4r xph_j = sortedPositionPhase[j];
        const vec3r x_j = make_vec3r(xph_j);
        const uint ph_j = (uint)xph_j.w;

        vec3r rel = x_i - x_j;
        const Real dist2 = dot(rel, rel);
        if (dist2 > params.h2 || j == index || !_d_isLiquid(ph_j))
            continue;
        const Real dist = sqrt(dist2);

        if (dot(-rel, normi) < 0.0f)
        {
            vec3r normj = make_vec3r(normal[j]);
            curvature += (1 - dot(normi, normj)) * _d_rsWeight(dist);
        }

        NEIGHBOR_END
    }

    Real Iwc = _d_clampAndNormalize(curvature, 4.0f, 12.0f);

    // calculate potential of energy
    Real energy = 0.5f * params.pmass[ph_i] * dot(veli, veli);
    // if (index % 1000 == 0) {
    //     printf("%f\n", energy);
    // }
    Real Ik = _d_clampAndNormalize(energy, 5.0f, 30.0f);

    uint nd = 2.0 * Ik * (60 * Ita + 200 * Iwc) * dt;
    // if (nd > 0) {
    //     printf("%d %d %f %f %f %f\n", index, nd, Ik, Ita, Vdiff, Iwc);
    // }
    curandState state;
    curand_init(index, index, 0, &state);

    if (nd > 0 && curand_uniform(&state) < 0.5)
    {
        uint startIndex = atomicAdd(foamParticleCount, nd);
        atomicMin(foamParticleCount, size);
        if (startIndex + nd > size)
        {
            nd = size - startIndex;
        }

        // create axis e1,e2
        vec3r e1, e2;
        vec3r vn = normalize(veli);
        _d_getOrthogonalVectors(vn, e1, e2);

        e1 = params.particleRadius * e1;
        e2 = params.particleRadius * e2;

        Real lifetimeMin = 2.0f, lifetimeMax = 8.0f;

        for (int i = startIndex; i < startIndex + nd; i++)
        {
            Real Xr = abs(curand_uniform(&state));
            Real Xtheta = abs(curand_uniform(&state));
            Real Xh = abs(curand_uniform(&state));

            Real r = (1.0 + Xr) * params.particleRadius;
            Real theta = Xtheta * 2.0 * REAL_PI;
            Real h = (Xh - 0.5) * dt * length(veli);

            vec3r xd = x_i + r * cos(theta) * e1 + r * sin(theta) * e2 + h * vn;
            vec3r vd = r * cos(theta) * e1 + r * sin(theta) * e2 + veli;

            foamParticlePositionPhase[i] = xd;
            foamParticleVelocity[i] = make_vec4r(vd, 0.0f);
            foamParticleLifetime[i] = lifetimeMin + max(max(Ik, Ita), Iwc) * (lifetimeMax - lifetimeMin);
        }
    }
    // color[index] = make_vec3r(0.0, 1.0f, 1.0f) * Iwc;
    // color[index] = make_vec3r(0.0, 1.0f, 1.0f) * length(normi);
}

__global__ void _g_generateFoamParticle(int size, int rigidbodySize, Real dt, int *foamParticleCount, vec3r *__restrict__ foamParticlePositionPhase, vec4r *__restrict__ foamParticleVelocity, Real *__restrict__ foamParticleLifetime, vec4r *__restrict__ sortedPositionPhase, vec4r *__restrict__ normal, vec4r *__restrict__ vel, uint *__restrict__ S2OParticleIndex, const uint *__restrict__ cellStart, const uint *__restrict__ cellEnd, vec3r *__restrict__ color)
{
    IF_IDX_VALID(size)
    _k_generateFoamParticle<MemType::GPU>(i, size, rigidbodySize, dt, foamParticleCount, foamParticlePositionPhase, foamParticleVelocity, foamParticleLifetime, sortedPositionPhase, normal, vel, S2OParticleIndex, cellStart, cellEnd, color);
}

template <MemType MT>
void generateFoamParticle(int size, int rigidbodySize, Real dt, int *foamParticleCount, VecArray<vec3r, MT> &foamParticlePositionPhase, VecArray<vec4r, MT> &foamParticleVelocity, VecArray<Real, MT> &foamParticleLifetime, VecArray<vec4r, MT> &sortedPositionPhase, VecArray<vec4r, MT> &normal, VecArray<vec4r, MT> &vel, VecArray<uint, MT> &S2OParticleIndex, VecArray<uint, MT> &cellStart, VecArray<uint, MT> &cellEnd, VecArray<vec3r, MT> &color)
{
    FILL_CALL_GPU_DEVICE_CODE(generateFoamParticle, rigidbodySize, dt, foamParticleCount, foamParticlePositionPhase.m_data, foamParticleVelocity.m_data, foamParticleLifetime.m_data, sortedPositionPhase.m_data, normal.m_data, vel.m_data, S2OParticleIndex.m_data, cellStart.m_data, cellEnd.m_data, color.m_data);
}
template void generateFoamParticle<MemType::GPU>(int, int, Real, int *, VecArray<vec3r, MemType::GPU> &foamParticlePositionPhase, VecArray<vec4r, MemType::GPU> &foamParticleVelocity, VecArray<Real, MemType::GPU> &foamParticleLifetime, VecArray<vec4r, MemType::GPU> &sortedPositionPhase, VecArray<vec4r, MemType::GPU> &normal, VecArray<vec4r, MemType::GPU> &vel, VecArray<uint, MemType::GPU> &S2OParticleIndex, VecArray<uint, MemType::GPU> &cellStart, VecArray<uint, MemType::GPU> &cellEnd, VecArray<vec3r, MemType::GPU> &);

template <MemType MT>
__device__ void _k_advectFoamParticle(int index, int size, Real dt, vec3r *__restrict__ foamParticlePositionPhase, vec4r *__restrict__ foamParticleVelocity, Real *__restrict__ foamParticleLifetime, vec4r *__restrict__ sortedPositionPhase, vec4r *__restrict__ vel, uint *__restrict__ S2OParticleIndex, const uint *__restrict__ cellStart, const uint *__restrict__ cellEnd)
{
    vec3r x_i = foamParticlePositionPhase[index];

    int neighborCount = 0;
    Real weight = 0.0f;
    vec3r weightVel = make_vec3r(0.0f);
    NEIGHBOR_BEGIN(x_i)
    const const vec4r xph_j = sortedPositionPhase[j];
    const const vec3r x_j = make_vec3r(xph_j);
    const const uint ph_j = (uint)xph_j.w;

    const vec3r rel = x_i - x_j;
    const Real dist2 = dot(rel, rel);
    if (dist2 > params.h2 || j == index || !_d_isLiquid(ph_j))
        continue;
    const Real dist = sqrt(dist2);

    neighborCount += 1;
    Real w = _d_poly6(dist);
    weight += w;
    weightVel += w * make_vec3r(vel[j]);
    NEIGHBOR_END

    if (weight > 1e-4 && neighborCount >= 6)
    {
        weightVel = weightVel / weight;
    }

    vec3r veli = make_vec3r(foamParticleVelocity[index]);

    if (neighborCount < 6)
    {
        // Spray particle
        veli = veli + params.gravity * dt;
        x_i = x_i + veli * dt;
        foamParticlePositionPhase[index] = _d_enforceBoundaryLocal(x_i);
        foamParticleVelocity[index] = make_vec4r(veli, 0.0f);
        // foamParticleLifetime[index] -= dt/2;
        return;
    }
    else if (neighborCount < 20)
    {
        // Foam particle
        x_i = x_i + weightVel * dt;
        foamParticlePositionPhase[index] = _d_enforceBoundaryLocal(x_i);
    }
    else
    {
        // Bubble particle
        veli = veli - 2.0 * params.gravity * dt + 0.8 * (weightVel - veli);
        x_i = x_i + veli * dt;
        foamParticlePositionPhase[index] = _d_enforceBoundaryLocal(x_i);
        foamParticleVelocity[index] = make_vec4r(veli, 0.0f);
    }
    foamParticleLifetime[index] -= dt;
}

__global__ void _g_advectFoamParticle(int size, Real dt, vec3r *__restrict__ foamParticlePositionPhase, vec4r *__restrict__ foamParticleVelocity, Real *__restrict__ foamParticleLifetime, vec4r *__restrict__ sortedPositionPhase, vec4r *__restrict__ vel, uint *__restrict__ S2OParticleIndex, const uint *__restrict__ cellStart, const uint *__restrict__ cellEnd)
{
    IF_IDX_VALID(size)
    _k_advectFoamParticle<MemType::GPU>(i, size, dt, foamParticlePositionPhase, foamParticleVelocity, foamParticleLifetime, sortedPositionPhase, vel, S2OParticleIndex, cellStart, cellEnd);
}

template <MemType MT>
void advectFoamParticle(int size, Real dt, VecArray<vec3r, MT> &foamParticlePositionPhase, VecArray<vec4r, MT> &foamParticleVelocity, VecArray<Real, MT> &foamParticleLifetime, VecArray<vec4r, MT> &sortedPositionPhase, VecArray<vec4r, MT> &vel, VecArray<uint, MT> &S2OParticleIndex, VecArray<uint, MT> &cellStart, VecArray<uint, MT> &cellEnd)
{
    FILL_CALL_GPU_DEVICE_CODE(advectFoamParticle, dt, foamParticlePositionPhase.m_data, foamParticleVelocity.m_data, foamParticleLifetime.m_data, sortedPositionPhase.m_data, vel.m_data, S2OParticleIndex.m_data, cellStart.m_data, cellEnd.m_data);
}
template void advectFoamParticle<MemType::GPU>(int, Real, VecArray<vec3r, MemType::GPU> &foamParticlePositionPhase, VecArray<vec4r, MemType::GPU> &foamParticleVelocity, VecArray<Real, MemType::GPU> &foamParticleLifetime, VecArray<vec4r, MemType::GPU> &sortedPositionPhase, VecArray<vec4r, MemType::GPU> &vel, VecArray<uint, MemType::GPU> &S2OParticleIndex, VecArray<uint, MemType::GPU> &cellStart, VecArray<uint, MemType::GPU> &cellEnd);

__global__ void _g_removeFoamParticle(int size, int *foamParticleCount, vec3r *__restrict__ foamParticlePositionPhase, vec4r *__restrict__ foamParticleVelocity, Real *__restrict__ foamParticleLifetime)
{
    IF_IDX_VALID((*foamParticleCount))
    {
        if (foamParticleLifetime[i] < 0.0f)
        {
            foamParticlePositionPhase[i] = make_vec3r(0.0f);
            atomicMin(foamParticleCount, i);
            // printf("%d %d %f\n", i, (*foamParticleCount), foamParticleLifetime[i]);
        }
    }
}

__global__ void _g_removeFoamParticleSingleThread(int size, int *foamParticleCount, vec3r *__restrict__ foamParticlePositionPhase, vec4r *__restrict__ foamParticleVelocity, Real *__restrict__ foamParticleLifetime)
{
    int removeCount = 0;
    int curNum = *foamParticleCount;
    for (int i = 0; i < curNum; i++)
    {
        if (foamParticleLifetime[i] < 0.0f)
        {
            removeCount++;
        }
        else
        {
            foamParticlePositionPhase[i - removeCount] = foamParticlePositionPhase[i];
            foamParticleVelocity[i - removeCount] = foamParticleVelocity[i];
            foamParticleLifetime[i - removeCount] = foamParticleLifetime[i];
        }
    }
    *foamParticleCount = curNum - removeCount;
}

template <MemType MT>
void removeFoamParticle(int size, int *foamParticleCount, VecArray<vec3r, MT> &foamParticlePositionPhase, VecArray<vec4r, MT> &foamParticleVelocity, VecArray<Real, MT> &foamParticleLifetime)
{
    FILL_CALL_GPU_DEVICE_CODE(removeFoamParticle, foamParticleCount, foamParticlePositionPhase.m_data, foamParticleVelocity.m_data, foamParticleLifetime.m_data);
    // if constexpr(MT==MemType::GPU){
    //     _g_removeFoamParticleSingleThread<<<1,1>>>(size, foamParticleCount, foamParticlePositionPhase.m_data, foamParticleVelocity.m_data, foamParticleLifetime.m_data);
    //     getLastCudaError("arrayFill kernel failed");
    // }
}
template void removeFoamParticle<MemType::GPU>(int, int *, VecArray<vec3r, MemType::GPU> &foamParticlePositionPhase, VecArray<vec4r, MemType::GPU> &foamParticleVelocity, VecArray<Real, MemType::GPU> &foamParticleLifetime);

__global__ void _g_initUselessParticle(int size, int *foamParticleCount, vec3r *__restrict__ foamParticlePositionPhase, vec4r *__restrict__ foamParticleVelocity, Real *__restrict__ foamParticleLifetime)
{
    IF_IDX_VALID(size)
    {
        if (i >= *foamParticleCount)
        {
            foamParticlePositionPhase[i] = make_vec3r(0.0f);
        }
    }
}

template <MemType MT>
void initUselessParticle(int size, int *foamParticleCount, VecArray<vec3r, MT> &foamParticlePositionPhase, VecArray<vec4r, MT> &foamParticleVelocity, VecArray<Real, MT> &foamParticleLifetime)
{
    FILL_CALL_GPU_DEVICE_CODE(initUselessParticle, foamParticleCount, foamParticlePositionPhase.m_data, foamParticleVelocity.m_data, foamParticleLifetime.m_data);
}
template void initUselessParticle<MemType::GPU>(int, int *, VecArray<vec3r, MemType::GPU> &foamParticlePositionPhase, VecArray<vec4r, MemType::GPU> &foamParticleVelocity, VecArray<Real, MemType::GPU> &foamParticleLifetime);

__global__ void _g_initS20FoamParticleIndex(int size, uint *__restrict__ S20FoamParticleIndex)
{
    IF_IDX_VALID(size)
    {
        S20FoamParticleIndex[i] = i;
    }
}

__global__ void _g_sortFoamParticle(int size, uint *__restrict__ S20FoamParticleIndex, vec3r *__restrict__ foamParticlePositionPhase, vec3r *__restrict__ tempFoamParticlePositionPhase, vec4r *__restrict__ foamParticleVelocity, vec4r *__restrict__ tempFoamParticleVelocity, Real *__restrict__ foamParticleLifetime)
{
    IF_IDX_VALID(size)
    {
        int index = S20FoamParticleIndex[i];
        tempFoamParticlePositionPhase[i] = foamParticlePositionPhase[index];
        tempFoamParticleVelocity[i] = foamParticleVelocity[index];
    }
}

__global__ void _g_copyFoamParticle(int size, vec3r *__restrict__ foamParticlePositionPhase, vec3r *__restrict__ tempFoamParticlePositionPhase, vec4r *__restrict__ foamParticleVelocity, vec4r *__restrict__ tempFoamParticleVelocity, Real *__restrict__ foamParticleLifetime)
{
    IF_IDX_VALID(size)
    {
        foamParticlePositionPhase[i] = tempFoamParticlePositionPhase[i];
        foamParticleVelocity[i] = tempFoamParticleVelocity[i];
    }
}

__global__ void _g_printFoamParticle(int size, uint *__restrict__ S20FoamParticleIndex, vec3r *__restrict__ foamParticlePositionPhase, vec4r *__restrict__ foamParticleVelocity, Real *__restrict__ foamParticleLifetime)
{
    printf("--------------------_g_printFoamParticleStart----------\n");
    for (int k = 0; k < size; k += int(size / 100) + 1)
    {
        printf("%d\t%d\t%f\n", k, S20FoamParticleIndex[k], foamParticleLifetime[k]);
    }
    printf("--------------------_g_printFoamParticleEnd----------\n");
}

template <MemType MT>
void sortFoamParticle(int size, VecArray<uint, MemType::GPU> S20FoamParticleIndex, VecArray<vec3r, MT> &foamParticlePositionPhase, VecArray<vec3r, MT> &tempFoamParticlePositionPhase, VecArray<vec4r, MT> &foamParticleVelocity, VecArray<vec4r, MT> &tempFoamParticleVelocity, VecArray<Real, MT> &foamParticleLifetime)
{
    FILL_CALL_GPU_DEVICE_CODE(initS20FoamParticleIndex, S20FoamParticleIndex.m_data);
    thrust::stable_sort_by_key(
        thrust::device,
        thrust::device_ptr<Real>(foamParticleLifetime.m_data),
        thrust::device_ptr<Real>(foamParticleLifetime.m_data + size),
        thrust::device_ptr<uint>(S20FoamParticleIndex.m_data),
        thrust::greater<Real>());

    //_g_printFoamParticle<<<1,1>>>(size, S20FoamParticleIndex.m_data, foamParticlePositionPhase.m_data, foamParticleVelocity.m_data, foamParticleLifetime.m_data);
    // getLastCudaError("arrayFill kernel failed");

    FILL_CALL_GPU_DEVICE_CODE(sortFoamParticle, S20FoamParticleIndex.m_data, foamParticlePositionPhase.m_data, tempFoamParticlePositionPhase.m_data, foamParticleVelocity.m_data, tempFoamParticleVelocity.m_data, foamParticleLifetime.m_data);
    FILL_CALL_GPU_DEVICE_CODE(copyFoamParticle, foamParticlePositionPhase.m_data, tempFoamParticlePositionPhase.m_data, foamParticleVelocity.m_data, tempFoamParticleVelocity.m_data, foamParticleLifetime.m_data);
}
template void sortFoamParticle<MemType::GPU>(int, VecArray<uint, MemType::GPU> S20FoamParticleIndex, VecArray<vec3r, MemType::GPU> &foamParticlePositionPhase, VecArray<vec3r, MemType::GPU> &tempFoamParticlePositionPhase, VecArray<vec4r, MemType::GPU> &foamParticleVelocity, VecArray<vec4r, MemType::GPU> &tempFoamParticleVelocity, VecArray<Real, MemType::GPU> &foamParticleLifetime);

__global__ void _g_collideTerrain(int size, vec4r *__restrict__ sortedPositionPhase, Real *__restrict__ terrainHeight, vec3r originPos, uint edgeCellNum, Real cellSize)
{
    IF_IDX_VALID(size)
    {
        vec3r pos = make_vec3r(sortedPositionPhase[i]);
        vec3r posInCell = (pos - originPos) / cellSize;
        int x = int(posInCell.x);
        int z = int(posInCell.z);
        Real a = posInCell.x - x;
        Real b = posInCell.z - z;

        if (x >= 0 && x < edgeCellNum - 1 && z >= 0 && z < edgeCellNum - 1)
        {
            Real h1 = terrainHeight[x * edgeCellNum + z];
            Real h2 = terrainHeight[(x + 1) * edgeCellNum + z];
            Real h3 = terrainHeight[x * edgeCellNum + z + 1];
            Real h4 = terrainHeight[(x + 1) * edgeCellNum + z + 1];
            Real h = (1 - a) * (1 - b) * h1 + a * (1 - b) * h2 + (1 - a) * b * h3 + a * b * h4;
            if (pos.y < h)
            {
                pos.y = h;
                sortedPositionPhase[i] = make_vec4r(pos, sortedPositionPhase[i].w);
            }
        }
    }
}

template <MemType MT>
void callCollideTerrain(int size, VecArray<vec4r, MT> &sortedPositionPhase, VecArray<Real, MT> &terrainHeight, vec3r originPos, uint edgeCellNum, Real cellSize)
{
    FILL_CALL_GPU_DEVICE_CODE(collideTerrain, sortedPositionPhase.m_data, terrainHeight.m_data, originPos, edgeCellNum, cellSize);
}
template void callCollideTerrain<MemType::GPU>(int, VecArray<vec4r, MemType::GPU> &sortedPositionPhase, VecArray<Real, MemType::GPU> &terrainHeight, vec3r, uint, Real);

__global__ void _g_collideTerrainFoam(int size, vec3r *__restrict__ foamParticlePosition, Real *__restrict__ terrainHeight, vec3r originPos, uint edgeCellNum, Real cellSize)
{
    IF_IDX_VALID(size)
    {
        vec3r pos = foamParticlePosition[i];
        vec3r posInCell = (pos - originPos) / cellSize;
        int x = int(posInCell.x);
        int z = int(posInCell.z);
        Real a = posInCell.x - x;
        Real b = posInCell.z - z;
        if (x >= 0 && x < edgeCellNum - 1 && z >= 0 && z < edgeCellNum - 1)
        {
            Real h1 = terrainHeight[x * edgeCellNum + z];
            Real h2 = terrainHeight[(x + 1) * edgeCellNum + z];
            Real h3 = terrainHeight[x * edgeCellNum + z + 1];
            Real h4 = terrainHeight[(x + 1) * edgeCellNum + z + 1];
            Real h = (1 - a) * (1 - b) * h1 + a * (1 - b) * h2 + (1 - a) * b * h3 + a * b * h4;
            if (pos.y < h)
            {
                pos.y = h;
                foamParticlePosition[i] = pos;
            }
        }
    }
}

template <MemType MT>
void callCollideTerrainFoam(int size, VecArray<vec3r, MT> &foamParticlePosition, VecArray<Real, MT> &terrainHeight, vec3r originPos, uint edgeCellNum, Real cellSize)
{
    FILL_CALL_GPU_DEVICE_CODE(collideTerrainFoam, foamParticlePosition.m_data, terrainHeight.m_data, originPos, edgeCellNum, cellSize);
}
template void callCollideTerrainFoam<MemType::GPU>(int, VecArray<vec3r, MemType::GPU> &foamParticlePosition, VecArray<Real, MemType::GPU> &terrainHeight, vec3r, uint, Real);

__global__ void _g_checkConstraint(int size, uint *__restrict__ start, uint *__restrict__ count)
{
    IF_IDX_VALID(size)
    {
        printf("start[%d] = %d, count[%d] = %d\n", i, start[i], i, count[i]);
    }
}

void checkConstraint(int size, VecArray<uint, MemType::GPU> &start, VecArray<uint, MemType::GPU> &count)
{
    unsigned int nb = 0, nt = 0;
    computeCudaThread(size, PE_CUDA_BLOCKS, nb, nt);
    _g_checkConstraint<<<nb, nt>>>(size, start.m_data, count.m_data);
    getLastCudaError("arrayFill kernel failed");
}
PHYS_NAMESPACE_END
