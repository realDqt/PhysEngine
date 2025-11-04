#include "object/grid_fluid.h"
#include "curand_kernel.h"
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"
#include "object/particle_fluid_util.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

PHYS_NAMESPACE_BEGIN

__constant__ GridSimParams gridParams;

/**
 * @brief CUDA kernel to update the density values by applying a decay factor.
 *
 * @param numCells Number of cells.
 * @param dt Time step.
 * @param density Array of density values.
 */
__global__ void _g_disappear(int numCells, Real dt, Real* __restrict__ density)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numCells) return;

    // Apply decay factor to density value
    density[i] *= exp(-gridParams.disappearRate * dt);
}

/**
 * @brief Call the CUDA kernel to update the density values by applying a decay factor.
 *
 * @tparam MT Memory type.
 * @param size Number of elements.
 * @param dt Time step.
 * @param density Array of density values.
 */
template<MemType MT>
void callDisappear(int size, Real dt, VecArray<Real, MT>& density){
    // Call the CUDA kernel to update density values
    FILL_CALL_GPU_DEVICE_CODE(disappear, dt, density.m_data);
}

// Instantiate the template for GPU memory type
template void callDisappear<MemType::GPU>(int, Real, VecArray<Real, MemType::GPU>&);

void setGridParameters(GridSimParams* hostParams)
{
    // copy parameters to constant memory
    //// const hparams on CPU
    hGridParams = *hostParams;
    //// const gridParams on GPU
    checkCudaErrors(cudaMemcpyToSymbol(gridParams, hostParams, sizeof(GridSimParams)));
}
///////////////////////////////////////////////////
// This template function performs diffusion calculations for a given index
template<MemType MT>
__device__ void _k_diffuse(int index, Real dt, Real* __restrict__ tempDensity, Real* __restrict__ density)
{
    // Get the grid index for the current thread
    int3 gridIdx = _d_getGridIdx(index);
    
    // Check if the grid index is valid
    if (!_d_isValid(gridIdx))
    {
        // Set the temporary density to 0 and return
        tempDensity[index] = 0;
        return;
    }
    
    // Calculate the indices of the neighboring cells
    int3 idxL = make_int3(max(0, gridIdx.x - 1), gridIdx.y, gridIdx.z); // Calculate the index of the left neighboring cell
    int3 idxR = make_int3(min(gridParams.gridSize.x - 1, gridIdx.x + 1), gridIdx.y, gridIdx.z); // Calculate the index of the right neighboring cell
    int3 idxB = make_int3(gridIdx.x, max(0, gridIdx.y - 1), gridIdx.z); // Calculate the index of the bottom neighboring cell
    int3 idxT = make_int3(gridIdx.x, min(gridParams.gridSize.y - 1, gridIdx.y + 1), gridIdx.z); // Calculate the index of the top neighboring cell
    int3 idxD = make_int3(gridIdx.x, gridIdx.y, max(0, gridIdx.z - 1)); // Calculate the index of the down neighboring cell
    int3 idxU = make_int3(gridIdx.x, gridIdx.y, min(gridParams.gridSize.z - 1, gridIdx.z + 1)); // Calculate the index of the up neighboring cell
    
    // Get the density values of the neighboring cells
    Real L = density[_d_getIdx(idxL)]; // Density value of the left neighboring cell
    Real R = density[_d_getIdx(idxR)]; // Density value of the right neighboring cell
    Real B = density[_d_getIdx(idxB)]; // Density value of the bottom neighboring cell
    Real T = density[_d_getIdx(idxT)]; // Density value of the top neighboring cell
    Real D = density[_d_getIdx(idxD)]; // Density value of the down neighboring cell
    Real U = density[_d_getIdx(idxU)]; // Density value of the up neighboring cell
    
    // Perform diffusion calculation using the neighboring densities
    tempDensity[index] = density[index] + dt * gridParams.kdiffusion * gridParams.invCellLength * gridParams.invCellLength
        * (L + R + B + T + D + U - 6 * density[index]);
}

// Declare the kernel template for diffusion
DECLARE_KERNEL_TEMP(diffuse, Real dt, Real* __restrict__ tempDensity, Real* __restrict__ density);

// Global kernel function for diffusion
__global__ void _g_diffuse(int size, Real dt, Real* __restrict__ tempDensity, Real* __restrict__ density)
{
    // Check if the thread index is valid
    IF_IDX_VALID(size) _k_diffuse<MemType::GPU>(i, dt, tempDensity, density);
}

/**
 * Perform diffusion on the given data arrays.
 *
 * @tparam MT The memory type of the data arrays.
 * @param size The size of the data arrays.
 * @param dt The time step for diffusion.
 * @param tempDensity The temporary density data array.
 * @param density The density data array.
 */
template<MemType MT>
void callDiffuse(int size, Real dt, VecArray<Real, MT>& tempDensity, VecArray<Real, MT>& density)
{
    // Call the GPU device code for diffusion
    FILL_CALL_GPU_DEVICE_CODE(diffuse, dt, tempDensity.m_data, density.m_data);
}

// Explicit template instantiation for GPU memory type
template void callDiffuse<MemType::GPU>(int, Real, VecArray<Real, MemType::GPU>&, VecArray<Real, MemType::GPU>&);
//////////////////////////////////////////////////////////
template<MemType MT>
__device__ void _k_advectProperty(int index, Real dt, Real* __restrict__ tempDensity, Real* __restrict__ density, vec3r* __restrict__ velocity) 
{
    // Get the grid index of the current element
    int3 gridIdx = _d_getGridIdx(index);
    
    // Check if the grid index is valid
    if (!_d_isValid(gridIdx))
    {
        // If not valid, set the temporary density to 0 and return
        tempDensity[index] = 0;
        return;
    }
    
    // Get the velocity at the current index
    vec3r vel = velocity[index];
    
    // Calculate the new position based on the velocity and time step
    float x = gridIdx.x - dt * vel.x * gridParams.invCellLength;
    float y = gridIdx.y - dt * vel.y * gridParams.invCellLength;
    float z = gridIdx.z - dt * vel.z * gridParams.invCellLength;

    // Check if the new position is outside the border of the grid
    // If so, clamp the position to the border
    if (x < 0.5f) x = 0.5f; if (x > gridParams.gridSize.x - 1.5f) x = gridParams.gridSize.x - 1.5f;
    if (y < 0.5f) y = 0.5f; if (y > gridParams.gridSize.y - 1.5f) y = gridParams.gridSize.y - 1.5f;
    if (z < 0.5f) z = 0.5f; if (z > gridParams.gridSize.z - 1.5f) z = gridParams.gridSize.z - 1.5f;

    // Create a vector representing the new position
    vec3r uvw = make_vec3r(x, y, z);
    
    // Sample the density at the new position and store it in the temporary density array
    tempDensity[index] = _d_sampleValue(density, uvw);
}

// Declare the kernel for advecting the property
DECLARE_KERNEL_TEMP(advectProperty, Real dt, Real* __restrict__ tempDensity, Real* __restrict__ density, vec3r* __restrict__ velocity);

// Define the global function for launching the advectProperty kernel
__global__ void _g_advectProperty(int size, Real dt, Real* __restrict__ tempDensity, Real* __restrict__ density, vec3r* __restrict__ velocity)
{
    // Check if the current index is valid
    IF_IDX_VALID(size)
        // Call the kernel to advect the property for the current index
        _k_advectProperty<MemType::GPU>(i, dt, tempDensity, density, velocity);
}

// Template function for calling the advectProperty kernel
template<MemType MT>
void callAdvectProperty(int size, Real dt, VecArray<Real, MT>& tempDensity, VecArray<Real, MT>& density, VecArray<vec3r, MT>& velocity)
{
    // Call the GPU device code to fill in the call for the advectProperty kernel
    FILL_CALL_GPU_DEVICE_CODE(advectProperty, dt, tempDensity.m_data, density.m_data, velocity.m_data);
}

// Explicit instantiation of the callAdvectProperty template function for GPU memory type
template void callAdvectProperty<MemType::GPU>(int, Real, VecArray<Real, MemType::GPU>&, VecArray<Real, MemType::GPU>&, VecArray<vec3r, MemType::GPU>&);
//////////////////////////////////////////////////////////
/**
 * Advects the velocity using the given time step.
 *
 * @tparam MT The memory type.
 * @param index The index of the current cell.
 * @param dt The time step.
 * @param tempVelocity The temporary velocity array.
 * @param velocity The velocity array.
 */
template<MemType MT>
__device__ void _k_advectVelocity(int index, Real dt, vec3r* tempVelocity, vec3r* velocity)
{
    // Get the grid index of the current cell
    int3 gridIdx = _d_getGridIdx(index);
    
    // Check if the cell is valid
    if (!_d_isValid(gridIdx))
    {
        // If the cell is invalid, set tempVelocity to zero and return
        tempVelocity[index] = make_vec3r(0, 0, 0);
        return;
    }
    vec3r vel = velocity[index];
    
    // Calculate the new position based on the velocity and time step
    float x = gridIdx.x - dt * vel.x * gridParams.invCellLength;
    float y = gridIdx.y - dt * vel.y * gridParams.invCellLength;
    float z = gridIdx.z - dt * vel.z * gridParams.invCellLength;
    
    // Handle border cases
    if (x < 0.5f) x = 0.5f;
    if (x > gridParams.gridSize.x - 1.5f) x = gridParams.gridSize.x - 1.5f;
    if (y < 0.5f) y = 0.5f;
    if (y > gridParams.gridSize.y - 1.5f) y = gridParams.gridSize.y - 1.5f;
    if (z < 0.5f) z = 0.5f;
    if (z > gridParams.gridSize.z - 1.5f) z = gridParams.gridSize.z - 1.5f;
    
    // Create a new position vector
    vec3r uvw = make_vec3r(x, y, z);
    
    // Sample the velocity at the new position
    tempVelocity[index] = _d_sampleVector(velocity, uvw);
}

// Declare the kernel template
// This template declares the kernel function advectVelocity
DECLARE_KERNEL_TEMP(advectVelocity, Real dt, vec3r* tempVelocity, vec3r* velocity);

// Define the global kernel function
// This function advects the velocity field using the tempVelocity field and a given time step size
__global__ void _g_advectVelocity(int size, Real dt, vec3r* tempVelocity, vec3r* velocity)
{
    // Check if the current thread index is valid
    IF_IDX_VALID(size)
        // Call the kernel function _k_advectVelocity with the current thread index i
        _k_advectVelocity<MemType::GPU>(i, dt, tempVelocity, velocity);
}

/**
 * Advects the velocity field using the given time step.
 *
 * @tparam MT The memory type of the velocity arrays.
 * @param size The size of the velocity arrays.
 * @param dt The time step.
 * @param tempVelocity The temporary velocity array.
 * @param velocity The velocity array to be advected.
 */
template<MemType MT>
void callAdvectVelocity(int size, Real dt, VecArray<vec3r, MT>& tempVelocity, VecArray<vec3r, MT>& velocity)
{
    // Call the GPU device code
    FILL_CALL_GPU_DEVICE_CODE(advectVelocity, dt, tempVelocity.m_data, velocity.m_data);
}
// Explicitly instantiate the template for GPU memory type
template void callAdvectVelocity<MemType::GPU>(int, Real, VecArray<vec3r, MemType::GPU>&, VecArray<vec3r, MemType::GPU>&);
//////////////////////////////////////////////////////////////
// This function adds density to a grid cell based on a given source position and radius.
// Parameters:
//   - index: the index of the current thread
//   - source: the position of the source
//   - radius: the radius within which the density should be added
//   - newDensity: the value of the new density to be added
//   - density: an array containing the density values for each grid cell
template<MemType MT>
__device__ void _k_addDensity(int index, int3 source, int radius, Real newDensity, Real* __restrict__ density)
{
    // Get the grid index of the current thread
    int3 gridIdx = _d_getGridIdx(index);

    // Check if the grid index is valid
    if (!_d_isValid(gridIdx))
    {
        // If not valid, set the density to 0 and return
        density[index] = 0;
        return;
    }

    // Calculate the squared distance between the grid index and the source
    int distance = _d_getDistance2(gridIdx - source);

    // Check if the distance is within the radius
    if (distance < radius)
        // If within the radius, set the density to the new density
        density[index] = newDensity;
}

// Declare the kernel template for adding density
DECLARE_KERNEL_TEMP(addDensity, int3 source, int radius, Real newDensity, Real* __restrict__ density);

__global__ void _g_addDensity(int size, int3 source, int radius, Real newDensity, Real* __restrict__ density)
{
    // Check if the current thread index is valid
    IF_IDX_VALID(size)
        // Call the kernel function to add density for the current thread
        _k_addDensity<MemType::GPU>(i, source, radius, newDensity, density);
}

template<MemType MT>
void callAddDensity(int size, int3 source, int radius, Real newDensity, VecArray<Real, MT>& density)
{
    FILL_CALL_GPU_DEVICE_CODE(addDensity, source, radius, newDensity, density.m_data);
}
template void callAddDensity<MemType::GPU>(int, int3, int, Real, VecArray<Real, MemType::GPU>&);
///////////////////////////////////////////////////////////
/**
 * Adds buoyancy to the velocity based on density and temperature.
 * @tparam MT The memory type.
 * @param index The index of the element.
 * @param dt The time step.
 * @param density Pointer to the density array.
 * @param temperature Pointer to the temperature array.
 * @param velocity Pointer to the velocity array.
 */
template<MemType MT>
__device__ void _k_addBuoyancy(int index, Real dt, Real* __restrict__ density, Real* __restrict__ temperature, vec3r* __restrict__ velocity)
{
    velocity[index] += gridParams.gravity * dt * (density[index] * gridParams.bdensity - temperature[index] * gridParams.btemperature);
}

/**
 * Calls the _k_addBuoyancy kernel for each valid index.
 * @param size The size of the arrays.
 * @param dt The time step.
 * @param density Pointer to the density array.
 * @param temperature Pointer to the temperature array.
 * @param velocity Pointer to the velocity array.
 */
__global__ void _g_addBuoyancy(int size, Real dt, Real* __restrict__ density, Real* __restrict__ temperature, vec3r* __restrict__ velocity)
{
    IF_IDX_VALID(size)
        _k_addBuoyancy<MemType::GPU>(i, dt, density, temperature, velocity);
}

/**
 * Calls the _g_addBuoyancy kernel for GPU memory type.
 * @tparam MT The memory type.
 * @param size The size of the arrays.
 * @param dt The time step.
 * @param density Reference to the density array.
 * @param temperature Reference to the temperature array.
 * @param velocity Reference to the velocity array.
 */
template<MemType MT>
void callAddBuoyancy(int size, Real dt, VecArray<Real, MT>& density, VecArray<Real, MT>& temperature, VecArray<vec3r, MT>& velocity)
{
    FILL_CALL_GPU_DEVICE_CODE(addBuoyancy, dt, density.m_data, temperature.m_data, velocity.m_data);
}

// Explicit instantiation for GPU memory type
template void callAddBuoyancy<MemType::GPU>(int, Real, VecArray<Real, MemType::GPU>&, VecArray<Real, MemType::GPU>&, VecArray<vec3r, MemType::GPU>&);
//////////////////////////////////////////////////////////
/**
 * \brief Add wind force to the velocity field.
 *
 * This function adds the wind force to the velocity field based on the given parameters.
 * It updates the velocity at the specified index by adding the windForce multiplied by dt, 
 * density, and temperature.
 *
 * \tparam MT The memory type (CPU or GPU).
 * \param index The index of the velocity field to update.
 * \param dt The time step.
 * \param windForce The wind force vector.
 * \param density The density array.
 * \param temperature The temperature array.
 * \param velocity The velocity array.
 */
template<MemType MT>
__device__ void _k_addWind(int index, Real dt, vec3r windForce, Real* __restrict__ density, Real* __restrict__ temperature, vec3r* __restrict__ velocity)
{
    // Add wind force to the velocity field
    velocity[index] += windForce * dt * (density[index] * gridParams.bdensity - temperature[index] * gridParams.btemperature);
}

/**
 * \brief GPU kernel for adding wind force to the velocity field.
 *
 * This GPU kernel function calls the '_k_addWind' function for each valid index in the velocity field.
 *
 * \param size The size of the velocity field.
 * \param dt The time step.
 * \param windForce The wind force vector.
 * \param density The density array.
 * \param temperature The temperature array.
 * \param velocity The velocity array.
 */
__global__ void _g_addWind(int size, Real dt, vec3r windForce, Real* __restrict__ density, Real* __restrict__ temperature, vec3r* __restrict__ velocity)
{
    IF_IDX_VALID(size)
        _k_addWind<MemType::GPU>(i, dt, windForce, density, temperature, velocity);
}

/**
 * \brief Call the GPU device code for adding wind force to the velocity field.
 *
 * This function calls the GPU device code to add wind force to the velocity field.
 *
 * \tparam MT The memory type (CPU or GPU).
 * \param size The size of the velocity field.
 * \param dt The time step.
 * \param windForce The wind force vector.
 * \param density The density array.
 * \param temperature The temperature array.
 * \param velocity The velocity array.
 */
template<MemType MT>
void callAddWind(int size, Real dt, vec3r windForce, VecArray<Real, MT>& density, VecArray<Real, MT>& temperature, VecArray<vec3r, MT>& velocity)
{
    FILL_CALL_GPU_DEVICE_CODE(addWind, dt, windForce, density.m_data, temperature.m_data, velocity.m_data);
}

// Explicit template instantiation for GPU memory type
template void callAddWind<MemType::GPU>(int, Real, vec3r, VecArray<Real, MemType::GPU>&, VecArray<Real, MemType::GPU>&, VecArray<vec3r, MemType::GPU>&);
//////////////////////////////////////////////////////////
/**
 * Compute the divergence of the velocity field at the specified index.
 *
 * @tparam MT The memory type of the arrays.
 * @param index The index of the element to compute the divergence for.
 * @param divergence Pointer to the array to store the divergence.
 * @param velocity Pointer to the array storing the velocity field.
 */
template<MemType MT>
__device__ void _k_computeDivergence(int index, Real* __restrict__ divergence, vec3r* __restrict__ velocity)
{
    // Get the grid index based on the global index
    int3  gridIdx = _d_getGridIdx(index);
    // Check if the grid index is valid
    if (!_d_isValid(gridIdx))
    {
        return;
    }

    // Compute the divergence using central differencing
    divergence[index] = (velocity[_d_getIdx(make_int3(gridIdx.x + 1, gridIdx.y, gridIdx.z))].x - velocity[_d_getIdx(make_int3(gridIdx.x - 1, gridIdx.y, gridIdx.z))].x
        + velocity[_d_getIdx(make_int3(gridIdx.x, gridIdx.y + 1, gridIdx.z))].y - velocity[_d_getIdx(make_int3(gridIdx.x, gridIdx.y - 1, gridIdx.z))].y
        + velocity[_d_getIdx(make_int3(gridIdx.x, gridIdx.y, gridIdx.z + 1))].z - velocity[_d_getIdx(make_int3(gridIdx.x, gridIdx.y, gridIdx.z - 1))].z) * -0.5f;
}
DECLARE_KERNEL_TEMP(computeDivergence, Real* __restrict__ divergence, vec3r* __restrict__ velocity);

__global__ void _g_computeDivergence(int size, Real* __restrict__ divergence, vec3r* __restrict__ velocity)
{
    // Check if the current thread index is valid
    IF_IDX_VALID(size)
        // Call the device function to compute the divergence
        _k_computeDivergence<MemType::GPU>(i, divergence, velocity);
}

template<MemType MT>
void callComputeDivergence(int size, VecArray<Real, MT>& divergence, VecArray<vec3r, MT>& velocity)
{
    // Call the GPU device code to compute the divergence
    FILL_CALL_GPU_DEVICE_CODE(computeDivergence, divergence.m_data, velocity.m_data);
}
template void callComputeDivergence<MemType::GPU>(int, VecArray<Real, MemType::GPU>&, VecArray<vec3r, MemType::GPU>&);
///////////////////////////////////////////////////
/**
 * Compute pressure for a given index.
 *
 * @param index - the index of the element
 * @param divergence - the array of divergence values
 * @param tempPressure - the temporary pressure array
 * @param pressure - the array of pressure values
 */
template<MemType MT>
__device__ void _k_computePressure(int index, Real* __restrict__ divergence, Real* __restrict__ tempPressure, Real* __restrict__ pressure)
{
    // Get the grid index for the current thread
    int3  gridIdx = _d_getGridIdx(index);
    // Check if the grid index is valid
    if (!_d_isValid(gridIdx))
    {
        // Skip invalid grid indices
        return;
    }
    // Calculate the temporary pressure value as the average of the surrounding pressures
    tempPressure[index] = (divergence[index]+ pressure[_d_getIdx(make_int3(gridIdx.x + 1, gridIdx.y, gridIdx.z))] + pressure[_d_getIdx(make_int3(gridIdx.x - 1, gridIdx.y, gridIdx.z))]+ pressure[_d_getIdx(make_int3(gridIdx.x, gridIdx.y + 1, gridIdx.z))] + pressure[_d_getIdx(make_int3(gridIdx.x, gridIdx.y - 1, gridIdx.z))]+ pressure[_d_getIdx(make_int3(gridIdx.x, gridIdx.y, gridIdx.z + 1))] + pressure[_d_getIdx(make_int3(gridIdx.x, gridIdx.y, gridIdx.z - 1))]) / 6.f;
}
// Declaration of the computePressure kernel template
DECLARE_KERNEL_TEMP(computePressure, Real* __restrict__ divergence, Real* __restrict__ tempPressure, Real* __restrict__ pressure);

// Definition of the computePressure global kernel function
__global__ void _g_computePressure(int size, Real* __restrict__ divergence, Real* __restrict__ tempPressure, Real* __restrict__ pressure)
{
    // Check if the current thread index is valid
    IF_IDX_VALID(size)
        // Call the computePressure device function with the appropriate template argument and parameters
        _k_computePressure<MemType::GPU>(i, divergence, tempPressure, pressure);
}

template<MemType MT>
void callComputePressure(int size, VecArray<Real, MT>& divergence, VecArray<Real, MT>& tempPressure, VecArray<Real, MT>& pressure)
{
    FILL_CALL_GPU_DEVICE_CODE(computePressure, divergence.m_data, tempPressure.m_data, pressure.m_data);
}
template void callComputePressure<MemType::GPU>(int, VecArray<Real, MemType::GPU>&, VecArray<Real, MemType::GPU>&, VecArray<Real, MemType::GPU>&);

//////////////////////////////////////////////////
template<MemType MT>
/**
 * Project the velocity field based on the pressure field.
 *
 * @param index - the index of the element
 * @param velocity - the array of velocity vectors
 * @param pressure - the array of pressure values
 */
__device__ void _k_project(int index, vec3r* __restrict__ velocity, Real* __restrict__ pressure)
{
    int3  gridIdx = _d_getGridIdx(index);
    if (!_d_isValid(gridIdx))
    {
        // Skip invalid grid indices
        return;
    }
    // Update the velocity vector at the current index based on the pressure gradient
    velocity[index] -= make_vec3r(0.5f * (pressure[_d_getIdx(make_int3(gridIdx.x + 1, gridIdx.y, gridIdx.z))] - pressure[_d_getIdx(make_int3(gridIdx.x - 1, gridIdx.y, gridIdx.z))]),0.5f * (pressure[_d_getIdx(make_int3(gridIdx.x, gridIdx.y + 1, gridIdx.z))] - pressure[_d_getIdx(make_int3(gridIdx.x, gridIdx.y - 1, gridIdx.z))]),0.5f * (pressure[_d_getIdx(make_int3(gridIdx.x, gridIdx.y, gridIdx.z + 1))] - pressure[_d_getIdx(make_int3(gridIdx.x, gridIdx.y, gridIdx.z - 1))]));
}

// Declaration of the project kernel template
DECLARE_KERNEL_TEMP(project, vec3r* __restrict__ velocity, Real* __restrict__ pressure);

// Definition of the project global kernel function
__global__ void _g_project(int size, vec3r* __restrict__ velocity, Real* __restrict__ pressure)
{
    // Check if the current thread index is valid
    IF_IDX_VALID(size)
        // Call the project device function with the appropriate template argument and parameters
        _k_project<MemType::GPU>(i, velocity, pressure);
}

template<MemType MT>
/**
 * Call the project kernel function on the GPU.
 *
 * @param size - the size of the arrays
 * @param velocity - the array of velocity vectors
 * @param pressure - the array of pressure values
 */
void callProject(int size, VecArray<vec3r, MT>& velocity, VecArray<Real, MT>& pressure)
{
    FILL_CALL_GPU_DEVICE_CODE(project, velocity.m_data, pressure.m_data);
}

// Explicit instantiation of the callProject function for GPU memory type
template void callProject<MemType::GPU>(int, VecArray<vec3r, MemType::GPU>&, VecArray<Real, MemType::GPU>&);
//////////////////////////////////////////////////
/**
 * Computes the vorticity at the given index using the velocity field.
 *
 * @tparam MT The memory type.
 * @param index The index of the cell.
 * @param vorticity The array to store the computed vorticity.
 * @param velocity The velocity field array.
 */
template<MemType MT>
__device__ void _k_computeVorticity(int index, vec3r* __restrict__ vorticity, vec3r* __restrict__ velocity)
{
    int3  gridIdx = _d_getGridIdx(index);
    if (!_d_isValid(gridIdx))
    {
        return;
    }

    // Compute neighboring indices
    int3 idxL = make_int3(max(0, gridIdx.x - 1), gridIdx.y, gridIdx.z);
    int3 idxR = make_int3(min(gridParams.gridSize.x - 1, gridIdx.x + 1), gridIdx.y, gridIdx.z);

    int3 idxB = make_int3(gridIdx.x, max(0, gridIdx.y - 1), gridIdx.z);
    int3 idxT = make_int3(gridIdx.x, min(gridParams.gridSize.y - 1, gridIdx.y + 1), gridIdx.z);

    int3 idxD = make_int3(gridIdx.x, gridIdx.y, max(0, gridIdx.z - 1));
    int3 idxU = make_int3(gridIdx.x, gridIdx.y, min(gridParams.gridSize.z - 1, gridIdx.z + 1));

    // Compute velocity values at neighboring indices
    vec3r L = velocity[_d_getIdx(idxL)];
    vec3r R = velocity[_d_getIdx(idxR)];

    vec3r B = velocity[_d_getIdx(idxB)];
    vec3r T = velocity[_d_getIdx(idxT)];

    vec3r D = velocity[_d_getIdx(idxD)];
    vec3r U = velocity[_d_getIdx(idxU)];

    // Compute vorticity using the velocity values
    vorticity[index] = 0.5f * make_vec3r(
        ((T.z - B.z) - (U.y - D.y)),
        ((U.x - D.x) - (R.z - L.z)),
        ((R.y - L.y) - (T.x - B.x)));
}
// Declaration of the computeVorticity kernel function
DECLARE_KERNEL_TEMP(computeVorticity, vec3r* __restrict__ vorticity, vec3r* __restrict__ velocity);

// Definition of the _g_computeVorticity global function
__global__ void _g_computeVorticity(int size, vec3r* __restrict__ vorticity, vec3r* __restrict__ velocity)
{
    // Check if the current thread index is valid
    IF_IDX_VALID(size)
        _k_computeVorticity<MemType::GPU>(i, vorticity, velocity);
}

// Template function to call the computeVorticity kernel function
template<MemType MT>
void callComputeVorticity(int size, VecArray<vec3r, MT>& vorticity, VecArray<vec3r, MT>& velocity)
{
    FILL_CALL_GPU_DEVICE_CODE(computeVorticity, vorticity.m_data, velocity.m_data);
}

// Explicit instantiation of the callComputeVorticity template function for GPU memory type
template void callComputeVorticity<MemType::GPU>(int, VecArray<vec3r, MemType::GPU>&, VecArray<vec3r, MemType::GPU>&);
//////////////////////////////////////////////////
// Function to confine vorticity
/**
 * Confines the vorticity at the given index using the specified time step and velocity and vorticity arrays.
 *
 * @tparam MT The memory type.
 * @param index The index of the vorticity to confine.
 * @param dt The time step.
 * @param velocity The array of velocities.
 * @param vorticity The array of vorticities.
 */
template<MemType MT>
__device__ void _k_confineVorticity(int index, Real dt, vec3r* __restrict__ velocity, vec3r* __restrict__ vorticity)
{
    // Get the grid index for the current element
    int3  gridIdx = _d_getGridIdx(index);

    // Check if the grid index is valid
    if (!_d_isValid(gridIdx))
    {
        return;  // Skip invalid grid indices
    }

    // Get the neighboring grid indices in x direction
    int3 idxL = make_int3(max(0, gridIdx.x - 1), gridIdx.y, gridIdx.z);
    int3 idxR = make_int3(min(gridParams.gridSize.x - 1, gridIdx.x + 1), gridIdx.y, gridIdx.z);

    // Get the neighboring grid indices in y direction
    int3 idxB = make_int3(gridIdx.x, max(0, gridIdx.y - 1), gridIdx.z);
    int3 idxT = make_int3(gridIdx.x, min(gridParams.gridSize.y - 1, gridIdx.y + 1), gridIdx.z);

    // Get the neighboring grid indices in z direction
    int3 idxD = make_int3(gridIdx.x, gridIdx.y, max(0, gridIdx.z - 1));
    int3 idxU = make_int3(gridIdx.x, gridIdx.y, min(gridParams.gridSize.z - 1, gridIdx.z + 1));

    // Get the vorticity values at the neighboring grid points
    vec3r omega = vorticity[index];
    float omegaL = length(vorticity[_d_getIdx(idxL)]);
    float omegaR = length(vorticity[_d_getIdx(idxR)]);
    float omegaB = length(vorticity[_d_getIdx(idxB)]);
    float omegaT = length(vorticity[_d_getIdx(idxT)]);
    float omegaD = length(vorticity[_d_getIdx(idxD)]);
    float omegaU = length(vorticity[_d_getIdx(idxU)]);

    // Calculate the gradient of vorticity
    vec3r omegaGrad = 0.5f * make_vec3r(omegaR - omegaL, omegaT - omegaB, omegaU - omegaD);

    // Normalize the gradient
    omegaGrad = normalize(omegaGrad + make_vec3r(0.00001f, 0.00001f, 0.00001f));

    // Calculate the change in velocity due to vorticity confinement
    vec3r deltaVel = dt * gridParams.kvorticity * cross(omegaGrad, omega);

    // Update the velocity
    velocity[index] += make_vec3r(deltaVel.x, deltaVel.y, deltaVel.z);
}

// Declare the kernel for confining vorticity
DECLARE_KERNEL_TEMP(confineVorticity, Real dt, vec3r* __restrict__ velocity, vec3r* __restrict__ vorticity);

// Global function to confine vorticity
__global__ void _g_confineVorticity(int size, Real dt, vec3r* __restrict__ velocity, vec3r* __restrict__ vorticity)
{
    IF_IDX_VALID(size)
        _k_confineVorticity<MemType::GPU>(i, dt, velocity, vorticity);
}

// Template function to call confineVorticity on GPU
template<MemType MT>
void callConfineVorticity(int size, Real dt, VecArray<vec3r, MT>& velocity, VecArray<vec3r, MT>& vorticity)
{
    FILL_CALL_GPU_DEVICE_CODE(confineVorticity, dt, velocity.m_data, vorticity.m_data);
}

// Explicit instantiation of callConfineVorticity for GPU
template void callConfineVorticity<MemType::GPU>(int, Real, VecArray<vec3r, MemType::GPU>&, VecArray<vec3r, MemType::GPU>&);
PHYS_NAMESPACE_END