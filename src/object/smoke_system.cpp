#include "smoke_system.h"
#include "common/timer.h"
#include <algorithm>

using namespace physeng;

// Constructor
SmokeSystem::SmokeSystem(uint3 gridSize, Real cellLength, vec3r worldMin, vec3r worldMax) :
	m_hd(),
	m_ht(),
	m_hv(),
	m_hc(),
	m_c()
{
	// Constants
	const float kPi = 3.141592654f;

	m_maxlife = INFINITY;

	// Initialization flags
	m_isInited = false;
	m_lifetime = 0.0;

	// Default parameters
	m_solverIterations = 6;
	m_subSteps = 1;

	// Grid info
	m_gridSize = gridSize;
	m_cellLength = cellLength;
	m_numCells = m_gridSize.x * m_gridSize.y * m_gridSize.z;

	// Simulation world
	m_params.worldOrigin = worldMin;
	m_params.worldMin = worldMin;
	m_params.worldMax = worldMax;

	// Grid parameters
	m_params.gridSize = m_gridSize;
	m_params.numCells = m_numCells;
	m_params.cellLength = cellLength;
	m_params.invCellLength = 1.0 / cellLength;
	m_params.gridHashMultiplier = make_uint3(1, m_gridSize.x, m_gridSize.x * m_gridSize.y);

	// Physics parameters
	m_params.gravity = make_vec3r(0.0f, -9.8f, 0.0f);
	m_params.kvorticity = 5;
	m_params.kdiffusion = 0.01;
	m_params.bdensity = 0.3;
	m_params.btemperature = 0.2;
	m_params.disappearRate = 0.1;

	// Resize the grid fluid
	gf.resizeGF(gridSize, cellLength);

	// Initialize the system
	_initialize();
}

// Destructor
SmokeSystem::~SmokeSystem(){
	_finalize();
}

/**
 * @brief system initialization
 * 
 */
void SmokeSystem::_initialize(){
	assert(!m_isInited);

	// Allocate host storage for grid properties
	m_hd.resize(m_numCells, false);
	m_hd.fill(m_numCells, 0);
	m_ht.resize(m_numCells, false);
	m_ht.fill(m_numCells, 0);
	m_hv.resize(m_numCells, false);
	m_hv.fill(m_numCells, make_vec3r(0.0f));
	m_hc.resize(m_numCells, false);
	m_hc.fill(m_numCells, make_vec3r(0.9f, 0.9f, 0.9f));
	m_c.resize(m_numCells, false);

	// Set grid parameters and submit to the device
	setGridParameters(&m_params);
	submitToDevice(0, m_numCells);

	// Set initialization flag
	m_isInited = true;
}

// System finalization
void SmokeSystem::_finalize(){
	assert(m_isInited);

	// Release host storage
	m_hd.release();
	m_ht.release();
	m_hv.release();
	m_hc.release();
	m_c.release();
}

/**
 * @brief update the smoke system
 * 
 * @param deltaTime 
 */
void SmokeSystem::update(Real deltaTime) {
	assert(m_isInited);

	// Calculate substep delta time
	Real subDt = deltaTime / m_subSteps;
	m_lifetime += subDt;

	// Source parameters
	int3 source = make_int3(m_gridSize.x / 4, m_gridSize.y / 4, m_gridSize.z / 4);
	int radius2 = 5 * 5;
	Real density = (rand() % 1000) / 1000.0f;

	if (m_lifetime > m_maxlife) {
		callDisappear<MemType::GPU>(m_numCells, deltaTime, gf.getDensityRef());
		return;
	}
	else
	{
		PHY_PROFILE("update");

		if (m_numCells > 0) {
			// Add density to the grid
			callAddDensity<MemType::GPU>(m_numCells,source,radius2,density,gf.getDensityRef());
			cudaDeviceSynchronize();

			// Add temperature to the grid
			callAddDensity<MemType::GPU>(m_numCells,source,radius2,10,gf.getTemperatureRef());
			cudaDeviceSynchronize();

			// Diffusion step for density
			callDiffuse<MemType::GPU>(m_numCells,deltaTime,gf.getTempDensityRef(),gf.getDensityRef());
			cudaDeviceSynchronize();
			gf.getDensityRef().swap(gf.getTempDensityRef());

			// Diffusion step for temperature
			callDiffuse<MemType::GPU>(m_numCells,deltaTime,gf.getTempTemperatureRef(),gf.getTemperatureRef());
			cudaDeviceSynchronize();
			gf.getTemperatureRef().swap(gf.getTempTemperatureRef());

			// Substeps
			for (uint s = 0; s < m_subSteps; s++) {
				// Advect density
				callAdvectProperty<MemType::GPU>(
					m_numCells,
					subDt,
					gf.getTempDensityRef(),
					gf.getDensityRef(),
					gf.getVelocityRef());
				cudaDeviceSynchronize();
				gf.getDensityRef().swap(gf.getTempDensityRef());

				// Advect velocity
				callAdvectVelocity<MemType::GPU>(
					m_numCells,
					subDt,
					gf.getTempVelocityRef(),
					gf.getVelocityRef());
				cudaDeviceSynchronize();
				gf.getVelocityRef().swap(gf.getTempVelocityRef());

				// Add buoyancy
				callAddBuoyancy<MemType::GPU>(
					m_numCells,
					subDt,
					gf.getDensityRef(),
					gf.getTemperatureRef(),
					gf.getVelocityRef());
				cudaDeviceSynchronize();

				// Add wind
				if(length(m_wind) > 1e-2){
					callAddWind<MemType::GPU>(
						m_numCells,
						subDt,
						m_wind,
						gf.getDensityRef(),
						gf.getTemperatureRef(),
						gf.getVelocityRef());
					cudaDeviceSynchronize();
				}


				// Compute vorticity
				callComputeVorticity<MemType::GPU>(
					m_numCells,
					gf.getVorticityRef(),
					gf.getVelocityRef());
				cudaDeviceSynchronize();

				// Confine vorticity
				callConfineVorticity<MemType::GPU>(
					m_numCells,
					subDt,
					gf.getVorticityRef(),
					gf.getVelocityRef());
				cudaDeviceSynchronize();

				// Compute divergence
				callComputeDivergence<MemType::GPU>(
					m_numCells,
					gf.getDivergenceRef(),
					gf.getVelocityRef());
				cudaDeviceSynchronize();

				// Solver iterations
				for (uint i = 0; i < m_solverIterations; i++) {
					// Compute pressure
					callComputePressure<MemType::GPU>(
						m_numCells,
						gf.getDivergenceRef(),
						gf.getTempPressureRef(),
						gf.getPressureRef());
					cudaDeviceSynchronize();
					gf.getPressureRef().swap(gf.getTempPressureRef());
				}

				// Project the velocity
				callProject<MemType::GPU>(
					m_numCells,
					gf.getVelocityRef(),
					gf.getPressureRef());
				cudaDeviceSynchronize();
			}
		}
	}
}

// Submit host arrays to the device
void SmokeSystem::submitToDevice(uint start, uint count){
	copyArray<Real, MemType::GPU, MemType::CPU>(&gf.getDensityRef().m_data, &m_hd.m_data, start, count);
	copyArray<Real, MemType::GPU, MemType::CPU>(&gf.getTemperatureRef().m_data, &m_ht.m_data, start, count);
	copyArray<vec3r, MemType::GPU, MemType::CPU>(&gf.getVelocityRef().m_data, &m_hv.m_data, start, count);
	copyArray<vec3r, MemType::GPU, MemType::CPU>(&m_c.m_data, &m_hc.m_data, start, count);
}

// Clear the system
void SmokeSystem::clear(){
}

