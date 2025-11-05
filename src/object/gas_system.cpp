#include "gas_system.h"
#include "common/timer.h"
#include "common/logger.h"
#include <algorithm>

#define ALMOST_EQUAL(a, b) ((fabs(a-b)<0.00001f)?true:false)

extern StopWatchInterface* timer;

using namespace physeng;

// Constructor
GasSystem::GasSystem(uint3 gridSize, Real cellLength, vec3r worldMin, vec3r worldMax) :
	m_hd(),
	m_hi(),
	m_texture()
{
	// Constants
	const float kPi = 3.141592654f;

	m_maxlife = INFINITY;
	m_windStrength = 0;

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

	m_ifGenerate = true;

	// Physics parameters
	m_params.gravity = make_vec3r(0.0f, -9.8f, 0.0f);
	m_params.kvorticity = 0.0001f;
	m_params.kdiffusion = 0.000000f;
	m_params.kbuoyancy = 4.0f;
	m_params.vc_eps = 5.0f;

	m_decreaseDensity = 0.001f;


	// Resize the grid fluid
	gg.resize(gridSize, cellLength);

	// Initialize the system
	_initialize();

	m_ambient = 100;
	m_decay = 0.06f;
	m_lightDir = make_vec3r(0, 0, -1);
	m_color = make_vec3r(1.0f, 1.0f, 1.0f);
	m_alpha = 0.5;
}

// Destructor
GasSystem::~GasSystem(){
	_finalize();
}

/**
 * @brief system initialization
 * 
 */
void GasSystem::_initialize(){
	assert(!m_isInited);

	// Allocate host storage for grid properties
	m_hd.resize(m_numCells, false);
	m_hd.fill(m_numCells, 0);
	m_hi.resize(m_numCells, false);
	m_rflag.resize(m_numCells, false);
	m_rflag.fill(m_numCells, false);
	m_texture.resize(m_numCells * 4, false);
	m_texture.fill(m_numCells * 4, 0);

	// Set grid parameters and submit to the device
	setGridParameters(&m_params);
	submitToDevice(0, m_numCells);

	// Set initialization flag
	m_isInited = true;
}

// System finalization
void GasSystem::_finalize(){
	assert(m_isInited);

	// Release host storage
	m_hd.release();
	m_hi.release();
	m_rflag.release();
	m_texture.release();
}

void GasSystem::generateRayTemplate() {
	rayTemplate.resize(4096);
	rayTemplate[0] = make_int3(0);
	float fx = 0.0f, fy = 0.0f, fz = 0.0f;
	int x = 0, y = 0, z = 0;
	float lx = m_lightDir.x + 0.000001f, ly = m_lightDir.y + 0.000001f, lz = m_lightDir.z + 0.000001f;
	int xinc = (lx > 0) ? 1 : -1;
	int yinc = (ly > 0) ? 1 : -1;
	int zinc = (lz > 0) ? 1 : -1;
	float tx, ty, tz;
	int i = 1;
	int len = 0;
	int edgeLen = m_params.gridSize.x;
	int maxlen = 3 * edgeLen * edgeLen;
	while (len <= maxlen)
	{
		// fx + t*lx = (x+1)   ->   t = (x+1-fx)/lx
		tx = (x + xinc - fx) / lx;
		ty = (y + yinc - fy) / ly;
		tz = (z + zinc - fz) / lz;

		if ((tx <= ty) && (tx <= tz)) {
			rayTemplate[i].x = rayTemplate[i - 1].x + xinc;
			x = +xinc;
			fx = x;

			if (ALMOST_EQUAL(ty, tx)) {
				rayTemplate[i].y = rayTemplate[i - 1].y + yinc;
				y += yinc;
				fy = y;
			}
			else {
				rayTemplate[i].y = rayTemplate[i - 1].y;
				fy += tx * ly;
			}

			if (ALMOST_EQUAL(tz, tx)) {
				rayTemplate[i].z = rayTemplate[i - 1].z + zinc;
				z += zinc;
				fz = z;
			}
			else {
				rayTemplate[i].z = rayTemplate[i - 1].z;
				fz += tx * lz;
			}
		}
		else if ((ty < tx) && (ty <= tz)) {
			rayTemplate[i].x = rayTemplate[i - 1].x;
			fx += ty * lx;

			rayTemplate[i].y = rayTemplate[i - 1].y + yinc;
			y += yinc;
			fy = y;

			if (ALMOST_EQUAL(tz, ty)) {
				rayTemplate[i].z = rayTemplate[i - 1].z + zinc;
				z += zinc;
				fz = z;
			}
			else {
				rayTemplate[i].z = rayTemplate[i - 1].z;
				fz += ty * lz;
			}
		}
		else {
			assert((tz < tx) && (tz < ty));
			rayTemplate[i].x = rayTemplate[i - 1].x;
			fx += tz * lx;
			rayTemplate[i].y = rayTemplate[i - 1].y;
			fy += tz * ly;
			rayTemplate[i].z = rayTemplate[i - 1].z + zinc;
			z += zinc;
			fz = z;
		}

		len = rayTemplate[i].x * rayTemplate[i].x
			+ rayTemplate[i].y * rayTemplate[i].y
			+ rayTemplate[i].z * rayTemplate[i].z;
		i++;
	}
}

/**
 * @brief update the smoke system
 * 
 * @param deltaTime 
 */
void GasSystem::update(Real deltaTime) {
	assert(m_isInited);

	if (m_windStrength < 0) {
		printf("��������δ���û����ô���\n");
		logger.Log(LogTypeKD::Error, "��������δ���û����ô���");
		exit(0);
	}

	// Calculate substep delta time
	Real subDt = deltaTime / m_subSteps;
	m_lifetime += subDt;

	// Source parameters
	int3 source = make_int3(5, m_gridSize.y / 4, m_gridSize.z / 2);
	

	{
		PHY_PROFILE("update");

		if (m_numCells > 0) {
			// Generate Smoke
			if (m_ifGenerate && sdkGetTimerValue(&timer) / 1000.f <= m_maxlife) {
				for (auto gs : gasSources) {
					Real density = (rand() % 1000) / 1000.0f;
					GenerateSmoke(m_numCells, gs.source, gs.radius, density * gs.density, gg.getDensityRef(), gs.velocity, gg.getVelocityRef());
				}
			}

			AddBuoyancy(m_numCells, deltaTime, make_vec3r(0.0f, 1.0f, 0.0f), gg.getDensityRef(), gg.getVelocityRef());

			if (m_windStrength > 0)
				AddWind(m_numCells, deltaTime, m_windDirection, m_windStrength, gg.getDensityRef(), gg.getVelocityRef());

			LockRigid(m_numCells, gg.getDensityRef(), gg.getVelocityRef(), gg.getRigidFlagRef());

			VorticityConfinement(m_numCells, deltaTime, gg.getVelocityRef(), gg.getTempVelocityRef(), gg.getRigidFlagRef());

			DiffuseVelocity(m_numCells, deltaTime, gg.getVelocityRef(), gg.getTempVelocityRef(), gg.getRigidFlagRef());

			Project(m_numCells, gg.getVelocityRef(), gg.getDivergenceRef(), gg.getPressureRef(), gg.getRigidFlagRef());

			AdvectVelocity(m_numCells, deltaTime, gg.getVelocityRef(), gg.getTempVelocityRef(), gg.getRigidFlagRef());

			Project(m_numCells, gg.getVelocityRef(), gg.getDivergenceRef(), gg.getPressureRef(), gg.getRigidFlagRef());

			DiffuseDensity(m_numCells, deltaTime, gg.getDensityRef(), gg.getTempDensityRef(), gg.getRigidFlagRef());

			AdvectDensity(m_numCells, deltaTime, gg.getDensityRef(), gg.getTempDensityRef(), gg.getVelocityRef(), gg.getRigidFlagRef());

			DecreaseDensity(m_numCells, m_decreaseDensity, gg.getDensityRef());
		}
	}

	copyArray<Real, MemType::CPU, MemType::GPU>(&m_hd.m_data, &gg.getDensityRef().m_data, 0, m_numCells);
}

// Submit host arrays to the device
void GasSystem::submitToDevice(uint start, uint count){
	copyArray<Real, MemType::GPU, MemType::CPU>(&gg.getDensityRef().m_data, &m_hd.m_data, start, count);
}

// Clear the system
void GasSystem::clear(){
}

void GasSystem::fillTexture() {
	castLight();

	for (int i = 0; i < m_params.gridSize.x; i++) {
		for (int j = 0; j < m_params.gridSize.y; j++) {
			for (int k = 0; k < m_params.gridSize.z; k++) {
				//unsigned char c = 200;
				int index = i + j * m_params.gridHashMultiplier.y + k * m_params.gridHashMultiplier.z;
				unsigned char c = m_hi[index];
				unsigned char a = m_alpha * ((m_hd[index] > 0.1f) ? 255 : (unsigned char)(m_hd[index] * 2550.0f));
				m_texture[index * 4 + 0] = (unsigned char)c * m_color.x;
				m_texture[index * 4 + 1] = (unsigned char)c * m_color.y;
				m_texture[index * 4 + 2] = (unsigned char)c * m_color.z;
				m_texture[index * 4 + 3] = a;
			}
		}
	}
}

void GasSystem::castLight() {
	int i, j;
	int sx = (m_lightDir.x > 0) ? 0 : m_params.gridSize.x - 1;
	int sy = (m_lightDir.y > 0) ? 0 : m_params.gridSize.y - 1;
	int sz = (m_lightDir.z > 0) ? 0 : m_params.gridSize.z - 1;

	for (i = 0; i < m_params.gridSize.y; i++)
		for (j = 0; j < m_params.gridSize.z; j++) {
			if (!ALMOST_EQUAL(m_lightDir.x, 0))
				lightRay(sx, i, j, m_params.gridSize.x, 1.0f / (m_params.gridSize.x * m_decay));
			if (!ALMOST_EQUAL(m_lightDir.y, 0))
				lightRay(i, sy, j, m_params.gridSize.y, 1.0f / (m_params.gridSize.y * m_decay));
			if (!ALMOST_EQUAL(m_lightDir.z, 0))
				lightRay(i, j, sz, m_params.gridSize.z, 1.0f / (m_params.gridSize.z * m_decay));
		}

	for (i = 0; i < m_params.gridSize.x; i++)
		for (j = 0; j < m_params.gridSize.z; j++) {
			if (!ALMOST_EQUAL(m_lightDir.y, 0))
				lightRay(i, sy, j, m_params.gridSize.y, 1.0f / (m_params.gridSize.y * m_decay));
		}

	for (i = 0; i < m_params.gridSize.x; i++)
		for (j = 0; j < m_params.gridSize.y; j++) {
			if (!ALMOST_EQUAL(m_lightDir.z, 0))
				lightRay(i, j, sz, m_params.gridSize.z, 1.0f / (m_params.gridSize.z * m_decay));
		}
}

void GasSystem::lightRay(int x, int y, int z, int n, Real decay) {
	int xx = x, yy = y, zz = z, i = 0;
	int offset;

	int l = 200;
	float d;

	do {
		offset = ((xx * n) + yy) * n + zz;//FIXME
		if (m_hi[offset] > 0)
			m_hi[offset] = (unsigned char)((m_hi[offset] + l) * 0.5f);
		else
			m_hi[offset] = (unsigned char)l;
		d = m_hd[offset] * 255.0f;
		if (l > m_ambient) {
			l -= d * decay;
			if (l < m_ambient)
				l = m_ambient;
		}

		i++;
		xx = x + rayTemplate[i].x;
		yy = y + rayTemplate[i].y;
		zz = z + rayTemplate[i].z;
	} while ((xx >= 0) && (xx < n) && (yy >= 0) && (yy < n) && (zz >= 0) && (zz < n));
}

void GasSystem::calcRenderData() {
	fillTexture();
}

bool GasSystem::addGasSource(vec3r source, Real radius, vec3r velocity, Real density) {
	GasSource gs;
	if(source.x < m_params.worldMin.x || source.x > m_params.worldMax.x ||
		source.y < m_params.worldMin.y || source.y > m_params.worldMax.y ||
		source.z < m_params.worldMin.z || source.z > m_params.worldMax.z){
		//printf("ָ��λ�ó������緶Χ��������������Դ������ʧ��\n");
		return false;
	}
		
	gs.source = make_int3((source - m_params.worldMin) * m_params.invCellLength);
	gs.radius = int(radius * m_params.invCellLength) + 1;
	gs.velocity = velocity;
	gs.density = density;
	gasSources.push_back(gs);
	return true;
}

void GasSystem::addBox(vec3r origin, vec3r size) {
	vec3r boxMin = origin - size * 0.5, boxMax = origin + size * 0.5;
	int3 boxStartGrid = make_int3((boxMin - m_params.worldMin) / m_params.cellLength);
	int3 boxEndGrid = make_int3((boxMax - m_params.worldMin) / m_params.cellLength);

	boxStartGrid = max(boxStartGrid, make_int3(0));
	boxEndGrid = min(boxEndGrid, make_int3(m_params.gridSize.x - 1, m_params.gridSize.y - 1, m_params.gridSize.z - 1));

	for(int i = boxStartGrid.x;i<boxEndGrid.x;i++)
		for(int j = boxStartGrid.y;j<boxEndGrid.y;j++)
			for(int k = boxStartGrid.z;k<boxEndGrid.z;k++)
				m_rflag[i + j * m_params.gridHashMultiplier.y + k * m_params.gridHashMultiplier.z] = true;
	copyArray<bool, MemType::GPU, MemType::CPU>(&gg.getRigidFlagRef().m_data, &m_rflag.m_data, 0, m_numCells);
}

Real GasSystem::getAverageDensity(){

	Real sum = 0;
	int validCellNum = 0;
	for(int i = 0;i<m_numCells;i++){
		if (m_hd[i] > 0) {
			sum += m_hd[i];
			validCellNum++;
		}
	}
	if(validCellNum == 0) validCellNum = 1;
	return sum / validCellNum;
}