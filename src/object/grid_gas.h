#pragma once
#include "object/grid_system.h"
#include "common/soa.h"
#include "common/array.h"

PHYS_NAMESPACE_BEGIN

void GenerateSmoke(int size, int3 source, int radius, Real newDensity, VecArray<Real, MemType::GPU>& density, vec3r newVelocity, VecArray<vec3r, MemType::GPU> velocity);
void AddBuoyancy(int size, Real dt, vec3r buoyancyDirection, VecArray<Real, MemType::GPU>& density, VecArray<vec3r, MemType::GPU>& velocity);
void AddWind(int size, Real dt, vec3r windDirection, Real windStrength, VecArray<Real, MemType::GPU>& density, VecArray<vec3r, MemType::GPU>& velocity);
void LockRigid(int size, VecArray<Real, MemType::GPU>& density, VecArray<vec3r, MemType::GPU>& velocity, VecArray<bool, MemType::GPU>& rigidFlag);
void VorticityConfinement(int size, Real dt, VecArray<vec3r, MemType::GPU>& velocity, VecArray<vec3r, MemType::GPU>& curl, VecArray<bool, MemType::GPU>& rigidFlag);
void DiffuseVelocity(int size, Real dt, VecArray<vec3r, MemType::GPU>& velocity, VecArray<vec3r, MemType::GPU>& tempVelocity, VecArray<bool, MemType::GPU>& rigidFlag);
void Project(int size, VecArray<vec3r, MemType::GPU>& velocity, VecArray<Real, MemType::GPU>& divergence, VecArray<Real, MemType::GPU>& pressure, VecArray<bool, MemType::GPU>& rigidFlag);
void AdvectVelocity(int size, Real dt, VecArray<vec3r, MemType::GPU>& velocity, VecArray<vec3r, MemType::GPU>& tempVelocity, VecArray<bool, MemType::GPU>& rigidFlag);
void DiffuseDensity(int size, Real dt, VecArray<Real, MemType::GPU>& density, VecArray<Real, MemType::GPU>& tempDensity, VecArray<bool, MemType::GPU>& rigidFlag);
void AdvectDensity(int size, Real dt, VecArray<Real, MemType::GPU>& density, VecArray<Real, MemType::GPU>& tempDensity, VecArray<vec3r, MemType::GPU>& velocity, VecArray<bool, MemType::GPU>& rigidFlag);
void DecreaseDensity(int size, Real disappearRate, VecArray<Real, MemType::GPU>& density);

template<MemType MT>
class GasGrid
{
  public:
	DEFINE_MEMBER_GET(Real, m_lengthPerCell, LengthPerCell);
    DEFINE_SOA_SET_GET(Real, MT, m_density, Density);
	DEFINE_SOA_SET_GET(Real, MT, m_tempDensity, TempDensity);
	DEFINE_SOA_SET_GET(vec3r, MT, m_velocity, Velocity);
	DEFINE_SOA_SET_GET(vec3r, MT, m_tempVelocity, TempVelocity);
	DEFINE_SOA_SET_GET(Real, MT, m_pressure, Pressure);
	DEFINE_SOA_SET_GET(Real, MT, m_divergence, Divergence);
	DEFINE_SOA_SET_GET(bool, MT, m_rigidFlag, RigidFlag);
  public:
    
	GasGrid(uint3 gridSize, Real lengthPerCell){
		m_lengthPerCell = lengthPerCell;
		int size = gridSize.x * gridSize.y * gridSize.z;
		if (size != 0) {
			m_density.resize(size);
			m_tempDensity.resize(size);
			m_velocity.resize(size);
			m_tempVelocity.resize(size);
			m_pressure.resize(size);
			m_divergence.resize(size);
			m_rigidFlag.resize(size);
		}
	}

	virtual ~GasGrid() {
		LOG_OSTREAM_DEBUG<<"release m_density 0x"<<std::hex<<&m_density<<std::dec<<std::endl;
		m_density.release();
		m_tempDensity.release();
		m_velocity.release();
		m_tempVelocity.release();
		m_pressure.release();
		m_divergence.release();
		m_rigidFlag.release();
		LOG_OSTREAM_DEBUG<<"release GridSystem finished"<<std::endl;
	}

	void resize(uint3 gridSize, Real lengthPerCell) {
        m_lengthPerCell = lengthPerCell;
        int size = gridSize.x * gridSize.y * gridSize.z;
        if (size != 0) {
            m_density.resize(size);
            m_tempDensity.resize(size);
            m_velocity.resize(size);
            m_tempVelocity.resize(size);
			m_pressure.resize(size);
			m_divergence.resize(size);
			m_rigidFlag.resize(size);
        }
    };

};

PHYS_NAMESPACE_END