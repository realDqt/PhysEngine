#pragma once

#include <mutex>

#include "common/array.h"

PHYS_NAMESPACE_BEGIN
struct GridSimParams
{
    //// simulation world
    vec3r worldOrigin; //!< origin
    vec3r worldMin;    //!< left lower corner of the world
    vec3r worldMax;    //!< right upper corner of the world

    //// particle grid
    uint3 gridSize; //!< grid size
    uint numCells;       //!< #cells
    Real cellLength;    //!< length of each cell
    Real invCellLength; //!< inverse of length of each cell
    uint3 gridHashMultiplier;

    //// physics parameter
    vec3r gravity;        //!< gravity
    Real kbuoyancy;       //!< buoyancy coefficient
    Real kvorticity;       //!< vorticity coefficient
    Real vc_eps;         //!< epsilon of vorticity confinement
    Real kdiffusion;
    Real bdensity;
    Real btemperature;
    Real disappearRate;
};

extern __constant__ GridSimParams gridParams;
extern GridSimParams hGridParams;

inline __host__ __device__ Real _d_getDistance(int3 p)
{
    return (Real)sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

inline __host__ __device__ int _d_getDistance2(int3 p)
{
    return p.x * p.x + p.y * p.y + p.z * p.z;
}

inline __host__ __device__ int3 _d_getGridPos(vec3r p)
{
    return make_int3(
        floorf(p.x),
        floorf(p.y),
        floorf(p.z));
}

inline __host__ __device__ bool _d_isValid(int3 gridIdx)
{
    return  (int)gridIdx.x != 0 && (int)gridIdx.x != gridParams.gridSize.x - 1
        && (int)gridIdx.y != 0 && (int)gridIdx.y != gridParams.gridSize.y - 1
        && (int)gridIdx.z != 0 && (int)gridIdx.z != gridParams.gridSize.z - 1;
}

inline __host__ __device__ int _d_getIdx(int3 gridIdx)
{
    return  gridIdx.x + gridIdx.y * gridParams.gridHashMultiplier.y
        + gridIdx.z * gridParams.gridHashMultiplier.z;
}

inline __host__  int _d_getIdx(int x, int y, int z)
{
    return  x + y * gridParams.gridHashMultiplier.y + z * gridParams.gridHashMultiplier.z;
}

inline __host__ __device__ int3 _d_getGridIdx(int idx)
{
    int gridZ = idx / gridParams.gridHashMultiplier.z;
    int gridY = idx % gridParams.gridHashMultiplier.z / gridParams.gridHashMultiplier.y;
    int gridX = idx % gridParams.gridHashMultiplier.z % gridParams.gridHashMultiplier.y;
    return make_int3(gridX, gridY, gridZ);

}
inline __host__ __device__ int3 _d_getGridIdx(int idx, uint3 gridHashMultiplier)
{
    int gridZ = idx / gridHashMultiplier.z;
    int gridY = idx % gridHashMultiplier.z / gridHashMultiplier.y;
    int gridX = idx % gridHashMultiplier.z % gridHashMultiplier.y;
    return make_int3(gridX, gridY, gridZ);

}
inline __host__ __device__ Real _d_sampleValue(const Real* data, const vec3r pos)
{
    static constexpr int dx[8] = { 0,1,0,1,0,1,0,1 };
    static constexpr int dy[8] = { 0,0,1,1,0,0,1,1 };
    static constexpr int dz[8] = { 0,0,0,0,1,1,1,1 };
    int3 coord = _d_getGridPos(pos);
    vec3r frac = pos - make_vec3r(coord);
    Real w[3][2] = { {1.0 - frac.x,frac.x},{1.0 - frac.y,frac.y} ,{1.0 - frac.z,frac.z} };
    Real intp_value = 0;

    for (int s = 0; s < 8; s++) {
        int d0 = dx[s], d1 = dy[s], d2 = dz[s];
        int idx = _d_getIdx(make_int3(coord.x + d0, coord.y + d1, coord.z + d2));
        intp_value += w[0][d0] * w[1][d1] * w[2][d2] * data[idx];
    }
    return intp_value;
}

inline __host__ __device__ vec3r _d_sampleVector(const vec3r* data, const vec3r pos)
{
    static constexpr int dx[8] = { 0,1,0,1,0,1,0,1 };
    static constexpr int dy[8] = { 0,0,1,1,0,0,1,1 };
    static constexpr int dz[8] = { 0,0,0,0,1,1,1,1 };

    int3 coord = _d_getGridPos(pos);
    vec3r frac = pos - make_vec3r(coord);
    Real w[3][2] = { {1.0 - frac.x,frac.x},{1.0 - frac.y,frac.y} ,{1.0 - frac.z,frac.z} };
    vec3r intp_value = make_vec3r(0, 0, 0);

    for (int s = 0; s < 8; s++) {
        int d0 = dx[s], d1 = dy[s], d2 = dz[s];
        int idx = _d_getIdx(make_int3(coord.x + d0, coord.y + d1, coord.z + d2));
        intp_value += w[0][d0] * w[1][d1] * w[2][d2] * data[idx];
    }
    return intp_value;
}

template<MemType MT>
void callDisappear(int numCells, Real dt, VecArray<Real, MT>& density);

template<MemType MT>
void callDiffuse(int numCells, Real dt, VecArray<Real, MT>& tempDensity, VecArray<Real, MT>& density);


template<MemType MT>
void callAdvectProperty(int numCells, Real dt, VecArray<Real, MT>& tempDensity, VecArray<Real, MT>& density, VecArray<vec3r, MT>& velocity);

template<MemType MT>
void callAdvectVelocity(int numCells, Real dt, VecArray<vec3r, MT>& tempVelocity, VecArray<vec3r, MT>& velocity);

template<MemType MT>
void callAddDensity(int numCells, int3 source, int radius, Real tempDensity, VecArray<Real, MT>& density);

template<MemType MT>
void callAddBuoyancy(int numCells, Real dt, VecArray<Real, MT>& density, VecArray<Real, MT>& temperature, VecArray<vec3r, MT>& velocity);

template<MemType MT>
void callAddWind(int numCells, Real dt, vec3r windForce,  VecArray<Real, MT>& density, VecArray<Real, MT>& temperature, VecArray<vec3r, MT>& velocity);

template<MemType MT>
void callComputeVorticity(int numCells, VecArray<vec3r, MT>& vorticity, VecArray<vec3r, MT>& velocity);

template<MemType MT>
void callConfineVorticity(int numCells, Real dt, VecArray<vec3r, MT>& velocity, VecArray<vec3r, MT>& vorticity);

template<MemType MT>
void callComputeDivergence(int numCells, VecArray<Real, MT>& divergence, VecArray<vec3r, MT>& velocity);

template<MemType MT>
void callComputePressure(int numCells, VecArray<Real, MT>& divergence, VecArray<Real, MT>& tempPressure, VecArray<Real, MT>& pressure);

template<MemType MT>
void callProject(int numCells, VecArray<vec3r, MT>& velocity, VecArray<Real, MT>& pressure);

template<MemType MT>
class GridSystem
{
  public:
	DEFINE_MEMBER_GET(Real, m_lengthPerCell, LengthPerCell);
    DEFINE_SOA_SET_GET(Real, MT, m_divergence, Divergence);
	DEFINE_SOA_SET_GET(Real, MT, m_pressure, Pressure);
	DEFINE_SOA_SET_GET(Real, MT, m_tempPressure, TempPressure);
    DEFINE_SOA_SET_GET(Real, MT, m_density, Density);
	DEFINE_SOA_SET_GET(Real, MT, m_tempDensity, TempDensity);
    DEFINE_SOA_SET_GET(Real, MT, m_temperature, Temperature);
    DEFINE_SOA_SET_GET(Real, MT, m_tempTemperature, TempTemperature);
	DEFINE_SOA_SET_GET(vec3r, MT, m_velocity, Velocity);
	DEFINE_SOA_SET_GET(vec3r, MT, m_tempVelocity, TempVelocity);
    DEFINE_SOA_SET_GET(vec3r, MT, m_vorticity, Vorticity);
  public:
    
	GridSystem(uint3 gridSize, Real lengthPerCell){
		m_lengthPerCell = lengthPerCell;
		int size = gridSize.x * gridSize.y * gridSize.z;
		if (size != 0) {
			m_divergence.resize(size);
			m_pressure.resize(size);
			m_tempPressure.resize(size);
			m_density.resize(size);
			m_tempDensity.resize(size);
            m_temperature.resize(size);
            m_tempTemperature.resize(size);
			m_velocity.resize(size);
			m_tempVelocity.resize(size);
			m_vorticity.resize(size);
		}
	}

	virtual ~GridSystem() {
		LOG_OSTREAM_DEBUG<<"release m_density 0x"<<std::hex<<&m_density<<std::dec<<std::endl;
		m_divergence.release();
		m_pressure.release();
		m_tempPressure.release();
		m_density.release();
		m_tempDensity.release();
        m_temperature.release();
        m_tempTemperature.release();
		m_velocity.release();
		m_tempVelocity.release();
		m_vorticity.release();
		LOG_OSTREAM_DEBUG<<"release GridSystem finished"<<std::endl;
	}

};
void setGridParameters(GridSimParams* hostParams);
PHYS_NAMESPACE_END