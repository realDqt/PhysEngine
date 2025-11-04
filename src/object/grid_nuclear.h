#pragma once
#include "object/grid_system.h"
#include "common/soa.h"

PHYS_NAMESPACE_BEGIN
template<MemType MT>
class Nuclear : public GridSystem<MT>{
public:
    using Base=GridSystem<MT>;

    //// inherent variables
    using Base::m_lengthPerCell;
    USING_BASE_SOA_SET_GET(m_divergence, Divergence);         //!< Base::divergence
    USING_BASE_SOA_SET_GET(m_pressure, Pressure);             //!< Base::pressure
    USING_BASE_SOA_SET_GET(m_tempPressure, TempPressure);     //!< Base::temp pressure
    USING_BASE_SOA_SET_GET(m_density, Density);               //!< Base::density
    USING_BASE_SOA_SET_GET(m_tempDensity, TempDensity);       //!< Base::temp density
    USING_BASE_SOA_SET_GET(m_temperature, Temperature);        //!< Base::temperature
    USING_BASE_SOA_SET_GET(m_tempTemperature, TempTemperature);//!< Base::temp temperature
    USING_BASE_SOA_SET_GET(m_velocity, Velocity);             //!< Base::velocity
    USING_BASE_SOA_SET_GET(m_tempVelocity, TempVelocity);     //!< Base::temp velocity
    USING_BASE_SOA_SET_GET(m_vorticity, Vorticity);           //!< Base::vorticity
    
    Nuclear(uint3 gridSize, Real lengthPerCell):Base(gridSize, lengthPerCell){

    }

    virtual ~Nuclear() {
        LOG_OSTREAM_DEBUG << "release grid nuclear" << std::hex << &m_density << std::dec << std::endl;
        
        LOG_OSTREAM_DEBUG << "release grid nuclear finished"<<std::endl;
    }

    void resizeGF(uint3 gridSize, Real lengthPerCell) {
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
    };
    void setVelocity(vec3r vel){
        callFillArray<MT>(m_velocity.size(), vel, m_velocity);
    }

    void clearVelocity(){
        callFillArray<MT>(m_velocity.size(), make_vec3r(0.,0.,0.), m_velocity);
    }

    void applyBodyForce(vec3r bodyForce, Real dt){
        callAddArray<MT>(m_velocity.size(), bodyForce*dt, m_velocity);
    }

};

//this is set to record the information of each source
struct Source{
    //the position of the source
    float3 center;
    //time since the source was released
    Real time;

    Source(float3 c, Real t){
        center = c;
        time = t;
    }
};

//calculate the influence of the source for each grid
template<MemType MT>
void callAddConcentration(int size,int blocksize, uint3 totalSize,VecArray<Real,MT> height, int3 source,Real time, VecArray<Real, MT>& Concentration);

//clear the concentration of each grid
template<MemType MT>
void callClearC(int size,  VecArray<Real, MT>& Concentration);

//second way to accelerate the computation
template<MemType MT>
void callAddConcentration2(int size,int source_number, VecArray<Source, MT> sources,VecArray<Real, MT>& Concentration);
PHYS_NAMESPACE_END