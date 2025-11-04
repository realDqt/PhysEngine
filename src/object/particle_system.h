#pragma once

#include <mutex>

#include "common/array.h"

PHYS_NAMESPACE_BEGIN

// Particle system class
template<MemType MT>
class ParticleSystem{

public:
	/**
	 * @param m_r Particle radius
	 * @param m_x Particle position array
	 * @param m_tx Temporary particle position array
	 * @param m_v Particle velocity array
	 * @param m_tv Temporary particle velocity array
	 * @param m_m Particle mass array
	 */
    DEFINE_MEMBER_GET(Real, m_r, Radius); 
    DEFINE_SOA_SET_GET(vec3r, MT, m_x, Position); 
    DEFINE_SOA_SET_GET(vec3r, MT, m_tx, TempPosition); 
    DEFINE_SOA_SET_GET(vec4r, MT, m_v, Velocity); 
    DEFINE_SOA_SET_GET(vec4r, MT, m_tv, TempVelocity); 
    DEFINE_SOA_SET_GET(Real, MT, m_m, Mass); 

public:
    // Constructor
    ParticleSystem(Real r, unsigned int size){
        m_r = r;
        //init particle system
        if(size!=0){
            m_x.resize(size);
            m_tx.resize(size);
            m_v.resize(size);
            m_tv.resize(size);
            m_m.resize(size);
        }
    }

    // Destructor
    virtual ~ParticleSystem() {
        LOG_OSTREAM_DEBUG<<"release m_x 0x"<<std::hex<<&m_x<<std::dec<<std::endl;
        m_x.release();
        m_tx.release();
        m_v.release();
        m_tv.release();
        m_m.release();
        LOG_OSTREAM_DEBUG<<"release ParticleSystem finished"<<std::endl;
    }
};

PHYS_NAMESPACE_END
