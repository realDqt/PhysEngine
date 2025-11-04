#include "object/particle_fluid.h"

PHYS_NAMESPACE_BEGIN

template<MemType MT>
void initParticleFluid(){
    LOG_OSTREAM_DEBUG<<"init"<<int(MT)<<std::endl;
};

template void initParticleFluid<MemType::GPU>();
template void initParticleFluid<MemType::CPU>();

PHYS_NAMESPACE_END
