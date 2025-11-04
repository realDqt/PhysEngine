#include "object/grid_fluid.h"

PHYS_NAMESPACE_BEGIN

template<MemType MT>
void initGridFluid(){
    LOG_OSTREAM_DEBUG<<"init"<<int(MT)<<std::endl;
};

template void initGridFluid<MemType::GPU>();
template void initGridFluid<MemType::CPU>();

GridSimParams hGridParams;

PHYS_NAMESPACE_END
