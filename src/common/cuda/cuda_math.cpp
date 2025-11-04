#include "common/cuda/cuda_math.h"

std::ostream& operator<<(std::ostream& os, const mat3r& m)
{
    os << "[[" << m(0,0) << ',' << m(0,1) << ',' << m(0,2) << "]," << std::endl
        << " [" << m(1,0) << ',' << m(1,1) << ',' << m(1,2) << "]," << std::endl
        << " [" << m(2,0) << ',' << m(2,1) << ',' << m(2,2) << "]]" << std::endl;
    return os;
}
