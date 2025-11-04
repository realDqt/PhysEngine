#pragma once

namespace linear_math {

#ifndef Real
    #ifdef USE_DOUBLE
        using Real = double;
    #else
        using Real = float;
    #endif
#endif

}

#define FUNC_ALIAS(alias, func, modifier)                                      \
    template <typename... Ts>                                                  \
    auto alias(Ts&&... ts) modifier->decltype(func(std::forward<Ts>(ts)...)) { \
        return func(std::forward<Ts>(ts)...);                                  \
    }
#define NON_CONST  // hack to pass empty argument to macro

#if defined(EIGEN_BACKEND)
    #include "eigen_backend.h"
#elif defined(ENOKI_BACKEND)
    #include "enoki_backend.h"
#else
    // use eigen backend by default
    #define EIGEN_BACKEND
    #include "eigen_backend.h"
    // #define ENOKI_BACKEND
    // #include "enoki_backend.h"
#endif