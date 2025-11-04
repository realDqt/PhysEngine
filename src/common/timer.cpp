#include <string>

#include "common/timer.h"

#ifdef PE_USE_OMP
std::unique_ptr<Timer> g_timer = std::make_unique<OmpTimer>();
#else
std::unique_ptr<Timer> g_timer = std::make_unique<BasicTimer>();
#endif

#ifdef GLOBALBENCHMARK
std::unique_ptr<BenchmarkTimer> g_benchmarkTimer = std::make_unique<BenchmarkTimer>();
#endif