#include "object/fluid_system.h"
#include "object/particle_fluid.h"
#include "cuda_viewer/cuda_viewer.h"
#include "object/config.h"
#include "common/timer.h"
#include "object/fluid_world.h"
#include <sstream>
#include <iostream>
using namespace physeng;
#define GRID_SIZE 64
uint numParticles = 0;
uint3 gridSize;

StopWatchInterface *timer = NULL;

int main(int argc, char** argv) {
    cudaInit(argc, argv);

    vec3r worldMin = make_vec3r(0, 0, -5.58 - 5.58), worldMax = make_vec3r(33.32, 32, 5.58);
    FluidWorld* fluidWorld = new FluidWorld(worldMin, worldMax);
    int fluidIndex = fluidWorld->initFluidSystem(make_vec3r(5.58, 11.16, 0), make_vec3r(11.15, 22.32, 11.15), 0.0f, 0.05f, true);

    printf("Fluid Symbol:%d\n", fluidIndex);
    if (fluidIndex < 0) {
        exit(0);
    }

    CudaViewer viewer;
    viewer.init(argc, argv);
    viewer.bindFunctions();
    viewer.camera_trans[2] = -60;
    
    fluidWorld->initViewer(fluidIndex, viewer);
    sdkCreateTimer(&timer);

    int frame = 0;

    viewer.updateCallback = [&]() {
        sdkStartTimer(&timer);
        {
            fluidWorld->setWorldBoundary(worldMin - make_vec3r(12 * sinr(frame * fluidWorld->getDt()), 0, 0), worldMax);
            fluidWorld->update(fluidIndex);
            frame++;
        }

        fluidWorld->updateViewer(fluidIndex, viewer);
        sdkStopTimer(&timer);


        if (frame % 100 == 0) {
            float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
            LOG_OSTREAM_INFO << "frame " << frame << ", fps=" << ifps << ", time=" << sdkGetTimerValue(&timer) / 1000.f << std::endl;
            BENCHMARK_REPORT();
        }

        return true;
    };

    viewer.closeCallback = [&]() {
        sdkDeleteTimer(&timer);
        delete fluidWorld;
        return true;
    };
    viewer.isPause = true;
    viewer.run();
}