#include "object/fluid_system.h"
#include "object/particle_fluid.h"
#include "cuda_viewer/cuda_viewer.h"
#include "object/config.h"
#include "common/timer.h"
#include "object/fluid_world.h"
using namespace physeng;
#define GRID_SIZE 64
uint numParticles = 0;
uint3 gridSize;

StopWatchInterface *timer = NULL;

int main(int argc, char** argv) {
    cudaInit(argc, argv);

    FluidWorld* fluidWorld = new FluidWorld(make_vec3r(-15, 0, -15), make_vec3r(15, 25, 15));
    int fluidIndex = fluidWorld->initFluidSystem(make_vec3r(-4, 9, -4), make_vec3r(7, 10, 8) * 1.8, 0.0f, 0.05f);
    fluidWorld->updateColumn(0, 2, make_vec3r(8, 0, 0));
    fluidWorld->updateColumn(1, 2, make_vec3r(8, 0, 5));
    fluidWorld->updateColumn(2, 2, make_vec3r(8, 0, -5));


    printf("Fluid Symbol:%d\n", fluidIndex);
    if (fluidIndex < 0) {
        exit(0);
    }

    fluidWorld->completeInit(fluidIndex);

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
            // PHY_PROFILE("grid system update");
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