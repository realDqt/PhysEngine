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
    vec3r worldMin = make_vec3r(-15, 0, -15), worldMax = make_vec3r(15, 25, 15);
    FluidWorld* fluidWorld = new FluidWorld(worldMin, worldMax);
    int fluidIndex = fluidWorld->initFluidSystem(make_vec3r(-4, 9, 0), make_vec3r(7, 10, 8)*1.4, 0.0f, 0.05f);

    fluidWorld->addSandpile();


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
            fluidWorld->setWorldBoundary(worldMin - make_vec3r(6 * sinr(frame * fluidWorld->getDt()), 0, 0), worldMax);
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