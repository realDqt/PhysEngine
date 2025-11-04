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

void readVec3(vec3r& vec) {
    std::istringstream iss;
    std::string str;

    std::cin >> str;
    iss.str(str);

    float tempArray[3];

    std::string tempStr;
    for (int i = 0; i < 3; i++) {
        getline(iss, tempStr, ',');
        tempArray[i] = atof(tempStr.data());
    }
    vec = make_vec3r(tempArray[0], tempArray[1], tempArray[2]);
}

void readReal(Real& r) {
    std::cin >> r;
}

int main(int argc, char** argv) {
    if (argc < 9) {
        printf("lack params\n");
        return -1;
    }
    cudaInit(argc, argv);

    vec3r center, scale;
    Real kvorticity, kviscosity;

    center.x = atof(argv[1]);
    center.y = atof(argv[2]);
    center.z = atof(argv[3]);

    scale.x = atof(argv[4]);
    scale.y = atof(argv[5]);
    scale.z = atof(argv[6]);

    kvorticity = atof(argv[7]);
    kviscosity = atof(argv[8]);

    FluidWorld* fluidWorld = new FluidWorld(make_vec3r(-15, 0, -15), make_vec3r(15, 25, 15));
    int fluidIndex = fluidWorld->initFluidSystem(center, scale, kvorticity, kviscosity);

    printf("Fluid Symbol:%d\n", fluidIndex);
    if (fluidIndex < 0) {
        exit(0);
    }
}