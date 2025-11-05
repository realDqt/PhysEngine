#include "fluid_world.h"
#include "common/logger.h"
#include <string>

Logger logger("fluidLog", 1024);

FluidWorld::FluidWorld(vec3r worldMin, vec3r worldMax)
{
    dt = 0.016f;
    this->worldMin = worldMin;
    this->worldMax = worldMax;
}

FluidWorld::~FluidWorld()
{
    for (auto fluid : fluidArray) {
        if (fluid) delete fluid;
    }
}

int FluidWorld::initFluidSystem(vec3r center, vec3r scale, Real kvorticity, Real kviscosity, bool useFoam) {
    Real radius = 0.2f;

    uint3 gridSize; gridSize.x = gridSize.y = gridSize.z = FLUID_GRID_SIZE;
    vec3r start = center - 0.5 * scale, end = center + 0.5 * scale;
    if (start.x < worldMin.x || start.y < worldMin.y || start.z < worldMin.z ||
        start.x > worldMax.x || start.y > worldMax.y || start.z > worldMax.z) {
        printf("ָ��λ�ó������緶Χ����������ˮ�壬����ʧ��\n");
        return -1;
    }

    if (kvorticity < 0 || kviscosity < 0) {
        printf("�������󣬲�������ˮ�壬����ʧ��\n");
        logger.Log(LogTypeKD::Error, "fluid's property is abnormal");
        return -1;
    }

    FluidSystem* fluidSystem = new FluidSystem(radius, gridSize, worldMin, worldMax, useFoam);
    fluidSystem->addDam(center, scale, radius * 1.8, PhaseType::Liquid);
    fluidSystem->m_params.kvorticity = kvorticity;
    fluidSystem->m_params.kviscosity = kviscosity;

    fluidArray.push_back(fluidSystem);

    int id = fluidArray.size() - 1;
    logger.Log(LogTypeKD::Info, "fluid id:" + std::to_string(id));
    return id;
}

int FluidWorld::initFluidSystem(const char* config_file, bool useFoam) {
    Config config(config_file);

    Real kvorticity, kviscosity;
    vec3r center, scale;
    GET_PARAM(config, center);
    GET_PARAM(config, scale);
    GET_PARAM(config, kvorticity);
    GET_PARAM(config, kviscosity);
    return initFluidSystem(center, scale, kvorticity, kviscosity, useFoam);
}

void FluidWorld::completeInit(int index) {
    fluidArray[index]->updateParams();
}

void FluidWorld::setWorldBoundary(vec3r worldMin, vec3r worldMax) {
    for (auto fsystem : fluidArray) {
        fsystem->m_params.worldMin = worldMin;
        fsystem->m_params.worldMax = worldMax;
        fsystem->updateParams();
    }
}

FluidSystem* FluidWorld::getFluid(int index) {
    if(index < 0 || index >= fluidArray.size()) {
        printf("fluid index %d is invalid\n", index);
        return nullptr;
	}
    return fluidArray[index];
}

void FluidWorld::updateColumn(int idx, Real r, vec3r x) {
    for (auto fsystem : fluidArray) {
        fsystem->m_params.useColumnObstacle = true;
        fsystem->updateColumn(idx, r, x);
    }
}

void FluidWorld::addCube(vec3r center, vec3r scale) {
    Real radius = 0.2f;
    for (auto fsystem : fluidArray) {
        fsystem->addDam(center, scale, radius * 2.0, PhaseType::Rigid);
    }
}

void FluidWorld::addSandpile() {
    Real widthX = (worldMax.x - worldMin.x) / 2.2;
    vec3r start = make_vec3r(worldMax.x - widthX, worldMin.y, worldMin.z);
    vec3r scale = make_vec3r(widthX, widthX / 2, worldMax.z - worldMin.z);
    Real radius = 0.2f;

    for (auto fsystem : fluidArray) {
        fsystem->addSandpile(start + scale * 0.5, scale, radius * 1.6, PhaseType::Sand);
    }
}

void FluidWorld::update(int index) {
    fluidArray[index]->update(dt);
}