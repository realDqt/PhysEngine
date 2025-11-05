#include "gas_system.h"
#include "nuclear_system.h"
#include "config.h"
#include <vector>

#define GAS_GRID_SIZE 60
#define NUCLEAR_GRID_SIZE 100

class GasWorld
{
private:
    std::vector<GasSystem*> gasArray;
    std::vector<NuclearSystem*> nuclearArray;

    DEFINE_MEMBER_SET_GET(Real, dt, Dt);
    DEFINE_MEMBER_SET_GET(vec3r, origin, Origin);
    bool useGas = true;
public:
    GasWorld();
    ~GasWorld();

    int initGasSystem(vec3r origin, Real vorticity, Real diffusion, Real buoyancy, Real vcEpsilon, Real decreaseDensity);
    int initGasSystem(const char* config_file);
    int initNuclearSystem(vec3r origin, Real vorticity);
    GasSystem* getGas(int index);
    NuclearSystem* getNuclear(int index) { return nuclearArray[index]; }
    void setRenderData(int index, vec3r color, vec3r lightDir, Real decay, unsigned char ambient);
    Real getAverageDensity(int index);


    void setIfDissipate(int index, bool flag);
    void update(int index);

    void setFireSmoke(int index);
    void setExplodeSmoke(int index);
    void setExhaust(int index);
    void setDust(int index);
    void setBiochemistry(int index);

    void setWind(Real strength, vec3r direction);
};

NuclearSystem* init(Real vorticity);