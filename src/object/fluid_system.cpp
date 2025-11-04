#include "object/particle_system_util.h"
#include "fluid_system.h"
#include "common/timer.h"
#include "object/voxelize.h"
#include "object/sdf.h"
#include "object/aabbtree.h"
#include "object/mesh.h"
#include <algorithm>


using namespace physeng;

extern vec3r g_phaseColor[5]={
	make_vec3r(0.396, 0.651, 0.314),//// rigid
	make_vec3r(0.314, 0.396, 0.651),//// liquid
	make_vec3r(0.313, 0.651, 0.569),//// oil
	make_vec3r(0.651, 0.569, 0.314),//// sand
	make_vec3r(0.651, 0.314, 0.396) //// cloth 
};

vec3r getPhaseColor(const PhaseType& phase){
	if(phase<0 || phase>5) LOG_OSTREAM_ERROR<<"invalid phase"<<std::endl;
	return g_phaseColor[phase];
};

FluidSystem::FluidSystem(Real radius, uint3 gridSize, vec3r worldMin, vec3r worldMax, bool useFoam, bool useSurfaceTension) :
	m_numParticles(0),
	m_hx(),
	m_hv(),
	m_hm(),
	m_hp(),
	m_hc(),
	m_dq(),
	m_dr(),
	m_c(),
	m_dConstraintStart(),
	m_dConstraintCnt(),
	m_hRigidParticleSign(),
	m_dConstraintParticleMap(){

	const float kPi = 3.141592654f;
	m_gridSortBits = 18;    // increase this for larger grids

	//// default params
	m_phaseFlag=0;
	m_isInited=false;
	
	//// default params for solver
	m_subSteps=1;
	m_solverIterations=3;
	
	//// default params for particles
	m_gridSize=gridSize;
	m_curNumParticles=0;
	m_rigidConstraintCount=0;
	m_rigidParticleCount=0;

	//// simulation world
	m_params.worldOrigin = worldMin;
	m_params.worldMin = worldMin;
	m_params.worldMax = worldMax;

	//// particle radius
	m_params.particleRadius = radius;
	m_params.maxNeighbours = 500;

	//// kernel radius
	m_params.h = radius*4;
	m_params.halfh = radius*2;
	m_params.h2 = m_params.h * m_params.h;
	m_params.volume = (4.f/3.f)*kPi*radius*radius*radius;

	//// particle grid
	m_params.gridSize = m_gridSize;
	m_params.gridHashMask = make_uint3(m_gridSize.x-1,m_gridSize.y-1,m_gridSize.z-1);
	m_params.gridHashMultiplier = make_uint3(1,m_gridSize.x,m_gridSize.x*m_gridSize.y);
	m_numCells = m_gridSize.x * m_gridSize.y * m_gridSize.z;
	m_params.numCells = m_numCells;
	m_params.cellLength = make_vec3r(m_params.h, m_params.h, m_params.h);
	m_params.invCellLength = make_vec3r(1./m_params.cellLength.x, 1./m_params.cellLength.y, 1./m_params.cellLength.z);

	//// density/mass
	m_params.density0=1000.f;
	m_params.invDensity0=1.f/m_params.density0;

	m_params.rho0[(int)PhaseType::Rigid]=1000.f;
	m_params.rho0[(int)PhaseType::Liquid]=1000.f;
	m_params.rho0[(int)PhaseType::Oil]=250.f;
	m_params.rho0[(int)PhaseType::Sand]=2000.f;
	m_params.rho0[(int)PhaseType::Cloth]=1000.f;
	for(int i=0;i<5;i++) {
		m_params.invRho0[i]=1.f/m_params.rho0[i];
		m_params.pmass[i]=m_params.volume*m_params.rho0[i];
	}

	//// kernel coef
	m_params.poly6Coef = 315.f / (64.f * kPi * pow(m_params.h, 9));
    m_params.zeroPoly6 = m_params.poly6Coef * pow(m_params.h2,3);
	m_params.spikyGradCoef = -45.f / (kPi * pow(m_params.h, 6));
	m_params.cohesionCoef = 32.f / (kPi * pow(m_params.h, 9));
	m_params.cohesionConstCoef = pow(m_params.h, 6) / 64.f;

	//// physics parameter
	m_params.gravity = make_vec3r(0.0f, -9.8f, 0.0f);
	m_params.globalDamping = 1.0f;
	m_params.boundaryDamping = 0.05f;

	//// rigid/granular material
	m_params.peneDist=radius*2.0f-0.0001f*radius;
	m_params.peneDist2=m_params.peneDist*m_params.peneDist;
	m_params.staticFriction = 50.0f;
	m_params.dynamicFriction = 50.0f;
	// m_params.invStaticFriction = 1.0f/m_params.staticFriction;

    //// fluid density constraint
	m_params.solidPressure = 1.f;
	m_params.lambdaRelaxation = 0.8f;
	float mul = m_params.h2 - 0.01 * m_params.h2;
	m_params.pressure = m_params.poly6Coef * mul * mul * mul;
	m_params.invPressure = 1.0f / m_params.pressure;

	//// vorticity/viscosity/surface tension
	m_params.kvorticity=1.0f;
	m_params.kviscosity=0.01f;
	m_params.useSurfaceTension=useSurfaceTension;
	m_params.kcurvature=1.0f;
	m_params.kcohesion=0.5f;

	m_params.useFoam = useFoam;

	//// control param
	m_params.sleepVelocity=9.81f*0.005f;
	m_params.sleepVelocity2=m_params.sleepVelocity*m_params.sleepVelocity;
	// m_params.maxVelocity=30.0f;
	m_params.maxVelocity=0.5f*radius*2/0.016;
	// std::cout<<"m_params.maxVelocity="<<m_params.maxVelocity<<std::endl;
	m_params.maxVelocity2=m_params.maxVelocity*m_params.maxVelocity;
	// m_params.maxVelocity=radius*100.0f;

	//// obstacles
	m_params.useColumnObstacle=false;
	m_params.useSphereObstacle=false;
	m_params.useTerrain=false;

	//// resize particle array
	pf.resizePF(m_curNumParticles);
	_initialize();

	if (useFoam) {
		cudaMallocManaged(&m_foamParticleCount, sizeof(int));
		*m_foamParticleCount = 0;
	}
}

FluidSystem::~FluidSystem(){
	_finalize();
	m_numParticles = 0;
}

void FluidSystem::_initialize(){
	assert(!m_isInited);
	
	m_dCellStart.resize(m_numCells, false);
	m_dCellEnd.resize(m_numCells, false);

	setParameters(&m_params);

	m_isInited = true;
}


void
FluidSystem::_finalize(){
	assert(m_isInited);

	m_hx.release();
	m_hv.release();
	m_hm.release();
	m_hp.release();
	m_hc.release();
	
	m_hRigidParticleSign.release();

	m_dConstraintStart.release();
	m_dConstraintCnt.release();
	m_dr.release();
	m_dq.release();
	m_dCellStart.release();
	m_dCellEnd.release();
	m_dConstraintParticleMap.release();
	// m_dNbrs.release();

	m_c.release();

	cudaFree(m_foamParticleCount);
}

void
FluidSystem::completeInit() {
	if (m_params.useFoam) {
		VecArray<vec3r, MemType::CPU> m_hc;
		m_hc.resize(m_curNumParticles);
		m_hc.fill(m_curNumParticles, make_vec3r(1.0f));
		copyArray<vec3r, MemType::GPU, MemType::CPU>(&m_c.m_data, &m_hc.m_data, m_curNumParticles, 0, m_curNumParticles);
	}
}

//// step the simulation
void
FluidSystem::update(float deltaTime){
	assert(m_isInited);

	if (!ifStartUpdate) {
		this->completeInit();
		ifStartUpdate = true;
	}
		

	Real subDt=deltaTime/m_subSteps;
	PHY_PROFILE("update");
	if (m_curNumParticles > 0)
		//// substeps
	for(uint s = 0; s < m_subSteps; s++) {
		{
			//// tx=x+v*dt
			callIntegrate<MemType::GPU>(
				m_curNumParticles,
				subDt,
				pf.getTempPositionRef(),
				pf.getPositionRef(),
				pf.getVelocityRef());
			cudaDeviceSynchronize();
		}
		{
			//// calculate grid hash
			PHY_PROFILE("update hash");
			callUpdateHash<MemType::GPU>(
				m_curNumParticles,
				pf.getParticleHashRef(),
				pf.getS2OParticleIndexRef(),
				pf.getTempPositionRef());
			cudaDeviceSynchronize();
		}
		{
			//// sort particles based on hash
			PHY_PROFILE("sort particles");
			sortParticles<MemType::GPU>(m_curNumParticles, pf.getParticleHashRef(), pf.getS2OParticleIndexRef());
			cudaDeviceSynchronize();
		}
		{
			//// reorder particle arrays into sorted order and find start and end of each cell
			PHY_PROFILE("reorder data");
			reorderDataAndFindCellStart(
				m_curNumParticles,
				m_dCellStart.m_data,
				m_dCellEnd.m_data,
				pf.getSortedPositionPhaseRef().m_data,
				pf.getSortedVelocityRef().m_data,
				pf.getSortedMassRef().m_data,
				pf.getSortedPhaseRef().m_data,
				pf.getParticleHashRef().m_data,
				pf.getO2SParticleIndexRef().m_data,
				pf.getS2OParticleIndexRef().m_data,
				pf.getTempPositionRef().m_data,
				pf.getVelocityRef().m_data,
				pf.getMassRef().m_data,
				pf.getPhaseRef().m_data,
				m_numCells,
				m_curNumParticles);
			cudaDeviceSynchronize();
		}
		
		//// solver iterations
		for (uint i = 0; i < m_solverIterations; i++){
			
			//// for solid material
			if(getPhaseFlag(PhaseType::Sand)||getPhaseFlag(PhaseType::Rigid)){

				PHY_PROFILE("resolve solid");
				//// solve penetration
				callResolvePenetration<MemType::GPU>(
					m_curNumParticles,
					pf.getTempPositionRef(), //// apply penetration
					pf.getSortedPositionPhaseRef(), //// predicted pos
					pf.getSortedMassRef(),
					// pf.getSortedPhaseRef(),
					pf.getS2OParticleIndexRef(),
					m_dCellStart,
					m_dCellEnd,
					pf.getRigidParticleSignRef());
				// pf.getSortedPositionRef().swap(pf.getTempPositionRef());

				//// solve friction
				callResolveFriction<MemType::GPU>(
					m_curNumParticles,
					pf.getTempStarPositionPhaseRef(), //// apply friction
					pf.getTempPositionRef(), //// apply penetration
					pf.getPositionRef(), //// old pos
					pf.getSortedPositionPhaseRef(), //// predicted pos
					pf.getSortedMassRef(),
					// pf.getSortedPhaseRef(),
					pf.getS2OParticleIndexRef(),
					m_dCellStart,
					m_dCellEnd,
					pf.getRigidParticleSignRef());
				// pf.getSortedPositionRef().swap(pf.getTempStarPositionRef());

				copyArray<vec4r, MemType::GPU, MemType::GPU>(&pf.getSortedPositionPhaseRef().m_data, &pf.getTempStarPositionPhaseRef().m_data, 0, m_curNumParticles);

				if(getPhaseFlag(PhaseType::Rigid))
				callSolveShapeMatching<MemType::GPU>(
					m_curNumParticles,
					m_rigidConstraintCount,
					pf.getTempStarPositionPhaseRef(),
					pf.getSortedPositionPhaseRef(),
					//// TODO: put into fluid???
					m_dConstraintStart,
					m_dConstraintCnt,
					m_dr,
					m_dConstraintParticleMap,
					m_dq,
					pf.getSortedMassRef(),
					// pf.getSortedPhaseRef(),
					pf.getO2SParticleIndexRef(),
					pf.getS2OParticleIndexRef());
				pf.getSortedPositionPhaseRef().swap(pf.getTempStarPositionPhaseRef());
				cudaDeviceSynchronize();
			}


			if(getPhaseFlag(PhaseType::Liquid)||getPhaseFlag(PhaseType::Oil)){
				// {
				// 	PHY_PROFILE("compute nbr");
				// 	callUpdateNbrList<MemType::GPU>(
				// 		m_curNumParticles, 
				// 		m_dNbrs, 
				// 		pf.getSortedPositionPhaseRef(),
				// 		pf.getS2OParticleIndexRef(),
				// 		m_dCellStart,
				// 		m_dCellEnd);
				// 	cudaDeviceSynchronize();
				// }
				{
					PHY_PROFILE("update lambda test");
					callUpdateLambda<MemType::GPU>(
						m_curNumParticles,
						pf.getLambdaRef(),
						pf.getInvDensityRef(),
						pf.getSortedPositionPhaseRef(),
						pf.getSortedMassRef(),
						// pf.getSortedPhaseRef(),
						pf.getS2OParticleIndexRef(),
						m_dCellStart,
						m_dCellEnd);
					// callUpdateLambdaFast<MemType::GPU>(
					// 	m_curNumParticles,
					// 	pf.getLambdaRef(),
					// 	pf.getInvDensityRef(),
					// 	m_dNbrs, 
					// 	pf.getSortedPositionPhaseRef());
					cudaDeviceSynchronize();
				}

				if(i!=m_solverIterations-1) {
					PHY_PROFILE("solve fluid");
					callSolveFluid<MemType::GPU>(
						m_curNumParticles,
						pf.getNormalRef(),
						// pf.getTempPositionRef(),
						pf.getTempPositionPhaseRef(),
						pf.getLambdaRef(),
						pf.getInvDensityRef(),
						// pf.getSortedPositionRef(),
						pf.getSortedPositionPhaseRef(),
						pf.getSortedMassRef(),
						// pf.getSortedPhaseRef(),
						pf.getS2OParticleIndexRef(),
						m_dCellStart,
						m_dCellEnd);
					// callSolveFluidFast<MemType::GPU>(
					// 	m_curNumParticles,
					// 	pf.getNormalRef(),
					// 	pf.getTempPositionPhaseRef(),
					// 	pf.getLambdaRef(),
					// 	pf.getInvDensityRef(),
					// 	m_dNbrs, 
					// 	pf.getSortedPositionPhaseRef());
					pf.getSortedPositionPhaseRef().swap(pf.getTempPositionPhaseRef());
					cudaDeviceSynchronize();
				}
			}
			
			if(m_params.useColumnObstacle){
				PHY_PROFILE("collide collumn");
				callCollideStaticObstacle<MemType::GPU>(
					m_curNumParticles,
					pf.getSortedPositionPhaseRef());
				cudaDeviceSynchronize();
			}

			if(m_params.useSphereObstacle){
				PHY_PROFILE("collide sphere");
				callCollideStaticSphere<MemType::GPU>(
					m_curNumParticles,
					pf.getSortedPositionPhaseRef());
				cudaDeviceSynchronize();
			}

			if(m_params.useTerrain){
				PHY_PROFILE("collide terrain");
				callCollideTerrain<MemType::GPU>(
					m_curNumParticles,
					pf.getSortedPositionPhaseRef(),
					m_dTerrainHeight,
					make_vec3r(-18.0f, 0.0f, -18.0f),
					100,
					0.35f); // dongqingtai: 0.6f -> 0.35f
				cudaDeviceSynchronize();
			}
		}
		
		if(getPhaseFlag(PhaseType::Liquid)||getPhaseFlag(PhaseType::Oil)){
			PHY_PROFILE("solve fluid and visc");
			callSolveFluidAndViscosity<MemType::GPU>(
				m_curNumParticles,
				pf.getNormalRef(),
				pf.getDeltaVelocityRef(),
				// pf.getTempPositionRef(),
				pf.getTempPositionPhaseRef(),
				pf.getLambdaRef(),
				pf.getInvDensityRef(),
				pf.getSortedPositionPhaseRef(),
				pf.getSortedVelocityRef(),
				pf.getSortedMassRef(),
				// pf.getSortedPhaseRef(),
				pf.getS2OParticleIndexRef(),
				m_dCellStart,
				m_dCellEnd);
			pf.getSortedPositionPhaseRef().swap(pf.getTempPositionPhaseRef());
			cudaDeviceSynchronize();
		}

		//// apply surface tension
		if((getPhaseFlag(PhaseType::Liquid)||getPhaseFlag(PhaseType::Oil)) && m_params.useSurfaceTension){
			PHY_PROFILE("update surface tension");
			callUpdateSurfaceTension<MemType::GPU>(
				m_curNumParticles,
				pf.getForceRef(),
				pf.getNormalRef(),
				pf.getInvDensityRef(),
				pf.getSortedPositionPhaseRef(),
				pf.getSortedMassRef(),
				// pf.getSortedPhaseRef(),
				pf.getS2OParticleIndexRef(),
				m_dCellStart,
				m_dCellEnd);
			cudaDeviceSynchronize();
		}

		//// update position & velocity
		//if(m_params.useFoam)
		//{
		//	PHY_PROFILE("update velocity");
		//	callUpdatePositionVelocity<MemType::GPU>(
		//		m_curNumParticles, 
		//		subDt, 
		//		pf.getSortedPositionPhaseRef(),
		//		pf.getTempPositionRef(),
		//		// pf.getPositionRef(),
		//		pf.getTempVelocityRef(),
		//		// pf.getVelocityRef(),
		//		pf.getDeltaVelocityRef(),
		//		pf.getForceRef(),
		//		pf.getS2OParticleIndexRef(),
		//		m_dCellStart,
		//		m_dCellEnd);
		//	cudaDeviceSynchronize();
		//}
		//else 
		{
			PHY_PROFILE("update velocity");
			callUpdatePositionVelocity<MemType::GPU>(
				m_curNumParticles,
				subDt,
				pf.getSortedPositionPhaseRef(),
				pf.getPositionRef(),
				pf.getVelocityRef(),
				pf.getDeltaVelocityRef(),
				pf.getForceRef(),
				pf.getS2OParticleIndexRef(),
				m_dCellStart,
				m_dCellEnd);
			cudaDeviceSynchronize();
		}


		
		//// apply vorticity
		if((getPhaseFlag(PhaseType::Liquid)||getPhaseFlag(PhaseType::Oil)) && m_params.kvorticity>0){
			PHY_PROFILE("add vorticity");
			callUpdateVorticity<MemType::GPU>(
				m_curNumParticles, 
				pf.getVorticityRef(),
				pf.getInvDensityRef(),
				pf.getSortedPositionPhaseRef(),
				pf.getVelocityRef(),
				pf.getSortedMassRef(),
				pf.getSortedPhaseRef(),
				pf.getS2OParticleIndexRef(),
				m_dCellStart,
				m_dCellEnd);
				
			callApplyVorticity<MemType::GPU>(
				m_curNumParticles, 
				subDt, 
				pf.getVelocityRef(),
				pf.getVorticityRef(),
				pf.getInvDensityRef(),
				pf.getSortedPositionPhaseRef(),
				pf.getSortedMassRef(),
				pf.getSortedPhaseRef(),
				pf.getS2OParticleIndexRef(),
				m_dCellStart,
				m_dCellEnd);
			cudaDeviceSynchronize();
		}

	}
	if(m_params.useFoam){
		{
			PHY_PROFILE("calc color field");
			callUpdateColorField<MemType::GPU>(
				m_curNumParticles - m_rigidParticleCount,
				pf.getColorFieldRef(),
				pf.getInvDensityRef(),
				pf.getSortedPositionPhaseRef(),
				m_dCellStart,
				m_dCellEnd);
		}
		{
			PHY_PROFILE("calc normal");
			callUpdateNormal<MemType::GPU>(
				m_curNumParticles - m_rigidParticleCount,
				pf.getNormalRef(),
				pf.getColorFieldRef(),
				pf.getSortedPositionPhaseRef(),
				m_dCellStart,
				m_dCellEnd);
		}
		{
			PHY_PROFILE("generate foam particle");
			generateFoamParticle<MemType::GPU>(
				m_curNumParticles,
				m_rigidParticleCount,
				deltaTime,
				m_foamParticleCount,
				m_fpx,
				m_fpv,
				m_fplife,
				pf.getSortedPositionPhaseRef(),
				pf.getNormalRef(),
				pf.getVelocityRef(),
				pf.getS2OParticleIndexRef(),
				m_dCellStart,
				m_dCellEnd,
				m_c);

			cudaDeviceSynchronize();
		}
		if((*m_foamParticleCount)>0)
		{
			PHY_PROFILE("advect foam particle");
			advectFoamParticle<MemType::GPU>(
				*m_foamParticleCount,
				deltaTime,
				m_fpx,
				m_fpv,
				m_fplife,
				pf.getSortedPositionPhaseRef(),
				pf.getVelocityRef(),
				pf.getS2OParticleIndexRef(),
				m_dCellStart,
				m_dCellEnd);
			cudaDeviceSynchronize();
		}
		 if ((*m_foamParticleCount) > 0)
		 {
		 	PHY_PROFILE("sort foam particle");
		 	sortFoamParticle<MemType::GPU>(
		 		*m_foamParticleCount,
		 		S20FoamParticleIndex,
		 		m_fpx,
		 		m_tfpx,
		 		m_fpv,
		 		m_tfpv,
		 		m_fplife);
		 	cudaDeviceSynchronize();
		 }
		{
			PHY_PROFILE("remove foam particle");
			removeFoamParticle<MemType::GPU>(
				m_curNumParticles,
				m_foamParticleCount,
				m_fpx,
				m_fpv,
				m_fplife);
			cudaDeviceSynchronize();
		}
		if (m_params.useTerrain && (*m_foamParticleCount))
		{
			PHY_PROFILE("collide terrain foam particle");
			callCollideTerrainFoam<MemType::GPU>(
				*m_foamParticleCount,
				m_fpx,
				m_dTerrainHeight,
				make_vec3r(-18.0f, 0.0f, -18.0f),
				100,
				0.6f);
			cudaDeviceSynchronize();
		}
		{
			PHY_PROFILE("copy foam particle");
			cudaMemcpy(pf.getPositionRef().m_data+m_curNumParticles, m_fpx.m_data, m_curNumParticles * sizeof(vec3r), cudaMemcpyDeviceToDevice);
		}
		//printf("m_foamParticleCount %d\n", *m_foamParticleCount);
	}
	//LOG_OSTREAM_DEBUG << "m_foamParticleCount " << (*m_foamParticleCount) << std::endl;

}


float SampleSDF(const float* sdf, int dim, int x, int y, int z)
{
	assert(x < dim && x >= 0);
	assert(y < dim && y >= 0);
	assert(z < dim && z >= 0);

	return sdf[z * dim * dim + y * dim + x];
}

// return normal of signed distance field
vec3r SampleSDFGrad(const float* sdf, int dim, int x, int y, int z)
{
	int x0 = std::max(x - 1, 0);
	int x1 = std::min(x + 1, dim - 1);

	int y0 = std::max(y - 1, 0);
	int y1 = std::min(y + 1, dim - 1);

	int z0 = std::max(z - 1, 0);
	int z1 = std::min(z + 1, dim - 1);

	float dx = (SampleSDF(sdf, dim, x1, y, z) - SampleSDF(sdf, dim, x0, y, z)) * (dim * 0.5f);
	float dy = (SampleSDF(sdf, dim, x, y1, z) - SampleSDF(sdf, dim, x, y0, z)) * (dim * 0.5f);
	float dz = (SampleSDF(sdf, dim, x, y, z1) - SampleSDF(sdf, dim, x, y, z0)) * (dim * 0.5f);

	return make_vec3r(dx, dy, dz);
}

void FluidSystem::addCubeFromMesh(vec3r center, vec3r scale, Real spacing, PhaseType phase)
{
	std::vector<vec3r> vertices;
	vec3r start = center - scale * 0.5;
	for(int i = 0; i<2; i++){
		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 2; k++) {
				vertices.push_back(start + make_vec3r(i * scale.x, j * scale.y, k * scale.z));
			}
		}
	}
	std::vector<int> indices = {
		0,1,2,1,2,3,
		4,5,6,5,6,7,
		0,1,4,1,4,5,
		2,3,6,3,6,7,
		0,2,4,2,4,6,
		1,3,5,3,5,7
	};
	std::vector<vec4r> particles; 
	CreateRigidFromMesh(particles, vertices, indices, spacing, 1, phase);

	addParticles(particles, phase);
}

void FluidSystem::addModel(const char* path, vec3r center, vec3r scale, Real spacing, PhaseType phase) {
	Mesh* mesh = ImportMesh(path);
	mesh->Normalize();
	for (int i = 0; i < mesh->m_positions.size(); i++) {
		vec3r localPos = mesh->m_positions[i];
		mesh->m_positions[i] = center + make_vec3r(localPos.x * scale.x, localPos.y * scale.y, localPos.z * scale.z);
	}
	std::vector<vec4r> particles;
	CreateRigidFromMesh(particles, mesh->m_positions, mesh->m_indices, spacing, 1, phase);
	addParticles(particles, phase);
}

void FluidSystem::addParticles(std::vector<vec4r>& particles, PhaseType phase) {
	vec3r centerOfMass = make_vec3r(0.f, 0.f, 0.f);
	float massSum = 0.f;
	int addNum = particles.size();
	pf.resizePF(m_curNumParticles + addNum, m_params.useFoam);
	resize(m_curNumParticles + addNum);

	for (int i = 0; i < particles.size(); i++)
	{
		int curIndex = m_curNumParticles + i;
		m_hx[curIndex] = make_vec3r(particles[i]);

		m_hv[curIndex] = make_vec4r(0.0f);
		m_hc[curIndex] = getPhaseColor(PhaseType(phase));

		// m_hm[curIndex]=m_params.fmass;
		m_hm[curIndex] = m_params.pmass[(int)phase];
		m_hp[curIndex] = phase;
		m_hRigidParticleSign[curIndex] = (phase == PhaseType::Rigid) ? m_rigidConstraintCount : 0;

		centerOfMass = centerOfMass + m_hx[curIndex] * m_hm[curIndex];
		massSum += m_hm[curIndex];
	}
	if (addNum > 0) setPhaseFlag(phase);
	
	submitToDevice(m_curNumParticles, addNum);

	if (phase != PhaseType::Rigid) {
		m_curNumParticles += addNum;
		return;
	}

	if (massSum == 0.f) {
		LOG_OSTREAM_ERROR << "AddRigidbody : massSum == 0.f" << std::endl;
		return;
	}

	centerOfMass = make_vec3r(centerOfMass.x / massSum,
		centerOfMass.y / massSum,
		centerOfMass.z / massSum);

	uint* hConstraintStartIndex = new uint[1];
	uint* hConstraintParticleCount = new uint[1];
	uint* hConstraintParticleMap = new uint[addNum];
	vec3r* hq = new vec3r[addNum];
	float* hr = new float[9]
	{ 1.f,0.f,0.f,
	  0.f,1.f,0.f,
	  0.f,0.f,1.f };

	hConstraintStartIndex[0] = m_rigidParticleCount;
	hConstraintParticleCount[0] = addNum;

	//cal particle map, gethq
	for (uint i = 0; i < addNum; i++) {
		hConstraintParticleMap[i] = m_curNumParticles + i;
		hq[i] = m_hx[m_curNumParticles + i] - centerOfMass;
	}
	//set host to device
	copyArray<uint, MemType::GPU, MemType::CPU>(&m_dConstraintStart.m_data, &hConstraintStartIndex, m_rigidConstraintCount, 0, 1);
	copyArray<uint, MemType::GPU, MemType::CPU>(&m_dConstraintCnt.m_data, &hConstraintParticleCount, m_rigidConstraintCount, 0, 1);
	copyArray<uint, MemType::GPU, MemType::CPU>(&m_dConstraintParticleMap.m_data, &hConstraintParticleMap, m_rigidParticleCount, 0, addNum);

	copyArray<vec3r, MemType::GPU, MemType::CPU>(&m_dq.m_data, (vec3r**)&hq, m_rigidParticleCount, 0, addNum);
	copyArray<vec3r, MemType::GPU, MemType::CPU>(&m_dr.m_data, (vec3r**)&hr, m_rigidConstraintCount * 3, 0, 3);

	//checkConstraint(2, m_dConstraintStart, m_dConstraintCnt);

	//Last Assignment
	m_curNumParticles += addNum;
	m_rigidConstraintCount++;
	m_rigidParticleCount += addNum;

	//release
	delete[] hConstraintStartIndex;
	delete[] hConstraintParticleCount;
	delete[] hConstraintParticleMap;
	delete[] hq;
	delete[] hr;
}

void FluidSystem::CreateRigidFromMesh(std::vector<vec4r> &particles, std::vector<vec3r> vertices, std::vector<int> indices, float spacing, float expand, PhaseType phase)
{
	// Switch to relative coordinates by computing the mean position of the vertices and subtracting the result from every vertex position
	// The increased precision will prevent ghost forces caused by inaccurate center of mass computations
	int numVertices = vertices.size();
	int numTriangleIndices = indices.size();
	vec3r meshOffset = make_vec3r(0.0f);
	for (int i = 0; i < numVertices; i++)
	{
		meshOffset += vertices[i];
	}
	meshOffset /= float(numVertices);

	vec3r* relativeVertices = new vec3r[numVertices];
	for (int i = 0; i < numVertices; i++)
	{
		relativeVertices[i] = vertices[i] - meshOffset;
	}

	std::vector<vec4r> normals;//sdf information
	std::vector<int> phases;

	const vec3r* positions = relativeVertices;

	vec3r meshLower = make_vec3r(FLT_MAX), meshUpper = make_vec3r(-FLT_MAX);
	for (int i = 0; i < numVertices; ++i)
	{
		meshLower = Min(meshLower, positions[i]);
		meshUpper = Max(meshUpper, positions[i]);
	}

	vec3r edges = meshUpper - meshLower;
	float maxEdge = std::max(std::max(edges.x, edges.y), edges.z);

	// tweak spacing to avoid edge cases for particles laying on the boundary
	// just covers the case where an edge is a whole multiple of the spacing.
	float spacingEps = spacing * (1.0f - 1e-4f);

	// make sure to have at least one particle in each dimension
	int dx, dy, dz;
	dx = spacing > edges.x ? 1 : int(edges.x / spacingEps);
	dy = spacing > edges.y ? 1 : int(edges.y / spacingEps);
	dz = spacing > edges.z ? 1 : int(edges.z / spacingEps);

	int maxDim = std::max(std::max(dx, dy), dz);

	// expand border by two voxels to ensure adequate sampling at edges
	//meshLower -= 2.0f * make_vec3r(spacing);
	//meshUpper += 2.0f * make_vec3r(spacing);
	//maxDim += 4;

	// we shift the voxelization bounds so that the voxel centers
	// lie symmetrically to the center of the object. this reduces the 
	// chance of missing features, and also better aligns the particles
	// with the mesh
	vec3r meshShift;
	meshShift.x = 0.5f * (spacing - (edges.x - (dx - 1) * spacing));
	meshShift.y = 0.5f * (spacing - (edges.y - (dy - 1) * spacing));
	meshShift.z = 0.5f * (spacing - (edges.z - (dz - 1) * spacing));
	meshLower -= meshShift;

	// don't allow samplings with > 64 per-side	
	if (maxDim > 64)
		return;

	std::vector<uint32_t> voxels(maxDim * maxDim * maxDim);

	Voxelize(relativeVertices, numVertices, &indices[0], numTriangleIndices, maxDim, maxDim, maxDim, &voxels[0], meshLower, meshUpper);

	delete[] relativeVertices;

	std::vector<float> sdf(maxDim * maxDim * maxDim);
	MakeSDF(&voxels[0], maxDim, maxDim, maxDim, &sdf[0]);

	for (int x = 0; x < maxDim; ++x)
	{
		for (int y = 0; y < maxDim; ++y)
		{
			for (int z = 0; z < maxDim; ++z)
			{
				const int index = z * maxDim * maxDim + y * maxDim + x;

				// if voxel is marked as occupied the add a particle
				if (voxels[index])
				{
					vec3r position = meshLower + spacing * make_vec3r(float(x) + 0.5f, float(y) + 0.5f, float(z) + 0.5f);

					// normalize the sdf value and transform to world scale
					vec3r n = normalize(SampleSDFGrad(&sdf[0], maxDim, x, y, z));
					float d = sdf[index] * maxEdge;

					// move particles inside or outside shape
					position += n * expand;

					//normals.push_back(make_vec4r(n, d));
					particles.push_back(make_vec4r(position.x, position.y, position.z, 1.0f));
					phases.push_back(0);
				}
			}
		}
	}

	const int numParticles = int(particles.size());

	for (int i = 0; i < numParticles; i++) {
		particles[i] += make_vec4r(meshOffset, 0.0f);
	}

	printf("voxelize: %d Particles\n", numParticles);
	return;
}

void FluidSystem::submitToDevice(uint start, uint count){
	copyArray<vec3r, MemType::GPU, MemType::CPU>(&pf.getPositionRef().m_data, &m_hx.m_data, start, count);
	// copyArray<vec3r, MemType::GPU, MemType::CPU>(&pf.getTempPositionRef().m_data, &m_hx.m_data, start, count);
	copyArray<vec4r, MemType::GPU, MemType::CPU>(&pf.getVelocityRef().m_data, &m_hv.m_data, start, count);
	// copyArray<vec4r, MemType::GPU, MemType::CPU>(&pf.getTempVelocityRef().m_data, &m_hv.m_data, start, count);
	copyArray<Real, MemType::GPU, MemType::CPU>(&pf.getMassRef().m_data, &m_hm.m_data, start, count);
	copyArray<uint, MemType::GPU, MemType::CPU>(&pf.getPhaseRef().m_data, &m_hp.m_data, start, count);
	copyArray<vec3r, MemType::GPU, MemType::CPU>(&m_c.m_data, &m_hc.m_data, start, count);
	copyArray<uint, MemType::GPU, MemType::CPU>(&pf.getRigidParticleSignRef().m_data, &m_hRigidParticleSign.m_data, start, count);
}


vec3r make_jitter(vec3r scale){
	return make_vec3r(
		scale.x*rand()/(RAND_MAX + 1),
		scale.y*rand()/(RAND_MAX + 1),
		scale.z*rand()/(RAND_MAX + 1));
}

void FluidSystem::addDam(vec3r center, vec3r scale, Real spacing, PhaseType phase){
	srand(1973);

	vec3r start=center-scale*0.5f;
	vec3r jitterScale=make_vec3r(m_params.particleRadius,m_params.particleRadius,m_params.particleRadius)*0.01f;

	LOG_OSTREAM_DEBUG<<"add dam("<<scale.x<<","<<scale.y<<","<<scale.z<<") at ("<<center.x<<","<<center.y<<","<<center.z<<") with spacing "<<spacing<<" and phase "<<(int)phase<<std::endl;

	vec3r centerOfMass=make_vec3r(0.f, 0.f, 0.f);
	float massSum=0.f;

	uint i=0;
	bool isBreak=false;

	std::vector<vec4r> particles;
	for (Real z=0; z < scale.z; z+=spacing){
		for (Real y=0; y < scale.y; y+=spacing){
			for (Real x=0; x < scale.x; x+=spacing){
				uint curIndex=m_curNumParticles+i;

				//if(curIndex>=m_numParticles) {
				//	LOG_OSTREAM_INFO<<"reach max particle number: "<<m_curNumParticles <<"+"<<i<<">="<<m_numParticles<<std::endl;
				//	isBreak=true;
				//}
				if(isBreak) break;

				particles.push_back(make_vec4r(start + make_vec3r((Real)x, (Real)y, (Real)z) + make_jitter(jitterScale)));
				i++;
			}
			if(isBreak) break;
		}
		if(isBreak) break;
	}
	LOG_OSTREAM_DEBUG<<"add dam by sampling #"<<i<<" particles at "<<m_curNumParticles<<std::endl;

	addParticles(particles, phase);
}

void FluidSystem::addSandpile(vec3r center, vec3r scale, Real spacing, PhaseType phase) {
	srand(1973);

	vec3r start = center - scale * 0.5f;
	vec3r jitterScale = make_vec3r(m_params.particleRadius, m_params.particleRadius, m_params.particleRadius) * 0.01f;

	LOG_OSTREAM_DEBUG << "add dam(" << scale.x << "," << scale.y << "," << scale.z << ") at (" << center.x << "," << center.y << "," << center.z << ") with spacing " << spacing << " and phase " << (int)phase << std::endl;

	vec3r centerOfMass = make_vec3r(0.f, 0.f, 0.f);
	float massSum = 0.f;

	uint i = 0;
	bool isBreak = false;

	std::vector<vec4r> particles;

	for (Real y = 0; y < scale.y; y += spacing) {
		for (Real x = 0; x < scale.x; x += spacing) {
			if (x < scale.x / scale.y * y) 
				continue;
			for (Real z = 0; z < scale.z; z += spacing) {
				uint curIndex = m_curNumParticles + i;

				//if(curIndex>=m_numParticles) {
				//	LOG_OSTREAM_INFO<<"reach max particle number: "<<m_curNumParticles <<"+"<<i<<">="<<m_numParticles<<std::endl;
				//	isBreak=true;
				//}
				
				if (isBreak) break;

				particles.push_back(make_vec4r(start + make_vec3r((Real)x, (Real)y, (Real)z) + make_jitter(jitterScale)));
				i++;
			}
			if (isBreak) break;
		}
		if (isBreak) break;
	}
	LOG_OSTREAM_DEBUG << "add dam by sampling #" << i << " particles at " << m_curNumParticles << std::endl;

	addParticles(particles, phase);
}

void FluidSystem::addTerrain()
{
	m_params.useTerrain = true;
	m_hTerrainHeight.resize(100*100);
	m_dTerrainHeight.resize(100*100);

	for (int i = 0; i < 100; i++)
	{
		for (int j = 0; j < 100;j++)
		{
			m_hTerrainHeight[i * 100 + j] = 2.0f * (sin(i/10.0f)+1) * (cos(j/8.0f)+1);
		}
	}

	copyArray<Real, MemType::GPU, MemType::CPU>(&m_dTerrainHeight.m_data, &m_hTerrainHeight.m_data, 0, 100 * 100);
}

void FluidSystem::addTerrain(const char* filePath)
{
	// 1. 与原来保持一致：固定 100×100
	const int nx = 100, ny = 100;
	const int n = nx * ny;

	m_params.useTerrain = true;
	m_hTerrainHeight.resize(n);
	m_dTerrainHeight.resize(n);

	// 2. 读取文件 ----------------------------------------------------------
	std::vector<float> raw(n);
	std::ifstream ifs(filePath, std::ios::binary);
	if (!ifs) {
		std::cerr << "[addTerrain] cannot open file: " << filePath << std::endl;
		return;
	}

	// 尝试二进制读取
	ifs.read(reinterpret_cast<char*>(raw.data()), n * sizeof(float));
	if (ifs.gcount() != n * static_cast<int>(sizeof(float))) {
		// 二进制失败，回退到文本读取
		ifs.close();
		ifs.open(filePath);
		if (!ifs) {
			std::cerr << "[addTerrain] cannot reopen file: " << filePath << std::endl;
			return;
		}
		for (int i = 0; i < n; ++i) {
			if (!(ifs >> raw[i])) {
				std::cerr << "[addTerrain] file does not contain 100×100 values." << std::endl;
				return;
			}
		}
	}

	// 3. 归一化到 [0, 8] 区间（与原 demo 幅度一致） -------------------------
	auto mm = std::minmax_element(raw.begin(), raw.end());
	float hMin = *mm.first;
	float hMax = *mm.second;
	float scale = (hMax > hMin) ? 8.0f / (hMax - hMin) : 1.0f;

	for (int i = 0; i < n; ++i)
		m_hTerrainHeight[i] = (raw[i] - hMin) * scale;

	// 4. 上传至 GPU --------------------------------------------------------
	copyArray<Real, MemType::GPU, MemType::CPU>(
		&m_dTerrainHeight.m_data, &m_hTerrainHeight.m_data, 0, n);
}


void FluidSystem::clear(){
	m_curNumParticles = 0;
	m_rigidConstraintCount = 0;
	m_rigidParticleCount = 0;
}

void FluidSystem::resize(int num) {
	////// allocate host storage
	m_hx.resize(num);
	m_hx.fill(num, make_vec3r(0.0f));

	m_hv.resize(num);
	m_hm.resize(num);
	m_hp.resize(num);
	m_hc.resize(num);
	m_hv.fill(num, make_vec4r(0.0f));
	m_hm.fill(num, 0.0f);
	m_hp.fill(num, 0);
	m_hc.fill(num, make_vec3r(0.5f,0.5f,0.5f));

	m_hRigidParticleSign.resize(num);
	m_hRigidParticleSign.fill(num, 0);

	//// allocate device storage
	m_dConstraintStart.resize(num);
	m_dConstraintCnt.resize(num);
	m_dConstraintParticleMap.resize(num);

	m_dr.resize(num *3);
	m_dq.resize(num);

	if (m_params.useFoam) {
		m_c.resize(num * 2);
	}
	else {
		m_c.resize(num);
	}
	if (m_params.useFoam) {
		m_fpx.resize(num);
		m_tfpx.resize(num);
		m_fpv.resize(num);
		m_tfpv.resize(num);
		m_fplife.resize(num);
		S20FoamParticleIndex.resize(num);
	}
}

// Real getJitter(const Real& scale){
// 	return (frand() * 2.0f - 1.0f)*scale;
// }

// vec3r getJitter(const vec3r& scale){
// 	return make_vec3r(
// 		(frand()*2.0f-1.0f)*scale.x,
// 		(frand()*2.0f-1.0f)*scale.y,
// 		(frand()*2.0f-1.0f)*scale.z
// 	);
// }