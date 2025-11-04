#include "nuclear_system.h"
#include "common/timer.h"
#include "common/logger.h"
#include <algorithm>

using namespace physeng;

NuclearSystem::NuclearSystem(uint3 gridSize, Real cellLength, vec3r worldMin, vec3r worldMax) :
	m_hd(),
	m_ht(),
	m_hv(),
	m_hc(),
	m_c(){

	const float kPi = 3.141592654f;

	m_isInited = false;

	m_solverIterations = 6;
	m_subSteps = 1;
	//// grid info
	m_gridSize = gridSize;
	m_cellLength = cellLength;
	//number of the grid
	m_numCells = m_gridSize.x * m_gridSize.y * m_gridSize.z;

	//// simulation world
	m_params.worldOrigin = worldMin;
	m_params.worldMin = worldMin;
	m_params.worldMax = worldMax;

	//// grid params
	m_params.gridSize = m_gridSize;
	m_params.numCells = m_numCells;
	m_params.cellLength = cellLength;
	m_params.invCellLength = 1. / cellLength;
	m_params.gridHashMultiplier = make_uint3(1, m_gridSize.x, m_gridSize.x * m_gridSize.y);

	//// physics parameter
	m_params.gravity = make_vec3r(0.0f, -9.8f, 0.0f);
	m_params.kvorticity = 5;
	m_params.kdiffusion = 0.01;
	m_params.bdensity = 0.3;
	m_params.btemperature = 0.2;

	//time since last source was released
	time_counter = 0;
	//the norm of the wind speed
	norm_wind = -1.0;
	//the angle of the wind
	wind_angle = 0;
	//the vector of the wind velocity
	wind_velocity = norm_wind * make_float3(cos(wind_angle/180.0 * REAL_PI),0,sin(wind_angle/180.0 * REAL_PI));
	//time interval between the source that was released
	r_interval = 2.0;
	//dry deposition rate
	Vs=make_float3(0,-(19050*9.8065*4/18/1.8)*10e-5,0);
	//whether the lead start
	start_pollution = true;

	gf.resizeGF(gridSize, cellLength);
	_initialize();
}

NuclearSystem::~NuclearSystem(){
	_finalize();
}

void NuclearSystem::_initialize(){
	assert(!m_isInited);

	// sources.alloc(1);
	// sources[0] = (Source(make_float3(m_gridSize.x / 4, m_gridSize.y - 20, m_gridSize.z / 4),0));
	//the first source
	sources.push_back(Source(make_float3(m_gridSize.x / 4, 60, m_gridSize.z / 4),0));

	
	height.resize(m_gridSize.x * m_gridSize.z);
	temp_height.resize(m_gridSize.x * m_gridSize.z);
	

	for(int i = 0 ;i < m_gridSize.z;++i){
		for(int j = 0; j < m_gridSize.x; ++j){
			if(j < 50){
				temp_height[j + i * m_gridSize.x] = 0;
			}
			else{
				temp_height[j+i * m_gridSize.x] = (j-50)* tan(30.0/180 * REAL_PI);
			}
		}
	}
	copyArray<Real,MemType::GPU,MemType::CPU>(&height.m_data,&temp_height.m_data,height.size());

	

	//// allocate host storage
	m_hd.resize(m_numCells, false);
	m_hd.fill(m_numCells, 0);
	m_ht.resize(m_numCells, false);
	m_ht.fill(m_numCells, 0);
	m_hv.resize(m_numCells, false);
	m_hv.fill(m_numCells, make_vec3r(0.0f));
	m_hc.resize(m_numCells, false);
	m_hc.fill(m_numCells, make_vec3r(0.9f,0.9f,0.9f));
	m_c.resize(m_numCells, false);
	sources_G.resize(m_numCells,false);
	

	setGridParameters(&m_params);
	submitToDevice(0, m_numCells);
	m_isInited = true;
}

void
NuclearSystem::_finalize(){
	assert(m_isInited);

	m_hd.release();
	m_ht.release();
	m_hv.release();
	m_hc.release();
	m_c.release();

	// sources.release();
}

float3 compute_v(float3 v0,float height){
	if(height < 0.2)
		height = 0.2;
	return v0*pow(height,0.3);
}

// void NuclearSystem::setM_hc(int idx,vec3r color){
// 	m_hc[idx] = color;
// }

//// step the simulation
void NuclearSystem::update(Real deltaTime) {
	assert(m_isInited);


	Real subDt = deltaTime / m_subSteps;
	//the factor when contact the mountains
	float floor_factor = 0.5;
	//just add this to constrain the speed
	float normal_factor = 0.5;

	if (norm_wind < 0.0f) {
		printf("��������δ����\n");
		logger.Log(LogType::Error, "��������δ����");
		exit(0);
	}

	//the angle of the wind
	wind_angle += 2* deltaTime;
	//calculate the wind velocity vector
	wind_velocity = norm_wind * make_float3(cos(wind_angle/180.0 * REAL_PI),0,sin(wind_angle/180.0 * REAL_PI));
	//update the source position
	for(int i =0;i<sources.size();++i){
		if((sources[i].center.x< 0 || sources[i].center.x >= m_gridSize.x ||sources[i].center.z < 0|| sources[i].center.z > m_gridSize.z)  
		||(sources[i].center.y>= temp_height[(int)sources[i].center.x + ((int)sources[i].center.z) * m_gridSize.x]))
			//wind velocity is changed with time
			sources[i].center+= deltaTime * (compute_v(wind_velocity,sources[i].center.y)+Vs)* normal_factor; 
		else{
			float3 norm_vector = make_float3(-0.5,pow(3,0.5)/2,0);
			float3 v_o = compute_v(wind_velocity,sources[i].center.y)+Vs;
			if(norm_vector.x* v_o.x +norm_vector.y* v_o.y +norm_vector.z* v_o.z >=0 ){
				sources[i].center+= deltaTime * (v_o+Vs)* normal_factor; 
			}
			else{
				//Add a velocity perpendicular to the slope
				float v_add_len = - v_o.y * cos(30.0/180*REAL_PI) + v_o.x * cos(60.0/180 * REAL_PI);
				float3 v_add = make_float3(- v_add_len * sin(30.0/180 * REAL_PI),v_add_len * cos(30.0/180 * REAL_PI),0);
				float3 v_final = v_o + v_add;
				//penalty factor to slow down
				sources[i].center += deltaTime * v_final * floor_factor; 
			}
		}
		

		sources[i].time += deltaTime;
	}
	
	// int3 source = make_int3(m_gridSize.x / 4, m_gridSize.y / 4, m_gridSize.z / 4);
	time_counter += deltaTime;
	//judge if a new source is released
	if(time_counter > r_interval){
		//whether start lead
		if(start_pollution)
			//position of the new source
			sources.push_back(Source(make_float3(m_gridSize.x / 4, 60, m_gridSize.z / 4),0));
		time_counter -= r_interval;
	}

	// copyArray<Source,MemType::GPU,MemType::CPU>(&sources_G.m_data,&sources.m_data,sources.size());
	
	Real density = 1;
	PHY_PROFILE("update");
	if (m_numCells > 0) {
		{
			//clear the source of each grid
			callClearC<MemType::GPU>(
				m_numCells,
				gf.getDensityRef());
			cudaDeviceSynchronize();
		}
		for(int i=0;i<sources.size();++i)
		{
			//add pollution for each grid
			callAddConcentration<MemType::GPU>(
				50*50*50,
				50,
				m_gridSize,
				height,
				make_int3(sources[i].center),
				sources[i].time,
				gf.getDensityRef());
			cudaDeviceSynchronize();
		}
		// {
		// 	callAddConcentration2<MemType::GPU>(
		// 	m_numCells,
		// 	sources.size(),
		// 	sources_G,
		// 	gf.getDensityRef());
		// 	cudaDeviceSynchronize();
		// }
		//// substeps
		for (uint s = 0; s < m_subSteps; s++) {
			
		}
	}	
	copyArray<Real, MemType::CPU, MemType::GPU>(&m_hd.m_data, &gf.getDensityRef().m_data, 0, m_numCells);
}
void NuclearSystem::submitToDevice(uint start, uint count){
	copyArray<Real, MemType::GPU, MemType::CPU>(&gf.getDensityRef().m_data, &m_hd.m_data, start, count);
	copyArray<Real, MemType::GPU, MemType::CPU>(&gf.getTemperatureRef().m_data, &m_ht.m_data, start, count);
	copyArray<vec3r, MemType::GPU, MemType::CPU>(&gf.getVelocityRef().m_data, &m_hv.m_data, start, count);
	copyArray<vec3r, MemType::GPU, MemType::CPU>(&m_c.m_data, &m_hc.m_data, start, count);
}

void NuclearSystem::clear(){
}