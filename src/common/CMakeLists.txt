# file(GLOB_RECURSE CPP_FILE
# 	${CMAKE_CURRENT_SOURCE_DIR}/*.c
#     ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
    
# file(GLOB_RECURSE HPP_FILE
# 	${CMAKE_CURRENT_SOURCE_DIR}/*.h
# 	${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

include_directories("${ENGINE_INCLUDE_DIR}")
include_directories("${TASKFLOW_INCLUDE_DIR}")
use_backend_warpper_local()


add_lib(EngineCudaCommon ${CMAKE_CURRENT_SOURCE_DIR})
use_cuda_local(EngineCudaCommon)

# set_target_properties(EngineCommon PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(EngineCommon PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)