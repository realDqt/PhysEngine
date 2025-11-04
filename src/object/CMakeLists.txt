include_directories("${ENGINE_INCLUDE_DIR}")
include_directories("${TASKFLOW_INCLUDE_DIR}")
use_backend_warpper_local()

add_lib(EngineCudaObject ${CMAKE_CURRENT_SOURCE_DIR})
use_cuda_local(EngineCudaObject)

# set_target_properties(EngineCommon PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(EngineCommon PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)