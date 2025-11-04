include_directories("${ENGINE_INCLUDE_DIR}")
include_directories("${TASKFLOW_INCLUDE_DIR}")

link_libraries(EngineCudaCommon)
link_libraries(EngineCudaViewer)
link_libraries(EngineCudaObject)
init_variables()
use_taskflow()
use_backend_warpper_local()
# include_and_link()

add_exe(SmokeRenderDemo ${CMAKE_CURRENT_SOURCE_DIR})
use_freeglut_local(SmokeRenderDemo)
use_cuda_local_exe(SmokeRenderDemo)