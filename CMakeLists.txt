cmake_minimum_required(VERSION 3.18)

#project name
project(PhysEngine)

SET(CMAKE_CXX_STANDARD 17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# add_definitions(-DUSE_DOUBLE)
add_definitions(-DGLOBALBENCHMARK)

option(ENABLE_EXTENDED_ALIGNED_STORAGE "Enable extended aligned storage" ON)

if (ENABLE_EXTENDED_ALIGNED_STORAGE)
    add_compile_definitions("_ENABLE_EXTENDED_ALIGNED_STORAGE")
else()
    add_compile_definitions("_DISABLE_EXTENDED_ALIGNED_STORAGE")
endif()


set(ENGINE_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/src)
set(TASKFLOW_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/3rdparty/taskflow)
set(ENGINE_PATH ${CMAKE_SOURCE_DIR})
set(PROJECT_PATH ${CMAKE_SOURCE_DIR})


include(./cmake/functions.cmake)
# init_path()
# use_taskflow_local()
# use_backend_warpper_local()

# init_variables()
set(CMAKE_CUDA_ARCHITECTURES "75")
enable_language(CUDA)
find_package(CUDAToolkit)
message(STATUS"COMMON")
add_subdirectory(src/common)
message(STATUS"VIEWER")
add_subdirectory(src/cuda_viewer)
message(STATUS"OBJECT")
add_subdirectory(src/object)
message(STATUS"EXAMPLES")
add_subdirectory(examples)
