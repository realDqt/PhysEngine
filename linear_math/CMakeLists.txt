cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(LinearMath)

SET(CMAKE_CXX_STANDARD 17)

# Check if linear_math is being used directly or via add_subdirectory
set(LINEAR_MATH_MASTER_PROJECT OFF)
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(LINEAR_MATH_MASTER_PROJECT ON)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

set(EIGEN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/eigen)
set(ENOKI_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/enoki/include)

# add_definitions(-DEIGEN_BACKEND) # use eigen as backend
add_definitions(-DENOKI_BACKEND) # use eigen as backend

add_library(LinearMath INTERFACE)
target_include_directories(LinearMath INTERFACE include)
target_include_directories(LinearMath INTERFACE ${EIGEN_INCLUDE_DIR} ${ENOKI_INCLUDE_DIR})

if(LINEAR_MATH_MASTER_PROJECT)
    add_subdirectory(bench)
endif()