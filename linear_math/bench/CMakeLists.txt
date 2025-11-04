
# add_executable(PureEigen pure_eigen.cpp)
add_executable(WrapperTest wrapper_test.cpp)

target_link_libraries(WrapperTest LinearMath)