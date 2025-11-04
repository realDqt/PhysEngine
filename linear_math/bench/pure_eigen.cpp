#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::Vector3f v(1.0, 1.0, 1.0);
    std::cout<<v.transpose()<<std::endl;
}