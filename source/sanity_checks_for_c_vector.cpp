//
// Created by andreas on 02.10.21.
//
#include "cpp_implementation/c_vector.h"

int main(){
    auto test_vector= c_vector<3>{0.0, 4.0, 3.0};
    auto test_vector2= c_vector<3>{3.0, 3.0, 3.0};
    auto factor = test_vector*3;
    auto dot_product = test_vector*test_vector2;
    auto addition = test_vector + test_vector2;
    auto subtraction = test_vector - test_vector2;
    auto normalized = test_vector.normalize();
    auto perpendicular = cross_product(test_vector, test_vector2);
    std::cout << factor << std::endl;
    std::cout << dot_product << std::endl;
    std::cout << addition << std::endl;
    std::cout << subtraction << std::endl;
    std::cout << normalized << std::endl;
    std::cout << perpendicular << std::endl;
    return 0;
}
