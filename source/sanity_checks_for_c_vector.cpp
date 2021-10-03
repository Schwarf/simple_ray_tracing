//
// Created by andreas on 02.10.21.
//
#include "cpp_implementation/c_vector.h"
#include "cpp_implementation/material.h"
#include "cpp_implementation/material_builder.h"
#include <memory>
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

    auto builder = MaterialBuilder();
    float test = 0.3;
    float test2 =0.1;
    builder.set_specular_reflection(test);
    builder.set_diffuse_reflection(test);
    builder.set_ambient_reflection(test);
    builder.set_shininess(test);
    builder.set_blue_value(test2);
    builder.set_red_value(test2);
    builder.set_green_value(test2);
    builder.set_refraction_coefficient(test2);
    builder.set_specular_exponent(test2);
    
    auto material = Material("smaple", builder);
    std::unique_ptr<IReflectionCoefficients> i_material = std::make_unique<Material>(material);
    //std::shared_ptr<IMaterialBuilder> i_material_setter = ;
    //std::shared_ptr<IReflectionCoefficients> i_material = reinterpret_cast<IReflectionCoefficients> (*i_material_setter);
    std::cout <<  " out  " << i_material->shininess() << std::endl;


    return 0;
}

