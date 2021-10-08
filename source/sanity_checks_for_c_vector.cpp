//
// Created by andreas on 02.10.21.
//
#include "miscellaneous/templates/c_vector.h"
#include "materials/material.h"
#include "materials/material_builder.h"
#include "miscellaneous/image_buffer.h"
#include "objects/sphere.h"
#include <memory>
#include <fstream>
int main(){
    auto test_vector= c_vector<3>{0.0, 4.0, 3.0};
    auto test_vector2= c_vector<3>{3.0, 3.0, 3.0};
    auto factor = test_vector*3;
    auto dot_product = test_vector*test_vector2;
    auto addition = test_vector + test_vector2;
    auto subtraction = test_vector - test_vector2;
    auto normalized = test_vector.normalize();
    std::cout << factor << std::endl;
    std::cout << dot_product << std::endl;
    std::cout << addition << std::endl;
    std::cout << subtraction << std::endl;
    std::cout << normalized << std::endl;

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
    
    auto material = Material("example", builder);
    std::unique_ptr<IReflectionCoefficients> i_material = std::make_unique<Material>(material);
    //std::shared_ptr<IMaterialBuilder> i_material_setter = ;
    //std::shared_ptr<IReflectionCoefficients> i_material = reinterpret_cast<IReflectionCoefficients> (*i_material_setter);
    std::cout <<  " out  " << i_material->shininess() << std::endl;
    auto width = 1920;
    auto height = 1080;
    auto image = ImageBuffer(width, height);
    auto buffer = image.buffer();
    std::cout << buffer[1] << std::endl;

    std::ofstream ofs; // save the framebuffer to file
    ofs.open("./sample.ppm");
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (size_t i = 0; i < height*width; ++i) {
        for (size_t j = 0; j<3; j++) {
            ofs << (char)(255 * std::max(0.f, std::min(1.f, buffer[i][j])));
        }
    }
    ofs.close();
    auto center = c_vector3{1.2,0.0, 0.0};
    auto radius = 1.5;
    auto sphere = Sphere(center, radius);
    std:: cout << sphere.center() << std::endl;
    return 0;
}

