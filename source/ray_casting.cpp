//
// Created by andreas on 03.10.21.
//

#include "cpp_implementation/c_vector.h"
#include "cpp_implementation/material.h"
#include "cpp_implementation/material_builder.h"
#include "cpp_implementation/image_buffer.h"
#include "cpp_implementation/sphere.h"
#include "cpp_implementation/ray.h"
#include <memory>
#include <fstream>
#include <limits>
#include <cmath>

c_vector3 cast_ray(const IRay &ray, const Sphere &sphere) {
    float sphere_dist = 10000.;
    if (!sphere.does_ray_intersect(ray, sphere_dist)) {
        return c_vector3{0.2, 0.7, 0.8}; // background color
    }
    return c_vector3{0.4, 0.4, 0.3};
}

void render(const Sphere &sphere) {
    auto width = 1024;
    auto height = 768;
    auto image = ImageBuffer(width, height);
    auto buffer = image.buffer();
    auto f_width = static_cast<float>(width);
    auto f_height = static_cast<float>(height);
    std::vector<c_vector3> imagebuffer(width * height);
    auto image_buffer = ImageBuffer(width, height);
    for (size_t j = 0; j < height; j++) {
        for (size_t i = 0; i < width; i++) {
            float x_direction = (static_cast<float>(i) + 0.5) - f_width/2.;
            float y_direction = -(static_cast<float>(j) + 0.5) + f_height/2.;
            float z_direction = -f_height/(2.*tan(M_PI/6.));
            c_vector3 direction = c_vector3{x_direction, y_direction, z_direction}.normalize();
            c_vector3 origin = c_vector3{0, 0, 0};
            auto ray = Ray(origin, direction);
            image_buffer.set_pixel_value(i, j, cast_ray(ray, sphere) );
            //imagebuffer[i + j * width] = cast_ray(ray, sphere);
        }
    }
    std::cout << "Hallo" << std::endl;
    std::ofstream ofs;
    ofs.open("./sphere.ppm");
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (size_t i = 0; i < height*width; ++i) {
        for (size_t j = 0; j<3; j++) {
            //ofs << (char)(255 * std::max(0.f, std::min(1.f, imagebuffer[i][j])));
            ofs << static_cast<char>(image_buffer.get_pixel(i)[j]);

        }
    }
    ofs.close();

}


int main() {
    auto sphere_center = c_vector3{-3.2, 0., -16};
    float sphere_radius = -2;
    auto sphere = Sphere(sphere_center, sphere_radius);
    render(sphere);

    return 0;
}