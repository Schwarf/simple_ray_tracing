//
// Created by andreas on 03.10.21.
//

#include "cpp_implementation/c_vector.h"
#include "materials/material.h"
#include "materials/material_builder.h"
#include "cpp_implementation/image_buffer.h"
#include "objects/sphere.h"
#include "cpp_implementation/ray.h"
#include "cpp_implementation/light_source.h"
#include <memory>
#include <fstream>
#include "cpp_implementation/ray_interactions.h"
#include <cmath>
#include "cpp_implementation/objects/rectangle.h"

c_vector3 cast_ray(const IRay &ray, Sphere &sphere, Rectangle & rectangle, const LightSource &light_source) {
    float sphere_dist = 10000.;
    auto hit_point = c_vector3 {0,0,0};
    if (!sphere.does_ray_intersect(ray, sphere_dist, hit_point)) {
        return c_vector3{0.5, 0.5, 0.5}; // background color
    }
    auto light_direction = (light_source.position() - hit_point ).normalize();
    auto normal = (hit_point -sphere.center()).normalize();
    auto interaction = RayInteractions();
    auto diffuse_intensity = light_source.intensity() * std::max(0.f, light_direction * normal);
    auto specular_intensity = std::pow(std::max(0.f, interaction.reflection(light_direction, normal)*light_direction),
                                       sphere.get_material()->specular_exponent())*light_source.intensity();
    auto diffuse_reflection = sphere.get_material()->rgb_color() * diffuse_intensity * sphere.get_material()->diffuse_reflection();
    auto specular_reflection = specular_intensity*c_vector3{1,1,1}*sphere.get_material()->specular_reflection();
    return diffuse_reflection + specular_reflection;
}

void render(Sphere &sphere, Rectangle & rectangle, const LightSource &light_source) {
    auto width = 1024;
    auto height = 768;
    auto f_width = static_cast<float>(width);
    auto f_height = static_cast<float>(height);
    auto image_buffer = ImageBuffer(width, height);
    for (size_t height_index = 0; height_index < height; height_index++) {
        for (size_t width_index = 0; width_index < width; width_index++) {
            float x_direction = (static_cast<float>(width_index) + 0.5) - f_width / 2.;
            float y_direction = -(static_cast<float>(height_index) + 0.5) + f_height / 2.;
            float z_direction = -f_height / (2. * tan(M_PI / 6.));
            c_vector3 direction = c_vector3{x_direction, y_direction, z_direction}.normalize();
            c_vector3 origin = c_vector3{0, 0, 0};
            auto ray = Ray(origin, direction);
            image_buffer.set_pixel_value(width_index, height_index, cast_ray(ray, sphere, rectangle, light_source));

        }
    }
    std::cout << "Hallo" << std::endl;
    std::ofstream ofs;
    ofs.open("./sphere.ppm");
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (size_t pixel_index = 0; pixel_index < height * width; ++pixel_index) {
        for (size_t color_index = 0; color_index < 3; color_index++) {
            ofs << static_cast<char>(image_buffer.get_rgb_pixel(pixel_index)[color_index]);

        }
    }
    ofs.close();

}

void push(std::vector<int> * vec)
{
    vec->push_back(2);
}

int main() {
    auto bottom_left_position = c_vector3{5.2, -4., -10};
    auto height_vector = c_vector3 {2,0,0};
    auto width_vector =  c_vector3 {0,2,0};
    auto normal = c_vector3 {1,1,1};
    auto rectangle = Rectangle(width_vector, height_vector, bottom_left_position, normal);

    auto sphere_center = c_vector3{-5.2, 3., -16};
    float sphere_radius = 2;
    auto sphere = Sphere(sphere_center, sphere_radius);

    auto builder = MaterialBuilder();
    float test = 0.3;
    float test2 = 0.1;
    builder.set_specular_reflection(0.6);
    builder.set_diffuse_reflection(0.3);
    builder.set_ambient_reflection(test);
    builder.set_shininess(test);
    builder.set_rgb_color(c_vector3{0.7, 0.4, 0.4});
    builder.set_refraction_coefficient(test2);
    builder.set_specular_exponent(50.);

    auto material = Material("sample", builder);
    sphere.set_material(std::make_unique<Material>(material));

    rectangle.set_material(std::make_unique<Material>(material));
    auto light_source_position = c_vector3{-20, 20, 20};
    float light_intensity = 2.5;
    auto light_source = LightSource(light_source_position, light_intensity);
    render(sphere, rectangle, light_source);

    return 0;
}