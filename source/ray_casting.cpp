//
// Created by andreas on 03.10.21.
//
#include <memory>
#include <fstream>
#include <cmath>
#include "miscellaneous/templates/c_vector.h"
#include "miscellaneous/image_buffer.h"
#include "rays/ray.h"
#include "rays/light_source.h"
#include "rays/ray_interactions.h"
#include "miscellaneous/quadratic_equation.h"
#include "miscellaneous/cubic_equation.h"
#include "create_scenes/create_object_list.h"
#include "create_scenes/create_scene_illumination.h"
#include "rays/camera.h"

c_vector3 cast_ray(std::shared_ptr<IRay> ray,
				   ObjectList &object_list,
				   SceneIllumination &scene_illumination,
				   size_t recursion_depth = 0)
{
	auto hit_point = c_vector3{0, 0, 0};
	auto hit_normal = c_vector3{0, 0, 0};
	auto air_refraction_index = 1.f;
	auto object = object_list.get_object_hit_by_ray(ray, hit_normal, hit_point);
	if (object == nullptr || recursion_depth > 1) {
		return scene_illumination.background_color();
	}
	auto interaction = RayInteractions();
	std::shared_ptr<IRay> reflected_ray = std::make_shared<Ray>(Ray(hit_point, interaction.reflection(ray->direction_normalized(), hit_normal).normalize()));
	std::shared_ptr<IRay> refracted_ray = std::make_shared<Ray>(Ray(hit_point,
							 interaction.refraction(ray->direction_normalized(),
													hit_normal,
													object->get_material()->refraction_coefficient(),
													air_refraction_index).normalize()));
	recursion_depth++;
	auto reflected_color = cast_ray(reflected_ray, object_list, scene_illumination, recursion_depth);
	auto refracted_color = cast_ray(refracted_ray, object_list, scene_illumination, recursion_depth);

	float diffuse_intensity = 0.f;
	float specular_intensity = 0.f;
	std::shared_ptr<ILightSource> light_source = nullptr;
	for (size_t ls_index = 0; ls_index < scene_illumination.number_of_light_sources(); ++ls_index) {
		light_source = scene_illumination.light_source(ls_index);
		auto light_direction = (light_source->position() - hit_point).normalize();
		std::shared_ptr<IRay> light_source_ray = std::make_shared<Ray>(Ray(hit_point, light_direction));
		auto shadow_point = c_vector3{0., 0., 0.};
		auto shadow_normal = c_vector3{0., 0., 0.};
		auto object_in_shadow = object_list.get_object_hit_by_ray(light_source_ray, shadow_normal, shadow_point);
		auto distance_shadow_point_to_point = (shadow_point - hit_point).norm();
		auto distance_light_source_to_point = (light_source->position() - hit_point).norm();
		if (object_in_shadow != nullptr && distance_shadow_point_to_point < distance_light_source_to_point) {
			continue;
		}

		diffuse_intensity += light_source->intensity() * std::max(0.f, light_direction * hit_normal);
		specular_intensity +=
			std::pow(std::max(0.f, interaction.reflection(light_direction, hit_normal) * ray->direction_normalized()),
					 object->get_material()->specular_exponent()) * light_source->intensity();
	}

	auto diffuse_reflection =
		object->get_material()->rgb_color() * diffuse_intensity * object->get_material()->diffuse_reflection();
	auto specular_reflection = specular_intensity * c_vector3{1, 1, 1} * object->get_material()->specular_reflection();
	auto ambient_reflection = reflected_color * object->get_material()->ambient_reflection();
	auto refraction = refracted_color*object->get_material()->shininess();
	return diffuse_reflection + specular_reflection + ambient_reflection + refraction;

}



void render(ObjectList &object_list, SceneIllumination &scene_illumination)
{
	auto width = 1024;
	auto height = 768;
	auto image_buffer = ImageBuffer(width, height);
	auto camera = Camera(width, height, 2.0, 1.0);
	#pragma omp parallel for
	for (int height_index = 0; height_index < height; height_index++) {
		for (int width_index = 0; width_index < width; width_index++) {
			auto u = double(width_index)/(width-1);
			auto v = double(height_index)/(height-1);
			auto ray = camera.get_ray(u, v);
			auto color_values = cast_ray(ray, object_list, scene_illumination);
			image_buffer.set_pixel_value(width_index, height_index, color_values);

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

int main()
{
//	auto bottom_left_position = c_vector3{0.2, 3., -16};
//	auto height_vector = c_vector3{5, 0, 0};
//	auto width_vector = c_vector3{0, 5, 0};
//	auto normal = c_vector3{1, 1, 0};
//	auto rectangle = Rectangle(width_vector, height_vector, bottom_left_position, normal);



	auto scene_illumination = create_scene_illumination();
	auto object_list = create_object_list();


	render(object_list, scene_illumination);
	c_vector<3, double> coeff{1.0, 5.0, -5.0};
	c_vector<4, double> coefficient_cubic{1.0, -6.0, 11.0, -6.0};
	c_vector<4, double> coefficient_cubic2{1.0, -5.0, 8.0, -4.0};
	double epsilon = 1.e-10;
	auto q_equation = QuadraticEquation<double>(coeff, epsilon);
	auto cubic_equation = CubicEquation<double>(coefficient_cubic, epsilon);
	auto cubic_equation2 = CubicEquation<double>(coefficient_cubic2, epsilon);
	for(int i = 0; i < q_equation.number_of_solutions(); ++i)
	{
		std::cout << q_equation.solutions()[i] << std::endl;
	}
	for(int i = 0; i < cubic_equation.number_of_solutions(); ++i)
	{
		std::cout << cubic_equation.solutions()[i] << std::endl;
	}
	for(int i = 0; i < cubic_equation2.number_of_solutions(); ++i)
	{
		std::cout << cubic_equation2.solutions()[i] << std::endl;
	}

	return 0;
}