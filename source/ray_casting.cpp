//
// Created by andreas on 03.10.21.
//
#include <memory>
#include <fstream>
#include <cmath>

#include "miscellaneous/templates/c_vector.h"
#include "materials/material.h"
#include "materials/material_builder.h"
#include "miscellaneous/image_buffer.h"
#include "objects/sphere.h"
#include "rays/ray.h"
#include "rays/light_source.h"
#include "rays/ray_interactions.h"
#include "cpp_implementation/rays/scene_illumination.h"
#include "cpp_implementation/objects/object_list.h"


c_vector3 cast_ray(const IRay &ray, ObjectList &object_list, SceneIllumination &scene_illumination, size_t recursion_depth = 0)
{
	auto hit_point = c_vector3{0, 0, 0};
	auto hit_normal = c_vector3{0, 0, 0};
	auto object = object_list.get_object_hit_by_ray(ray, hit_normal, hit_point);
	if (object == nullptr || recursion_depth > 4) {
		return scene_illumination.background_color();
	}
	auto interaction = RayInteractions();
	auto reflected_ray = Ray(hit_point, interaction.reflection(ray.direction_normalized(), hit_normal).normalize());
	recursion_depth++;
	auto reflected_color = cast_ray(reflected_ray, object_list, scene_illumination, recursion_depth);

	float diffuse_intensity = 0.f;
	float specular_intensity = 0.f;
	std::shared_ptr<ILightSource> light_source = nullptr;
	for (size_t ls_index = 0; ls_index < scene_illumination.number_of_light_sources(); ++ls_index) {
		light_source = scene_illumination.light_source(ls_index);
		auto light_direction = (light_source->position() - hit_point).normalize();
		auto light_source_ray = Ray(hit_point, light_direction);
		auto shadow_point = c_vector3 {0.,0.,0.};
		auto shadow_normal = c_vector3 {0.,0.,0.};
		auto object_in_shadow = object_list.get_object_hit_by_ray(light_source_ray, shadow_normal, shadow_point);
		auto distance_shadow_point_to_point = (shadow_point - hit_point).norm();
		auto distance_light_source_to_point = (light_source->position() - hit_point).norm();
		if(object_in_shadow != nullptr && distance_shadow_point_to_point < distance_light_source_to_point)
		{
			continue;
		}

		diffuse_intensity += light_source->intensity() * std::max(0.f, light_direction * hit_normal);
		specular_intensity +=
			std::pow(std::max(0.f, interaction.reflection(light_direction, hit_normal) * ray.direction_normalized()),
					 object->get_material()->specular_exponent()) * light_source->intensity();
	}

	auto diffuse_reflection =
		object->get_material()->rgb_color() * diffuse_intensity * object->get_material()->diffuse_reflection();
	auto specular_reflection = specular_intensity * c_vector3{1, 1, 1} * object->get_material()->specular_reflection();
	auto ambient_reflection = reflected_color*object->get_material()->ambient_reflection();

	return diffuse_reflection + specular_reflection + ambient_reflection;

}

ObjectList create_object_list()
{
	auto sphere_center = c_vector3{5, 3., -16};
	float sphere_radius = 2;
	auto sphere = Sphere(sphere_center, sphere_radius);
	auto builder1 = MaterialBuilder();
	builder1.set_specular_reflection(0.3);
	builder1.set_diffuse_reflection(0.6);
	builder1.set_ambient_reflection(0.3);
	builder1.set_shininess(0.3);
	builder1.set_rgb_color(c_vector3{0.7, 0.4, 0.4});
	builder1.set_refraction_coefficient(0.1);
	builder1.set_specular_exponent(50.);
	sphere.set_material(std::make_unique<Material>(Material("red_sphere", builder1)));

	auto sphere_center2 = c_vector3{1.5, 2.5, -18};
	auto sphere_radius2 = 2;
	auto sphere2 = Sphere(sphere_center2, sphere_radius2);
	auto builder2 = MaterialBuilder();
	builder2.set_specular_reflection(0.3);
	builder2.set_diffuse_reflection(0.6);
	builder2.set_ambient_reflection(0.3);
	builder2.set_shininess(0.3);
	builder2.set_rgb_color(c_vector3{0.4, 0.4, 0.7});
	builder2.set_refraction_coefficient(0.1);
	builder2.set_specular_exponent(120.);
	sphere2.set_material(std::make_unique<Material>(Material("blue_sphere", builder2)));


	auto sphere_center3 = c_vector3{4.5, -1.5, -20};
	auto sphere_radius3 = 4;
	auto sphere3 = Sphere(sphere_center3, sphere_radius3);
	auto builder3 = MaterialBuilder();
	builder3.set_specular_reflection(0.3);
	builder3.set_diffuse_reflection(0.6);
	builder3.set_ambient_reflection(0.3);
	builder3.set_shininess(0.3);
	builder3.set_rgb_color(c_vector3{0.4, 0.7, 0.4});
	builder3.set_refraction_coefficient(0.1);
	builder3.set_specular_exponent(130.);
	sphere3.set_material(std::make_unique<Material>(Material("green_sphere", builder3)));

	auto sphere_center4 = c_vector3{-4.5, -1.5, -12};
	auto sphere_radius4 = 2.5;
	auto sphere4 = Sphere(sphere_center4, sphere_radius4);
	auto builder4 = MaterialBuilder();
	builder4.set_specular_reflection(0.5);
	builder4.set_diffuse_reflection(0.000001);
	builder4.set_ambient_reflection(0.1);
	builder4.set_shininess(0.8);
	builder4.set_rgb_color(c_vector3{0.6, 0.7, 0.8});
	builder4.set_refraction_coefficient(0.1);
	builder4.set_specular_exponent(1200.);
	sphere4.set_material(std::make_unique<Material>(Material("glass_sphere", builder4)));

	auto object_list = ObjectList();
	std::shared_ptr<ITargetObject> red_sphere = std::make_shared<Sphere>(sphere);
	std::shared_ptr<ITargetObject> blue_sphere = std::make_shared<Sphere>(sphere2);
	std::shared_ptr<ITargetObject> green_sphere = std::make_shared<Sphere>(sphere3);
	std::shared_ptr<ITargetObject> glass_sphere = std::make_shared<Sphere>(sphere4);
	object_list.add_object(red_sphere);
	object_list.add_object(blue_sphere);
	object_list.add_object(green_sphere);
	object_list.add_object(glass_sphere);
	return object_list;

}

SceneIllumination create_scene_illumination()
{
	auto light_source_position1 = c_vector3{-20, -20, 20};
	float light_intensity1 = 1.5;
	std::shared_ptr<ILightSource>
		light_source1 = std::make_shared<LightSource>(LightSource(light_source_position1, light_intensity1));

	auto light_source_position2 = c_vector3{30, -50, -25};
	float light_intensity2 = 1.8;
	std::shared_ptr<ILightSource>
		light_source2 = std::make_shared<LightSource>(LightSource(light_source_position2, light_intensity2));;

	auto light_source_position3 = c_vector3{30, 20, 30};
	float light_intensity3 = 1.7;
	std::shared_ptr<ILightSource>
		light_source3 = std::make_shared<LightSource>(LightSource(light_source_position3, light_intensity3));

	auto scene_illumination = SceneIllumination(light_source1);

	scene_illumination.set_background_color(c_vector3{0.2, 0.7, 0.8});
	scene_illumination.add_light_source(light_source2);
	scene_illumination.add_light_source(light_source3);
	return scene_illumination;
}

void render(ObjectList &object_list, SceneIllumination &scene_illumination)
{
	auto width = 1024;
	auto height = 768;
	auto image_buffer = ImageBuffer(width, height);

	for (int height_index = 0; height_index < height; height_index++) {
		for (int width_index = 0; width_index < width; width_index++) {
			float x_direction = float(width_index) - float(width) / 2.f;
			float y_direction = float(height_index) - float(height) / 2.f;
			float z_direction = -float(height + width) / 2.f;
			c_vector3 direction = c_vector3{x_direction, y_direction, z_direction}.normalize();
			c_vector3 origin = c_vector3{0, 0, 0};
			auto ray = Ray(origin, direction);
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

	return 0;
}