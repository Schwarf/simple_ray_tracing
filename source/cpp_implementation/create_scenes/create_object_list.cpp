//
// Created by andreas on 21.11.21.
//

#include "create_object_list.h"

ObjectList create_object_list()
{
	float almost_zero = 1.e-7;
	auto sphere_center = Point3D{5, 3., -16};
	float sphere_radius = 2;
	auto sphere = Sphere(sphere_center, sphere_radius);
	auto red_builder = MaterialBuilder();
	red_builder.set_specular_coefficient(0.3);
	red_builder.set_diffuse_coefficient(0.6);
	red_builder.set_ambient_coefficient(0.3);
	red_builder.set_shininess(50.);
	red_builder.set_rgb_color(Color{0.7, 0.4, 0.4});
	red_builder.set_refraction_index(1.0);
	red_builder.set_transparency(almost_zero);
	sphere.set_material(std::make_unique<Material>(Material("red_sphere", red_builder)));

	auto sphere_center2 = Point3D{1.5, 2.5, -18};
	auto sphere_radius2 = 2;
	auto sphere2 = Sphere(sphere_center2, sphere_radius2);
	auto blue_builder = MaterialBuilder();
	blue_builder.set_specular_coefficient(0.3);
	blue_builder.set_diffuse_coefficient(0.6);
	blue_builder.set_ambient_coefficient(0.3);
	blue_builder.set_shininess(10);
	blue_builder.set_rgb_color(Color{0.4, 0.4, 0.7});
	blue_builder.set_refraction_index(1.0);
	blue_builder.set_transparency(almost_zero);
	sphere2.set_material(std::make_unique<Material>(Material("blue_sphere", blue_builder)));


	auto sphere_center3 = Point3D{4.5, -1.5, -20};
	auto sphere_radius3 = 4;
	auto sphere3 = Sphere(sphere_center3, sphere_radius3);
	auto green_builder = MaterialBuilder();
	green_builder.set_specular_coefficient(0.3);
	green_builder.set_diffuse_coefficient(0.6);
	green_builder.set_ambient_coefficient(0.3);
	green_builder.set_shininess(10.);
	green_builder.set_rgb_color(Color{0.4, 0.7, 0.4});
	green_builder.set_refraction_index(1.0);
	green_builder.set_transparency(almost_zero);
	sphere3.set_material(std::make_unique<Material>(Material("green_sphere", green_builder)));

	auto sphere_center4 = Point3D{-4.5, -1.5, -16};
	auto sphere_radius4 = 2.5;
	auto sphere4 = Sphere(sphere_center4, sphere_radius4);
	auto glass_builder = MaterialBuilder();
	glass_builder.set_specular_coefficient(0.5);
	glass_builder.set_diffuse_coefficient(almost_zero);
	glass_builder.set_ambient_coefficient(0.1);
	glass_builder.set_shininess(125.);
	glass_builder.set_rgb_color(Color{0.5, 0.5, 0.5});
	glass_builder.set_refraction_index(1.5);
	glass_builder.set_transparency(0.8);
	sphere4.set_material(std::make_unique<Material>(Material("glass_sphere", glass_builder)));


	auto sphere_center5 = Point3D{-3.5, 3.5, -15};
	auto sphere_radius5 = 1.5;
	auto sphere5 = Sphere(sphere_center5, sphere_radius5);
	auto builder5 = MaterialBuilder();
	builder5.set_specular_coefficient(10.0);
	builder5.set_diffuse_coefficient(almost_zero);
	builder5.set_ambient_coefficient(0.8);
	builder5.set_shininess(1200.);
	builder5.set_rgb_color(Color{0.39, 0.3, 0.3});
	builder5.set_refraction_index(1.);
	builder5.set_transparency(almost_zero);
	sphere5.set_material(std::make_unique<Material>(Material("mirror_sphere", builder5)));


	auto object_list = ObjectList();
	std::shared_ptr<ITargetObject> red_sphere = std::make_shared<Sphere>(sphere);
	std::shared_ptr<ITargetObject> blue_sphere = std::make_shared<Sphere>(sphere2);
	std::shared_ptr<ITargetObject> green_sphere = std::make_shared<Sphere>(sphere3);
	std::shared_ptr<ITargetObject> glass_sphere = std::make_shared<Sphere>(sphere4);
	std::shared_ptr<ITargetObject> mirror_sphere = std::make_shared<Sphere>(sphere5);
	object_list.add_object(red_sphere);
	object_list.add_object(blue_sphere);
	object_list.add_object(green_sphere);
	object_list.add_object(glass_sphere);
	object_list.add_object(mirror_sphere);
	return object_list;

}
