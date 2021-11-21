//
// Created by andreas on 21.11.21.
//

#include "create_object_list.h"

ObjectList create_object_list()
{
	auto sphere_center = c_vector3{5, 3., -16};
	float sphere_radius = 2;
	auto sphere = Sphere(sphere_center, sphere_radius);
	auto builder1 = MaterialBuilder();
	builder1.set_specular_reflection(0.3);
	builder1.set_diffuse_reflection(0.6);
	builder1.set_ambient_reflection(0.3);
	builder1.set_shininess(0.00001);
	builder1.set_rgb_color(c_vector3{0.7, 0.4, 0.4});
	builder1.set_refraction_coefficient(1.0);
	builder1.set_specular_exponent(50.);
	sphere.set_material(std::make_unique<Material>(Material("red_sphere", builder1)));

	auto sphere_center2 = c_vector3{1.5, 2.5, -18};
	auto sphere_radius2 = 2;
	auto sphere2 = Sphere(sphere_center2, sphere_radius2);
	auto builder2 = MaterialBuilder();
	builder2.set_specular_reflection(0.3);
	builder2.set_diffuse_reflection(0.6);
	builder2.set_ambient_reflection(0.3);
	builder2.set_shininess(0.00001);
	builder2.set_rgb_color(c_vector3{0.4, 0.4, 0.7});
	builder2.set_refraction_coefficient(1.0);
	builder2.set_specular_exponent(10.);
	sphere2.set_material(std::make_unique<Material>(Material("blue_sphere", builder2)));


	auto sphere_center3 = c_vector3{4.5, -1.5, -20};
	auto sphere_radius3 = 4;
	auto sphere3 = Sphere(sphere_center3, sphere_radius3);
	auto builder3 = MaterialBuilder();
	builder3.set_specular_reflection(0.3);
	builder3.set_diffuse_reflection(0.6);
	builder3.set_ambient_reflection(0.3);
	builder3.set_shininess(0.00001);
	builder3.set_rgb_color(c_vector3{0.4, 0.7, 0.4});
	builder3.set_refraction_coefficient(1.0);
	builder3.set_specular_exponent(50.);
	sphere3.set_material(std::make_unique<Material>(Material("green_sphere", builder3)));

	auto sphere_center4 = c_vector3{-4.5, -1.5, -16};
	auto sphere_radius4 = 2.5;
	auto sphere4 = Sphere(sphere_center4, sphere_radius4);
	auto builder4 = MaterialBuilder();
	builder4.set_specular_reflection(0.5);
	builder4.set_diffuse_reflection(0.000001);
	builder4.set_ambient_reflection(0.1);
	builder4.set_shininess(0.8);
	builder4.set_rgb_color(c_vector3{0.6, 0.7, 0.8});
	builder4.set_refraction_coefficient(1.5);
	builder4.set_specular_exponent(125.);
	sphere4.set_material(std::make_unique<Material>(Material("glass_sphere", builder4)));


	auto sphere_center5 = c_vector3{-3.5, 3.5, -15};
	auto sphere_radius5 = 1.5;
	auto sphere5 = Sphere(sphere_center5, sphere_radius5);
	auto builder5 = MaterialBuilder();
	builder5.set_specular_reflection(10.0);
	builder5.set_diffuse_reflection(0.000001);
	builder5.set_ambient_reflection(0.8);
	builder5.set_shininess(0.00001);
	builder5.set_rgb_color(c_vector3{0.999, 0.999, 0.999});
	builder5.set_refraction_coefficient(1.);
	builder5.set_specular_exponent(1200.);
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
