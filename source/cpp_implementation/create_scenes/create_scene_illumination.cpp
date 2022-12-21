//
// Created by andreas on 21.11.21.
//

#include "create_scene_illumination.h"

SceneIllumination create_scene_illumination()
{
	auto light_source_position1 = Point3D{-20, -20, 20};
	float light_intensity1 = 1.5;
	std::shared_ptr<ILightSource>
		light_source1 = std::make_shared<LightSource>(LightSource(light_source_position1, light_intensity1));

	auto light_source_position2 = Point3D{30, -50, -25};
	float light_intensity2 = 1.8;
	std::shared_ptr<ILightSource>
		light_source2 = std::make_shared<LightSource>(LightSource(light_source_position2, light_intensity2));;

	auto light_source_position3 = Point3D{30, 20, 30};
	float light_intensity3 = 1.7;
	std::shared_ptr<ILightSource>
		light_source3 = std::make_shared<LightSource>(LightSource(light_source_position3, light_intensity3));

	auto scene_illumination = SceneIllumination(light_source1);

	scene_illumination.add_light_source(light_source2);
	scene_illumination.add_light_source(light_source3);

	auto background_color1 = Color{0.2f, 0.7f, 0.8f};
	auto background_color2 = Color{1.f, 1.f, 1.f};
	scene_illumination.set_background_colors(background_color1, background_color2);

	auto almost_zero = 0.0000001f;
	auto ground_material_builder = MaterialBuilder();
	ground_material_builder.set_specular_coefficient(almost_zero);
	ground_material_builder.set_diffuse_coefficient(0.1);
	ground_material_builder.set_ambient_coefficient(0.05);
	ground_material_builder.set_shininess(almost_zero);
	ground_material_builder.set_rgb_color(Color{0.1, 0.5, 0.1});
	ground_material_builder.set_refraction_index(almost_zero);
	ground_material_builder.set_transparency(almost_zero);
	IMaterialPtr ground_material = std::make_shared<Material>(Material("ground_material", ground_material_builder));

	auto radius = 50.f;
	auto center = Point3D {0, radius, -1};
	auto ground_sphere = std::make_shared<Sphere>(Sphere(center, radius));
	ground_sphere->set_material(ground_material);

	scene_illumination.set_ground_sphere(ground_sphere);


	return scene_illumination;
}
