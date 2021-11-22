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
#include "camera/camera.h"


int main()
{

	auto scene_illumination = create_scene_illumination();
	auto object_list = create_object_list();
	std::shared_ptr<IObjectList> objects_in_scene(std::shared_ptr<ObjectList>(), & object_list);
	std::shared_ptr<ISceneIllumination> scene_lights(std::shared_ptr<SceneIllumination>(), & scene_illumination);
	auto image_width = 1920;
	auto image_height = 1080;
	auto focal_length = 1.f;
	auto viewport_width = 2.f;

	auto camera = Camera(image_width, image_height, viewport_width, focal_length);
	camera.render_image(objects_in_scene, scene_lights);
	auto image_buffer = camera.get_image_buffer();
	std::ofstream ofs;
	ofs.open("./sphere.ppm");
	ofs << "P6\n" << image_width << " " << image_height << "\n255\n";
	for (size_t pixel_index = 0; pixel_index < image_height * image_width; ++pixel_index) {
		for (size_t color_index = 0; color_index < 3; color_index++) {
			ofs << static_cast<char>(image_buffer->get_rgb_pixel(pixel_index)[color_index]);

		}
	}
	ofs.close();



	//render(object_list, scene_illumination);
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