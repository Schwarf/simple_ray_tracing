//
// Created by andreas on 22.11.21.
//

#include "camera.h"

Camera::Camera(int image_width, int image_height, float viewport_width, float focal_length)
{
	image_width_ = image_width;
	image_height_ = image_height;
	focal_length_ = focal_length;
	aspect_ratio_ = static_cast<float>(image_width) / static_cast<float>(image_height);
	float viewport_height = viewport_width / aspect_ratio_;
	horizontal_direction_[0] = viewport_width;
	vertical_direction_[1] = viewport_height;
	lower_left_corner_ =
		origin_ - horizontal_direction_ / 2.f - vertical_direction_ / 2.f - c_vector3{0, 0, focal_length};
	image_buffer_ = std::make_shared<ImageBuffer>(ImageBuffer(image_width_, image_height_));
}

std::shared_ptr<IRay> Camera::get_ray(float width_coordinate, float height_coordinate)
{
	auto direction = lower_left_corner_ + width_coordinate*horizontal_direction_ + height_coordinate*vertical_direction_ -origin_;
	return std::make_shared<Ray>(Ray(origin_, direction));
}
int Camera::image_width()
{
	return image_width_;
}
int Camera::image_height()
{
	return image_height_;
}
float Camera::aspect_ratio()
{
	return aspect_ratio_;
}
float Camera::focal_length()
{
	return focal_length_;
}

void Camera::render_image(std::shared_ptr<IObjectList> &objects_in_scene, std::shared_ptr<ISceneIllumination> &scene_illumination)
{
	#pragma omp parallel for
	for (int height_index = 0; height_index < image_height_; height_index++) {
		for (int width_index = 0; width_index < image_width_; width_index++) {
			auto u = double(width_index)/(image_width_-1);
			auto v = double(height_index)/(image_height_-1);
			auto ray = get_ray(u, v);
			auto color_values = get_pixel_color(ray, objects_in_scene, scene_illumination);
			image_buffer_->set_pixel_value(width_index, height_index, color_values);
		}
	}


}
c_vector3 Camera::get_pixel_color(std::shared_ptr<IRay> &ray,
								  std::shared_ptr<IObjectList> &objects_in_scene,
								  std::shared_ptr <ISceneIllumination> &scene_illumination,
								  size_t recursion_depth)
{
	auto hit_point = c_vector3{0, 0, 0};
	auto hit_normal = c_vector3{0, 0, 0};
	auto air_refraction_index = 1.f;
	auto object = objects_in_scene->get_object_hit_by_ray(ray, hit_normal, hit_point);
	if (object == nullptr || recursion_depth > 1) {
		return scene_illumination->background_color();
	}
	auto interaction = RayInteractions();
	std::shared_ptr<IRay> reflected_ray = std::make_shared<Ray>(Ray(hit_point, interaction.reflection(ray->direction_normalized(), hit_normal).normalize()));
	std::shared_ptr<IRay> refracted_ray = std::make_shared<Ray>(Ray(hit_point,
																	interaction.refraction(ray->direction_normalized(),
																						   hit_normal,
																						   object->get_material()->refraction_coefficient(),
																						   air_refraction_index).normalize()));
	recursion_depth++;
	auto reflected_color = get_pixel_color(reflected_ray, objects_in_scene, scene_illumination, recursion_depth);
	auto refracted_color = get_pixel_color(refracted_ray, objects_in_scene, scene_illumination, recursion_depth);

	float diffuse_intensity = 0.f;
	float specular_intensity = 0.f;
	std::shared_ptr<ILightSource> light_source = nullptr;
	for (size_t ls_index = 0; ls_index < scene_illumination->number_of_light_sources(); ++ls_index) {
		light_source = scene_illumination->light_source(ls_index);
		auto light_direction = (light_source->position() - hit_point).normalize();
		std::shared_ptr<IRay> light_source_ray = std::make_shared<Ray>(Ray(hit_point, light_direction));
		auto shadow_point = c_vector3{0., 0., 0.};
		auto shadow_normal = c_vector3{0., 0., 0.};
		auto object_in_shadow = objects_in_scene->get_object_hit_by_ray(light_source_ray, shadow_normal, shadow_point);
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
std::shared_ptr<IImageBuffer> Camera::get_image_buffer()
{
	return image_buffer_;
}
