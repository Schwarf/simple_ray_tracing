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
		origin_ - horizontal_direction_ / 2.f - vertical_direction_ / 2.f - Point3D{0, 0, focal_length};
	image_buffer_ = std::make_shared<ImageBuffer>(ImageBuffer(image_width_, image_height_));
}

IRayPtr Camera::get_camera_ray(float width_coordinate, float height_coordinate)
{
	auto direction =
		lower_left_corner_ + width_coordinate * horizontal_direction_ + height_coordinate * vertical_direction_
			- origin_;
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

std::pair<float, float> Camera::get_pixel_coordinates(const size_t &width_index, const size_t &height_index) const
{
	float add_u = antialiasing_enabled_ ? UniformRandomNumberGenerator::get_random<float>(0.f, 1.f) : 0.f;
	float add_v = antialiasing_enabled_ ? UniformRandomNumberGenerator::get_random<float>(0.f, 1.f) : 0.f;
	auto u = (float(width_index) + add_u) / float(image_width_ - 1);
	auto v = (float(height_index) + add_v) / float(image_height_ - 1);
	return {u, v};
}



void Camera::render_image(const IObjectListPtr &objects_in_scene,
						  const ISceneIlluminationPtr &scene_illumination)
{
	size_t samples_per_pixel = 1;
	size_t recursion_depth = 2;
	if (antialiasing_enabled_)
		samples_per_pixel = 4;
#pragma omp parallel for
	for (int height_index = 0; height_index < image_height_; height_index++) {
		for (int width_index = 0; width_index < image_width_; width_index++) {
			Color color_values{0, 0, 0};
			for (size_t sample = 0; sample < samples_per_pixel; ++sample) {
				color_values += compute_one_pixel(width_index, height_index, objects_in_scene, scene_illumination, recursion_depth);
			}
			image_buffer_->set_pixel_value(width_index, height_index, color_values, samples_per_pixel);
		}
	}

}
Color Camera::get_pixel_color(const IRayPtr &camera_ray,
							  const IObjectListPtr &objects_in_scene,
							  const ISceneIlluminationPtr &scene_illumination,
							  size_t recursion_depth)
{
	IHitRecordPtr hit_record = std::make_shared<HitRecord>(HitRecord());
	auto air_refraction_index = 1.f;
	auto object = objects_in_scene->get_object_hit_by_ray(camera_ray, hit_record);
	if (object == nullptr || recursion_depth < 1) {
		auto
			mix_parameter =
			1.f / 2.f * ((camera_ray->direction_normalized()[0] + camera_ray->direction_normalized()[1]) / 2.f + 1.f);
		return scene_illumination->background_color(mix_parameter);
	}
	// Start recursion
	recursion_depth--;
	auto reflected_color = get_pixel_color(ray_interaction_.reflected_ray(camera_ray, hit_record),
										   objects_in_scene,
										   scene_illumination,
										   recursion_depth);
	auto refracted_color = get_pixel_color(ray_interaction_.refracted_ray(camera_ray, hit_record, air_refraction_index),
										   objects_in_scene,
										   scene_illumination,
										   recursion_depth);

	float diffuse_intensity = 0.f;
	float specular_intensity = 0.f;
	const auto hit_normal = hit_record->hit_normal();
	const auto hit_point = hit_record->hit_point();

	IHitRecordPtr shadow_hit_record = std::make_shared<HitRecord>(HitRecord());
	for (size_t ls_index = 0; ls_index < scene_illumination->number_of_light_sources(); ++ls_index) {
		const ILightSourcePtr light_source = scene_illumination->light_source(ls_index);
		const auto light_direction = (light_source->position() - hit_record->hit_point()).normalize();
		const IRayPtr light_source_ray = std::make_shared<Ray>(Ray(hit_point, light_direction));

		const auto object_in_shadow = objects_in_scene->get_object_hit_by_ray(light_source_ray, shadow_hit_record);
		const auto shadow_point = shadow_hit_record->hit_point();
		const auto distance_shadow_point_to_point = (shadow_point - hit_point).norm();
		const auto distance_light_source_to_point = (light_source->position() - hit_point).norm();
		if (object_in_shadow && distance_shadow_point_to_point < distance_light_source_to_point) {
			continue;
		}

		diffuse_intensity += light_source->intensity() * std::max(0.f, light_direction * hit_normal);
		const auto scalar_product = ray_interaction_.reflected_ray(light_source_ray, hit_record)->direction_normalized()
			* camera_ray->direction_normalized();
		specular_intensity +=
			std::pow(std::max(0.f, scalar_product),object->get_material()->shininess()) * light_source->intensity();
	}

	Color diffuse_color =
		object->get_material()->rgb_color() * diffuse_intensity * object->get_material()->diffuse();
	Color white = Color{1, 1, 1};
	Color specular_color = specular_intensity * white * object->get_material()->specular();
	Color ambient_color = reflected_color * object->get_material()->ambient();
	Color refraction_color = refracted_color * object->get_material()->transparency();
	return diffuse_color + specular_color + ambient_color + refraction_color;
}

IImageBufferPtr Camera::get_image_buffer()
{
	return image_buffer_;
}
void Camera::enable_antialiasing()
{
	antialiasing_enabled_ = true;
}
void Camera::disable_antialiasing()
{
	antialiasing_enabled_ = false;
}

bool Camera::antialiasing_enabled()
{
	return antialiasing_enabled_;
}

Color Camera::compute_one_pixel(const size_t &width_index, const size_t &height_index,
						const IObjectListPtr &objects_in_scene,
						const ISceneIlluminationPtr &scene_illumination,
						size_t recursion_depth)
{
	auto pixel_coordinates = get_pixel_coordinates(width_index, height_index);
	auto camera_ray = get_camera_ray(pixel_coordinates.first, pixel_coordinates.second);
	return get_pixel_color(camera_ray, objects_in_scene, scene_illumination, recursion_depth);
}
