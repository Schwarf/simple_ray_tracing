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

void Camera::get_pixel_coordinates(const size_t &width_index, const size_t &height_index, float &u, float &v)
{
	if (antialiasing_enabled_) {
		auto add_u = random_number_generator_.get_random(0.f, 1.f);
		auto add_v = random_number_generator_.get_random(0.f, 1.f);
		u = (float(width_index) + add_u) / float(image_width_ - 1);
		v = (float(height_index) + add_v) / float(image_height_ - 1);
		return;
	}
	u = float(width_index)  / float(image_width_ - 1);
	v = float(height_index)  / float(image_height_ - 1);
}

void Camera::render_image(std::shared_ptr<IObjectList> &objects_in_scene,
						  std::shared_ptr<ISceneIllumination> &scene_illumination)
{
	float u{};
	float v{};
	int samples_per_pixel = 1;
	size_t recursion_depth = 2;
	if(antialiasing_enabled_)
		samples_per_pixel = 10;
	#pragma omp parallel for
	for (int height_index = 0; height_index < image_height_; height_index++) {
		for (int width_index = 0; width_index < image_width_; width_index++) {
			c_vector3 color_values{0,0,0};
			for(size_t sample =0; sample < samples_per_pixel; ++sample) {
				get_pixel_coordinates(width_index, height_index, u, v);
				auto ray = get_ray(u, v);
				color_values = color_values + get_pixel_color(ray, objects_in_scene, scene_illumination, recursion_depth);
			}
			image_buffer_->set_pixel_value(width_index, height_index, color_values, samples_per_pixel);
		}
	}

}
c_vector3 Camera::get_pixel_color(std::shared_ptr<IRay> &ray,
								  std::shared_ptr<IObjectList> &objects_in_scene,
								  std::shared_ptr<ISceneIllumination> &scene_illumination,
								  size_t recursion_depth)
{
	std::shared_ptr<IHitRecord> hit_record = std::make_shared<HitRecord>(HitRecord()) ;
	auto air_refraction_index = 1.f;
	auto object = objects_in_scene->get_object_hit_by_ray(ray, hit_record);
	if (object == nullptr || recursion_depth < 1) {
		auto mix_parameter = 1.f/2.f*(ray->direction_normalized()[1] + 1.f);
		return scene_illumination->background_color(mix_parameter);
	}
	auto interaction = RayInteractions();
	auto hit_normal = hit_record->hit_normal();
	auto hit_point = hit_record->hit_point();

	auto reflection_direction = interaction.reflection(ray->direction_normalized(), hit_normal).normalize();
	std::shared_ptr<IRay> reflected_ray = std::make_shared<Ray>(Ray(hit_point, reflection_direction));
	auto refraction_coefficient = object->get_material()->refraction_coefficient();
	auto refraction_direction =
		interaction.refraction(ray->direction_normalized(), hit_normal, refraction_coefficient, air_refraction_index)
			.normalize();
	std::shared_ptr<IRay> refracted_ray = std::make_shared<Ray>(Ray(hit_point, refraction_direction));

	// Start recursion
	recursion_depth--;
	auto reflected_color = get_pixel_color(reflected_ray, objects_in_scene, scene_illumination, recursion_depth);
	auto refracted_color = get_pixel_color(refracted_ray, objects_in_scene, scene_illumination, recursion_depth);

	float diffuse_intensity = 0.f;
	float specular_intensity = 0.f;
	std::shared_ptr<ILightSource> light_source = nullptr;
	for (size_t ls_index = 0; ls_index < scene_illumination->number_of_light_sources(); ++ls_index) {
		light_source = scene_illumination->light_source(ls_index);
		auto light_direction = (light_source->position() - hit_point).normalize();
		std::shared_ptr<IRay> light_source_ray = std::make_shared<Ray>(Ray(hit_point, light_direction));
		std::shared_ptr<IHitRecord> shadow_hit_record = std::make_shared<HitRecord>(HitRecord()) ;

		auto object_in_shadow = objects_in_scene->get_object_hit_by_ray(light_source_ray, shadow_hit_record);
		auto shadow_point = shadow_hit_record->hit_point();
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
	auto refraction = refracted_color * object->get_material()->shininess();
	return diffuse_reflection + specular_reflection + ambient_reflection + refraction;
}

std::shared_ptr<IImageBuffer> Camera::get_image_buffer()
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
