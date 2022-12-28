//
// Created by andreas on 22.11.21.
//

#ifndef CAMERA_H
#define CAMERA_H

#include <memory>
#include <random>
#include <utility>
#include "miscellaneous/interfaces/i_image_buffer.h"
#include "camera/interfaces/i_camera.h"
#include "rays/interfaces/i_ray.h"
#include "rays/interfaces/i_ray_interactions.h"
#include "rays/ray.h"
#include "miscellaneous/image_buffer.h"
#include "rays/ray_interactions.h"
#include "miscellaneous/random_number_generator.h"
#include <rays/hit_record.h>


template<int denominator, std::size_t...I>
constexpr std::array<float, sizeof...(I)> fill_array(std::index_sequence<I...>)
{
	return std::array<float, sizeof...(I)>{
		static_cast<float>(I) / static_cast<float>(denominator)...
	};
}

template<int denominator, std::size_t N>
constexpr std::array<float, N> fill_array()
{
	return fill_array<denominator>(std::make_index_sequence<N>{});
}

template<int image_width_>
constexpr std::array<float, image_width_> width_coordinates = fill_array<image_width_ - 1, image_width_>();

template<int image_height_>
constexpr std::array<float, image_height_> height_coordinates = fill_array<image_height_ - 1, image_height_>();

template<int image_width_, int image_height_>
class Camera final: ICamera
{
public:
	Camera(float viewport_width, float focal_length)
	{
		focal_length_ = focal_length;
		aspect_ratio_ = static_cast<float>(image_width_) / static_cast<float>(image_height_);
		float viewport_height = viewport_width / aspect_ratio_;
		horizontal_direction_[0] = viewport_width;
		vertical_direction_[1] = viewport_height;
		lower_left_corner_ =
			origin_ - horizontal_direction_ / 2.f - vertical_direction_ / 2.f - Point3D{0, 0, focal_length};
		image_buffer_ = std::make_shared<ImageBuffer>(ImageBuffer(image_width_, image_height_));
	}

	void render_image(const IObjectListPtr &objects_in_scene,
					  const ISceneIlluminationPtr &scene_illumination) final
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

					auto pixel_coordinates = get_pixel_coordinates(width_index, height_index);
					auto camera_ray = get_camera_ray(pixel_coordinates.first, pixel_coordinates.second);
					color_values =
						color_values
							+ get_pixel_color(*camera_ray.get(), objects_in_scene, scene_illumination, recursion_depth);
				}
				image_buffer_->set_pixel_value(width_index, height_index, color_values, samples_per_pixel);
			}
		}

	}

	IRayPtr get_camera_ray(float width_coordinate, float height_coordinate) final
	{
		auto direction =
			lower_left_corner_ + width_coordinate * horizontal_direction_ + height_coordinate * vertical_direction_
				- origin_;
		return std::make_unique<Ray>(Ray(origin_, direction));
	}

	IImageBufferPtr get_image_buffer() final
	{
		return image_buffer_;
	}
	void enable_antialiasing() final
	{
		antialiasing_enabled_ = true;
	}

	void disable_antialiasing() final
	{
		antialiasing_enabled_ = false;
	}
	bool antialiasing_enabled() final
	{
		return antialiasing_enabled_;
	}
	int image_width()
	{
		return image_width_;
	}

	int image_height()
	{
		return image_height_;
	}
	float aspect_ratio() final
	{
		return aspect_ratio_;
	}

	float focal_length() final
	{
		return focal_length_;
	}

private:
	Color get_pixel_color(IRay &camera_ray,
						  const IObjectListPtr &objects_in_scene,
						  const ISceneIlluminationPtr &scene_illumination,
						  size_t recursion_depth)
	{
		auto hit_record = HitRecord();
		auto reflected_ray = Ray();
		auto refracted_ray = Ray();

		auto air_refraction_index = 1.f;
		auto object = objects_in_scene->get_object_hit_by_ray(camera_ray, hit_record);
		if (object == nullptr || recursion_depth < 1) {
			auto
				mix_parameter =
				1.f / 2.f * ((camera_ray.direction_normalized()[0] + camera_ray.direction_normalized()[1]) / 2.f + 1.f);
			return scene_illumination->background_color(mix_parameter);
		}
		// Start recursion
		recursion_depth--;
		ray_interaction_.compute_reflected_ray(camera_ray, hit_record, reflected_ray);
		ray_interaction_.compute_refracted_ray(camera_ray, hit_record, refracted_ray);
		auto reflected_color = get_pixel_color(reflected_ray,
											   objects_in_scene,
											   scene_illumination,
											   recursion_depth);
		auto refracted_color = get_pixel_color(refracted_ray,
											   objects_in_scene,
											   scene_illumination,
											   recursion_depth);

		float diffuse_intensity = 0.f;
		float specular_intensity = 0.f;
		const auto hit_normal = hit_record.hit_normal();
		const auto hit_point = hit_record.hit_point();

		auto shadow_hit_record = HitRecord();
		for (size_t ls_index = 0; ls_index < scene_illumination->number_of_light_sources(); ++ls_index) {
			const ILightSourcePtr light_source = scene_illumination->light_source(ls_index);
			const auto light_direction = (light_source->position() - hit_record.hit_point()).normalize();
			auto light_source_ray = std::make_unique<Ray>(Ray(hit_point, light_direction));

			const auto
				object_in_shadow = objects_in_scene->get_object_hit_by_ray(*light_source_ray.get(), shadow_hit_record);
			const auto shadow_point = shadow_hit_record.hit_point();
			const auto distance_shadow_point_to_point = (shadow_point - hit_point).norm();
			const auto distance_light_source_to_point = (light_source->position() - hit_point).norm();
			if (object_in_shadow && distance_shadow_point_to_point < distance_light_source_to_point) {
				continue;
			}

			diffuse_intensity += light_source->intensity() * std::max(0.f, light_direction * hit_normal);
			ray_interaction_.compute_reflected_ray(*light_source_ray.get(), hit_record, reflected_ray);
			const auto scalar_product = reflected_ray.direction_normalized()*camera_ray.direction_normalized();
			specular_intensity +=
				std::pow(std::max(0.f, scalar_product), object->get_material()->shininess())
					* light_source->intensity();
		}

		Color diffuse_color =
			object->get_material()->rgb_color() * diffuse_intensity * object->get_material()->diffuse();
		Color white = Color{1, 1, 1};
		Color specular_color = specular_intensity * white * object->get_material()->specular();
		Color ambient_color = reflected_color * object->get_material()->ambient();
		Color refraction_color = refracted_color * object->get_material()->transparency();
		return diffuse_color + specular_color + ambient_color + refraction_color;
	}

	std::pair<float, float> get_pixel_coordinates(const size_t &width_index, const size_t &height_index) const
	{
		float add_u = antialiasing_enabled_ ? UniformRandomNumberGenerator::get_random<float>(0.f, 1.f) : 0.f;
		float add_v = antialiasing_enabled_ ? UniformRandomNumberGenerator::get_random<float>(0.f, 1.f) : 0.f;
		auto u = width_coordinates<image_width_>[width_index];
		auto v = height_coordinates<image_height_>[height_index];
		return {u, v};
	}


private:
	float focal_length_{};
	float aspect_ratio_{};
	Point3D origin_{0., 0., 0.};
	Vector3D horizontal_direction_{0., 0., 0.};
	Vector3D vertical_direction_{0., 0., 0.};
	Point3D lower_left_corner_{0., 0., 0.};
	IImageBufferPtr image_buffer_;
	bool antialiasing_enabled_{};
	RayInteractions ray_interaction_;
};


#endif //CAMERA_H
