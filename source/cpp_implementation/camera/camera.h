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


class Camera final: ICamera
{
public:
	Camera(int image_width, int image_height, float viewport_width, float focal_length);
	void render_image(const IObjectListPtr &objects_in_scene,
					  const ISceneIlluminationPtr &scene_illumination) final;
	IRayPtr get_camera_ray(float width_coordinate, float height_coordinate) final;
	IImageBufferPtr get_image_buffer() final;
	void enable_antialiasing() final;
	void disable_antialiasing() final;
	bool antialiasing_enabled() final;
	int image_width() final;
	int image_height() final;
	float aspect_ratio() final;
	float focal_length() final;

private:
	Color get_pixel_color(const IRayPtr &camera_ray,
						  const IObjectListPtr &objects_in_scene,
						  const ISceneIlluminationPtr &scene_illumination,
						  size_t recursion_depth) final;

	std::pair<float, float> get_pixel_coordinates(const size_t &width_index, const size_t &height_index) const;

	Color compute_one_pixel(const size_t &width_index, const size_t &height_index,
							const IObjectListPtr &objects_in_scene,
							const ISceneIlluminationPtr &scene_illumination,
							size_t recursion_depth);

private:
	int image_width_{};
	int image_height_{};
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
