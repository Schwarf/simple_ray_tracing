//
// Created by andreas on 22.11.21.
//

#ifndef CAMERA_H
#define CAMERA_H

#include <memory>
#include "miscellaneous/interfaces/i_image_buffer.h"
#include "camera/interfaces/i_camera.h"
#include "rays/interfaces/i_ray.h"
#include "rays/interfaces/i_ray_interactions.h"

#include "rays/ray.h"
#include "miscellaneous/image_buffer.h"
#include "rays/ray_interactions.h"

class Camera final: ICamera
{
public:
	Camera(int image_width, int image_height, float viewport_width, float focal_length);
	void render_image(std::shared_ptr<IObjectList> &objects_in_scene, std::shared_ptr<ISceneIllumination> &scene_illumination) final;
	std::shared_ptr<IRay> get_ray(float width_coordinate, float height_coordinate) final;
	std::shared_ptr<IImageBuffer> get_image_buffer() final;

private:
	int image_width() override;
	int image_height() override;
	float aspect_ratio() override;
	float focal_length() override;
	c_vector3 get_pixel_color(std::shared_ptr<IRay> &ray,
							  std::shared_ptr<IObjectList> &objects_in_scene,
							  std::shared_ptr<ISceneIllumination> &scene_illumination,
							  size_t recursion_depth=0) final;


private:
	int image_width_{};
	int image_height_{};
	float focal_length_{};
	float aspect_ratio_{};
	c_vector3 origin_{0., 0., 0.};
	c_vector3 horizontal_direction_{0., 0., 0.};
	c_vector3 vertical_direction_{0., 0., 0.};
	c_vector3 lower_left_corner_{0., 0., 0.};
	std::shared_ptr<IImageBuffer> image_buffer_;
};


#endif //CAMERA_H
