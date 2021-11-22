//
// Created by andreas on 22.11.21.
//

#ifndef CAMERA_H
#define CAMERA_H

#include "interfaces/i_camera.h"
#include "ray.h"
class Camera final : ICamera
{
public:
	Camera(int image_width, int image_height, float viewport_width, float focal_length);
	std::shared_ptr<IRay> get_ray(float width_coordinate, float height_coordinate) final;
private:
	int image_width() override;
	int image_height() override;
	float aspect_ratio() override;
	float focal_length() override;
private:
	int image_width_{};
	int image_height_{};
	float focal_length_{};
	float aspect_ratio_{};
	c_vector3 origin_{0.,0.,0.};
	c_vector3 horizontal_direction_{0.,0.,0.};
	c_vector3 vertical_direction_{0.,0.,0.};
	c_vector3 lower_left_corner_{0.,0.,0.};
};


#endif //CAMERA_H
