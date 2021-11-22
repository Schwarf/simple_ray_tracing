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
