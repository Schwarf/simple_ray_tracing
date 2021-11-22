//
// Created by andreas on 05.10.21.
//

#include "rectangle.h"

float Rectangle::width() const
{
	return width_;
}

float Rectangle::height() const
{
	return height_;
}

c_vector3 Rectangle::bottom_left_position() const
{
	return bottom_left_position_;
}

bool Rectangle::does_ray_intersect(std::shared_ptr<IRay> &ray, c_vector3 &closest_hit_distance, c_vector3 &hit_point) const
{
	auto denominator_dot_product = this->bottom_left_position() * ray->direction_normalized();
	auto epsilon = 1.e-5;
	if (std::abs(denominator_dot_product) < epsilon) {
		return false;
	}
	auto numerator_dot_product = (this->bottom_left_position() - ray->origin()) * this->normal_;
	if (std::abs(numerator_dot_product) < epsilon) {
		return false;
	}
	auto d = numerator_dot_product / denominator_dot_product;
	if (d < 0) {
		return false;
	}
	auto point = ray->origin() + d * ray->direction_normalized();
	auto check_width = (point - bottom_left_position()) * width_vector_;
	auto check_height = (point - bottom_left_position()) * height_vector_;
	if (check_width > width_ || check_width < 0.f) {
		return false;
	}
	if (check_height > height_ || check_height < 0.f) {
		return false;
	}
	hit_point = point;
	return true;
}

void Rectangle::set_material(std::shared_ptr<IMaterial> material)
{
	material_ = material;
}

std::shared_ptr<IMaterial> Rectangle::get_material()
{
	return material_;
}

Rectangle::Rectangle(c_vector3 width_vector,
					 c_vector3 height_vector,
					 const c_vector3 &position,
					 const c_vector3 &normal)
{
	width_ = width_vector.norm();
	height_ = height_vector.norm();
	width_vector_ = width_vector.normalize();
	height_vector_ = height_vector.normalize();
	bottom_left_position_ = position;
	normal_ = normal;
}

void Rectangle::init() const
{
	auto width_dot_height = std::abs(width_vector_ * height_vector_);
	validate_.is_above_threshold("edge dot product", width_dot_height, 1.e-5, " Rectangle");

}


