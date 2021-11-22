//
// Created by andreas on 03.10.21.
//

#include "sphere.h"

Sphere::Sphere(c_vector3 &center, float radius)
	:
	material_(nullptr)
{
	center_ = center;
	radius_ = radius;
	init();
}

c_vector3 Sphere::center() const
{
	return center_;
}

float Sphere::radius() const
{
	return radius_;
}

void Sphere::init() const
{
	validate_.is_above_threshold("radius", radius_, 0.0, " Sphere");
}

bool Sphere::does_ray_intersect(std::shared_ptr<IRay> &ray, c_vector3 &hit_normal, c_vector3 &hit_point) const
{
	float closest_hit_distance = -1.0;
	c_vector3 origin_to_center = (center_ - ray->origin());
	float origin_to_center_dot_direction = origin_to_center * ray->direction_normalized();
	float epsilon = 1e-3;
	float discriminant = origin_to_center_dot_direction * origin_to_center_dot_direction -
		((origin_to_center * origin_to_center) - radius_ * radius_);
	if (discriminant < 0.0) {
		return false;
	}

	closest_hit_distance = origin_to_center_dot_direction - std::sqrt(discriminant);
	float hit_distance = origin_to_center_dot_direction + std::sqrt(discriminant);
	if (closest_hit_distance < epsilon) {
		closest_hit_distance = hit_distance;
	}
	if (closest_hit_distance < epsilon) {
		return false;
	}
	hit_point = ray->origin() + ray->direction_normalized() * closest_hit_distance;
	hit_normal = (hit_point - center_).normalize();
	return true;
}

void Sphere::set_material(std::shared_ptr<IMaterial> material)
{
	material_ = material;
}

std::shared_ptr<IMaterial> Sphere::get_material()
{
	return material_;
}

