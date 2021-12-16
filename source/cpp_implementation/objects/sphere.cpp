//
// Created by andreas on 03.10.21.
//

#include "sphere.h"

Sphere::Sphere(Point3D &center, float radius)
	:
	material_(nullptr)
{
	center_ = center;
	radius_ = radius;
	init();
}

Point3D Sphere::center() const
{
	return center_;
}

float Sphere::radius() const
{
	return radius_;
}

void Sphere::init() const
{
	Validate<float>::is_above_threshold("radius", radius_, 0.0, " Sphere");
}

bool Sphere::does_ray_intersect(std::shared_ptr<IRay> &ray, std::shared_ptr<IHitRecord> &hit_record) const
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
	hit_record->set_hit_point(ray->origin() + ray->direction_normalized() * closest_hit_distance);
	hit_record->set_hit_normal((hit_record->hit_point() - center_).normalize());
	hit_record->set_material(this->get_material());
	return true;
}

void Sphere::set_material(std::shared_ptr<IMaterial> material)
{
	material_ = material;
}

std::shared_ptr<IMaterial> Sphere::get_material() const
{
	return material_;
}

