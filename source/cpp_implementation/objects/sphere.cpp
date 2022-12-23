//
// Created by andreas on 03.10.21.
//

#include "sphere.h"

Sphere::Sphere(Point3D &center, float radius)
	:
	material_(nullptr),
	object_id_{}
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

void Sphere::init()
{
	Validate<float>::is_above_threshold("radius", radius_, 0.0, " Sphere");
	auto x_hash = std::hash<float>()(center_[0]);
	auto y_hash = std::hash<float>()(center_[1]);
	auto z_hash = std::hash<float>()(center_[2]);
	auto r_hash = std::hash<float>()(radius_);
	object_id_ = x_hash ^ (y_hash << 1) ^ (z_hash) ^ (r_hash << 1);
}

bool Sphere::does_ray_intersect(const IRayPtr &ray, const IHitRecordPtr &hit_record) const
{
	Vector3D origin_to_center = (center_ - ray->origin());
	float origin_to_center_dot_direction = origin_to_center * ray->direction_normalized();
	float discriminant = origin_to_center_dot_direction * origin_to_center_dot_direction -
		((origin_to_center * origin_to_center) - radius_ * radius_);
	if (discriminant < 0.0) {
		return false;
	}

	auto closest_hit_distance = origin_to_center_dot_direction - std::sqrt(discriminant);
	float hit_distance = origin_to_center_dot_direction + std::sqrt(discriminant);
	if (closest_hit_distance < epsilon_) {
		closest_hit_distance = hit_distance;
	}
	if (closest_hit_distance < epsilon_) {
		return false;
	}
	hit_record->set_hit_point(ray->origin() + ray->direction_normalized() * closest_hit_distance);
	hit_record->set_hit_normal((hit_record->hit_point() - center_).normalize());
	hit_record->set_material(this->get_material());
	return true;
}

void Sphere::set_material(const IMaterialPtr &material)
{
	material_ = material;
}

IMaterialPtr Sphere::get_material() const
{
	return material_;
}
size_t Sphere::object_id() const
{
	return object_id_;
}

