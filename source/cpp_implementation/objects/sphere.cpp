//
// Created by andreas on 03.10.21.
//

#include "sphere.h"

Sphere::Sphere(Point3D &center, float radius)
	:
	center_{center},
	radius_{radius},
	radius_squared_{radius * radius},
	material_(nullptr),
	object_id_{}
{
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
	const auto x_hash = std::hash<float>()(center_[0]);
	const auto y_hash = std::hash<float>()(center_[1]);
	const auto z_hash = std::hash<float>()(center_[2]);
	const auto r_hash = std::hash<float>()(radius_);
	object_id_ = x_hash ^ (y_hash << 1) ^ (z_hash) ^ (r_hash << 1);
}

bool Sphere::does_ray_intersect(const IRayPtr &ray, const IHitRecordPtr &hit_record) const
{
	constexpr float epsilon{1e-3};
	constexpr float zero{0.0};
	const Vector3D origin_to_center = (center_ - ray->origin());
	const float origin_to_center_dot_direction = origin_to_center * ray->direction_normalized();
	const float discriminant = origin_to_center_dot_direction * origin_to_center_dot_direction -
		(origin_to_center.squared() - radius_squared_);
	if (discriminant < zero) {
		return false;
	}
	const auto square_root = std::sqrt(discriminant);
	auto closest_hit_distance = origin_to_center_dot_direction - square_root;
	const float hit_distance = origin_to_center_dot_direction + square_root;
	if (closest_hit_distance < epsilon) {
		closest_hit_distance = hit_distance;
	}
	if (closest_hit_distance < epsilon) {
		return false;
	}
	const auto hit_point = ray->origin() + ray->direction_normalized() * closest_hit_distance;
	hit_record->set_hit_point(hit_point);
	const auto hit_normal = (hit_point - center_).normalize();
	hit_record->set_hit_normal(hit_normal);
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

