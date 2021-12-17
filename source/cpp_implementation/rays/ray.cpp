//
// Created by andreas on 03.10.21.
//

#include "ray.h"

Ray::Ray(const Point3D &origin, Vector3D &direction)
{
	direction_normalized_ = direction.normalize();
	origin_ = origin;
}

Vector3D Ray::direction_normalized() const
{
	return direction_normalized_;
}

Point3D Ray::origin() const
{
	return origin_;
}
void Ray::set_direction(Vector3D direction)
{
	direction_normalized_ = direction.normalize();
}
void Ray::set_origin(const Point3D &origin)
{
	origin_ = origin;
}

