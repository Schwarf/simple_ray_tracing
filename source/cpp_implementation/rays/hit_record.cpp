//
// Created by andreas on 16.12.21.
//

#include "hit_record.h"
HitRecord::HitRecord(const Point3D &hit_point, const Vector3D &hit_normal)
{
	hit_point_ = hit_point;
	hit_normal_ = hit_normal;
}

Point3D HitRecord::hit_point() const
{
	return hit_point_;
}
Vector3D HitRecord::hit_normal() const
{
	return hit_normal_;
}
std::shared_ptr<IMaterial> HitRecord::material() const
{
	return material_;
}
void HitRecord::set_hit_point(const Point3D &hit_point)
{
	hit_point_ = hit_point;
}
void HitRecord::set_hit_normal(const Vector3D &hit_normal)
{
	hit_normal_ = hit_normal;
}
void HitRecord::set_material(const std::shared_ptr<IMaterial> &material)
{
	material_ = material;
}
