//
// Created by andreas on 16.12.21.
//

#ifndef HIT_RECORD_H
#define HIT_RECORD_H

#include <memory>
#include "miscellaneous/templates/fixed_sized_array.h"
#include "materials/interfaces/i_material.h"
class HitRecord final
{
public:
	HitRecord() = default;
	HitRecord(const Point3D &hit_point, const Vector3D &hit_normal);
	Point3D hit_point() const;
	Vector3D hit_normal() const;
	IMaterialPtr material() const;
	void set_hit_point(const Point3D &hit_point);
	void set_hit_normal(const Vector3D &hit_normal);
	void set_material(const IMaterialPtr &material);
	~HitRecord() = default;
private:
	Point3D hit_point_;
	Vector3D hit_normal_;
	IMaterialPtr material_;
};


#endif //HIT_RECORD_H
