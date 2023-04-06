//
// Created by andreas on 16.12.21.
//

#ifndef HIT_RECORD_H
#define HIT_RECORD_H

#include <memory>
#include "interfaces/i_hit_record.h"
class HitRecord final: public IHitRecord
{
public:
	HitRecord() = default;
	HitRecord(const Point3D &hit_point, const Vector3D &hit_normal);
	Point3D hit_point() const final;
	Vector3D hit_normal() const final;
	IMaterialPtr material() const final;
	void set_hit_point(const Point3D &hit_point) final;
	void set_hit_normal(const Vector3D &hit_normal) final;
	void set_material(const IMaterialPtr &material) final;

private:
	Point3D hit_point_;
	Vector3D hit_normal_;
	IMaterialPtr material_;
};


#endif //HIT_RECORD_H
