//
// Created by andreas on 16.12.21.
//

#ifndef I_HIT_RECORD_H
#define I_HIT_RECORD_H
#include <memory>
#include "miscellaneous/templates/fixed_sized_array.h"
#include "materials/interfaces/i_material.h"

class IHitRecord{
public:
	virtual Point3D hit_point() const = 0;
	virtual Vector3D hit_normal() const = 0;
	virtual IMaterialPtr material() const = 0;
	virtual void set_hit_point(const Point3D & hit_point) = 0;
	virtual void set_hit_normal(const Vector3D & hit_normal) = 0;
	virtual void set_material(const IMaterialPtr & material) = 0;
	virtual ~IHitRecord() = default;
};

using IHitRecordPtr = std::shared_ptr<IHitRecord>;
#endif //I_HIT_RECORD_H
