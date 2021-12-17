//
// Created by andreas on 16.12.21.
//

#ifndef I_HIT_RECORD_H
#define I_HIT_RECORD_H
#include <memory>
#include "miscellaneous/templates/n_tuple.h"
#include "materials/interfaces/i_material.h"

class IHitRecord{
public:
	virtual Point3D hit_point() const = 0;
	virtual Vector3D hit_normal() const = 0;
	virtual std::shared_ptr<IMaterial> material() const = 0;
	virtual void set_hit_point(const Point3D & hit_point) = 0;
	virtual void set_hit_normal(const Vector3D & hit_normal) = 0;
	virtual void set_material(const std::shared_ptr<IMaterial> & material) = 0;
	virtual ~IHitRecord() = default;
};

#endif //I_HIT_RECORD_H
