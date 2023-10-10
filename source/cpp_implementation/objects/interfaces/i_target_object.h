//
// Created by andreas on 04.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_GEOMETRIC_OBJECT_H
#define SIMPLE_RAY_TRACING_I_GEOMETRIC_OBJECT_H
#include <memory>
#include "materials/interfaces/i_material.h"
#include "rays/ray.h"
#include "rays/hit_record.h"

class ITargetObject
{
public:
	virtual void set_material(const IMaterialPtr &material) = 0;
	virtual bool does_ray_intersect( Ray &ray,  HitRecord &hit_record) const = 0;
	virtual IMaterialPtr get_material() const = 0;
	virtual ~ITargetObject() = default;
	virtual size_t object_id() const = 0;
};

using ITargetObjectPtr = std::shared_ptr<ITargetObject>;

#endif //SIMPLE_RAY_TRACING_I_GEOMETRIC_OBJECT_H
