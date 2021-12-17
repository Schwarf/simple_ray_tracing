//
// Created by andreas on 04.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_GEOMETRIC_OBJECT_H
#define SIMPLE_RAY_TRACING_I_GEOMETRIC_OBJECT_H
#include <memory>
#include "materials/interfaces/i_material.h"
#include "rays/interfaces/i_ray_intersection.h"

class ITargetObject: public IRayIntersection
{
public:
	virtual void set_material(const std::shared_ptr<IMaterial> &material) = 0;
	virtual std::shared_ptr<IMaterial> get_material() const = 0;
	virtual ~ITargetObject() = default;
};

#endif //SIMPLE_RAY_TRACING_I_GEOMETRIC_OBJECT_H
