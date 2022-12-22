//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_SPHERE_H
#define SIMPLE_RAY_TRACING_I_SPHERE_H
#include "miscellaneous/templates/fixed_sized_array.h"
#include "rays/interfaces/i_ray_intersection.h"
#include "i_target_object.h"

class ISphere: public ITargetObject
{
public:
	virtual Point3D center() const = 0;
	virtual float radius() const = 0;
	virtual ~ISphere() = default;
};

using ISpherePtr = std::shared_ptr<ISphere>;
#endif //SIMPLE_RAY_TRACING_I_SPHERE_H
