//
// Created by andreas on 04.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_RAY_REFLECTION_H
#define SIMPLE_RAY_TRACING_I_RAY_REFLECTION_H
#include "i_ray.h"
#include "miscellaneous/templates/c_vector.h"
#include "objects/interfaces/i_target_object.h"
#include "i_light_source.h"

class IRayInteractions
{
public:
	virtual c_vector3 reflection(const c_vector3 &light_direction, c_vector3 &point_normal) const = 0;
	virtual c_vector3 casting(const IRay &ray, ITargetObject &sphere, const ILightSource &light_source) const = 0;
};

#endif //SIMPLE_RAY_TRACING_I_RAY_REFLECTION_H
