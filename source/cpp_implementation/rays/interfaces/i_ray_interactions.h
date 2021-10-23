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
	virtual c_vector3 reflection(const c_vector3 &light_direction, const c_vector3 &point_normal) const = 0;
	virtual c_vector3 refraction(const c_vector3 &light_direction, const c_vector3 &point_normal, const float &material_refraction_index,
								const float &air_refraction_index) const = 0;
};

#endif //SIMPLE_RAY_TRACING_I_RAY_REFLECTION_H
