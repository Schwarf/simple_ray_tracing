//
// Created by andreas on 04.10.21.
//

#ifndef SIMPLE_RAY_TRACING_RAY_INTERACTIONS_H
#define SIMPLE_RAY_TRACING_RAY_INTERACTIONS_H
#include "rays/interfaces/i_ray_interactions.h"

class RayInteractions final: IRayInteractions
{
public:
	RayInteractions() = default;
	c_vector3 reflection(const c_vector3 &light_direction, const c_vector3 &point_normal) const final;
	c_vector3 refraction(const c_vector3 &light_direction,
						 const c_vector3 &point_normal,
						 const float &material_refraction_index, const float &air_refraction_index) const final;

};


#endif //SIMPLE_RAY_TRACING_RAY_INTERACTIONS_H
