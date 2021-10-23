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
	c_vector3 reflection(const c_vector3 &light_direction, c_vector3 &point_normal) const final;
	c_vector3 casting(const IRay &ray, ITargetObject &sphere, const ILightSource &light_source) const final;
private:

};


#endif //SIMPLE_RAY_TRACING_RAY_INTERACTIONS_H
