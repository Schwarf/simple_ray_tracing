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
	Vector3D reflection(const Vector3D &light_direction, const Vector3D &point_normal) const final;
	Vector3D refraction(const Vector3D &light_direction,
						 const Vector3D &point_normal,
						 const float &material_refraction_index, const float &air_refraction_index) const final;

};


#endif //SIMPLE_RAY_TRACING_RAY_INTERACTIONS_H
