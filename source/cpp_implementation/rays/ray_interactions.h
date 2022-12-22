//
// Created by andreas on 04.10.21.
//

#ifndef SIMPLE_RAY_TRACING_RAY_INTERACTIONS_H
#define SIMPLE_RAY_TRACING_RAY_INTERACTIONS_H
#include "miscellaneous/templates/random_number_generator.h"
#include "rays/interfaces/i_ray_interactions.h"
#include "ray.h"

class RayInteractions final: IRayInteractions
{
public:
	RayInteractions() = default;
	IRayPtr  reflected_ray(const IRayPtr &ray, const IHitRecordPtr &hit_record) const final;
	IRayPtr  refracted_ray(const IRayPtr &ray,
										 const IHitRecordPtr &hit_record,
										 const float &air_refraction_index) const final;
};


#endif //SIMPLE_RAY_TRACING_RAY_INTERACTIONS_H
