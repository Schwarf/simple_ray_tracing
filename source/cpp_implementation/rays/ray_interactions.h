//
// Created by andreas on 04.10.21.
//

#ifndef SIMPLE_RAY_TRACING_RAY_INTERACTIONS_H
#define SIMPLE_RAY_TRACING_RAY_INTERACTIONS_H
#include "miscellaneous/random_number_generator.h"
#include "rays/interfaces/i_ray_interactions.h"
#include "ray.h"

class RayInteractions final: IRayInteractions
{
public:
	RayInteractions() = default;
	void compute_reflected_ray(const IRay &ray, const IHitRecord &hit_record, IRay &reflected_ray) const final;
	void compute_refracted_ray(const IRay &ray,
							   const IHitRecord &hit_record,
							   IRay &refracted_ray,
							   float air_refraction_index) const final;
};


#endif //SIMPLE_RAY_TRACING_RAY_INTERACTIONS_H
