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

	std::shared_ptr<IRay>  reflected_ray(const std::shared_ptr<IRay> &ray, const std::shared_ptr<IHitRecord> &hit_record) const final;
	std::shared_ptr<IRay>  refracted_ray(const std::shared_ptr<IRay> &ray,
										 const std::shared_ptr<IHitRecord> &hit_record,
										 const float &air_refraction_index) const final;

};


#endif //SIMPLE_RAY_TRACING_RAY_INTERACTIONS_H
