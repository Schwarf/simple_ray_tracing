//
// Created by andreas on 04.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_RAY_REFLECTION_H
#define SIMPLE_RAY_TRACING_I_RAY_REFLECTION_H
#include "i_ray.h"
#include "miscellaneous/templates/n_tuple.h"
#include "objects/interfaces/i_target_object.h"
#include "i_light_source.h"

class IRayInteractions
{
public:
	virtual std::shared_ptr<IRay> reflected_ray(const std::shared_ptr<IRay> &ray, const std::shared_ptr<IHitRecord> & hit_record) const = 0;
	virtual std::shared_ptr<IRay> refracted_ray(const std::shared_ptr<IRay> &ray, const std::shared_ptr<IHitRecord> & hit_record,
												const float &air_refraction_index) const = 0;
};

#endif //SIMPLE_RAY_TRACING_I_RAY_REFLECTION_H
