//
// Created by andreas on 04.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_RAY_REFLECTION_H
#define SIMPLE_RAY_TRACING_I_RAY_REFLECTION_H
#include "i_ray.h"
#include "miscellaneous/templates/fixed_sized_array.h"
#include "objects/interfaces/i_target_object.h"
#include "i_light_source.h"

class IRayInteractions
{
public:
	virtual void compute_reflected_ray(const IRay &ray, const IHitRecord &hit_record, IRay &reflected_ray) const = 0;
	virtual void compute_refracted_ray(const IRay &ray,
									   const IHitRecord &hit_record,
									   IRay &refracted_ray,
									   float air_refraction_index) const = 0;
	virtual void set_air_refraction_index(float air_refraction_index) = 0;

};

#endif //SIMPLE_RAY_TRACING_I_RAY_REFLECTION_H
