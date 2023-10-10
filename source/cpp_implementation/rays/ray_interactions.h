//
// Created by andreas on 04.10.21.
//

#ifndef SIMPLE_RAY_TRACING_RAY_INTERACTIONS_H
#define SIMPLE_RAY_TRACING_RAY_INTERACTIONS_H
#include "miscellaneous/random_number_generator.h"
#include "ray.h"
#include "hit_record.h"

class RayInteractions final
{
public:
	RayInteractions() = default;
	void compute_reflected_ray(const Ray &ray, const HitRecord &hit_record, Ray &reflected_ray) const;
	void compute_refracted_ray(const Ray &ray,
							   const HitRecord &hit_record,
							   Ray &refracted_ray) const;
	void set_air_refraction_index(float air_refraction_index);
private:
	float air_refraction_index_{1.f};

};

#endif //SIMPLE_RAY_TRACING_RAY_INTERACTIONS_H
