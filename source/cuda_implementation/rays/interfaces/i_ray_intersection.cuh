//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_RAY_INTERSECTION_H
#define SIMPLE_RAY_TRACING_I_RAY_INTERSECTION_H
#include "i_ray.cuh"

class IRayIntersection
{
public:
	__device__ virtual bool does_ray_intersect(std::shared_ptr<IRay> &ray, float_triple &hit_normal, float_triple &hit_point) const = 0;
};

#endif //SIMPLE_RAY_TRACING_I_RAY_INTERSECTION_H
