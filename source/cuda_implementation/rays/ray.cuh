//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_RAY_H
#define SIMPLE_RAY_TRACING_RAY_H

#include "interfaces/i_ray.cuh"
#include "../miscellaneous/templates/c_vector.cuh"

class Ray final: public IRay
{
public:
	__device__ Ray(float_triple &origin, float_triple &direction);
	__device__ Vector3D direction_normalized() const final;

	__device__ float_triple origin() const final;

	__device__ ~Ray() final = default;
private:
	float_triple direction_normalized_;
	float_triple origin_;
};


#endif //SIMPLE_RAY_TRACING_RAY_H
