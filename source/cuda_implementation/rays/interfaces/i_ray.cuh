//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_RAY_H
#define SIMPLE_RAY_TRACING_I_RAY_H
#include "./../../miscellaneous/templates/c_vector.cuh"
class IRay
{
public:
	__device__ virtual FloatTriple direction_normalized() const = 0;
	__device__ virtual FloatTriple origin() const = 0;
	__device__ virtual ~IRay() = default;
};
#endif //SIMPLE_RAY_TRACING_I_RAY_H
