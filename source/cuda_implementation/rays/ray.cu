//
// Created by andreas on 03.10.21.
//

#include <miscellaneous/templates/n_tuple.h>
#include "ray.cuh"

__device__ Ray::Ray(FloatTriple &origin, FloatTriple &direction)
{
	direction_normalized_ = direction.normalize();
	origin_ = origin;
}

__device__ Vector3D Ray::direction_normalized() const
{
	return direction_normalized_;
}

__device__ FloatTriple Ray::origin() const
{
	return origin_;
}

