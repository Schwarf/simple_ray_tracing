//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_MATERIAL_H
#define SIMPLE_RAY_TRACING_I_MATERIAL_H
#include "./../../miscellaneous/templates/c_vector.cuh"
#include "i_reflection_coefficients.cuh"
#include "i_specular_exponent.cuh"
#include "i_refraction_coefficient.cuh"


class IMaterial: public IRefractionCoefficients, public ISpecularExponent, public IPhongReflectionCoefficients
{
public:
	__device__ virtual FloatTriple rgb_color() const = 0;
	__device__ virtual void set_rgb_color(FloatTriple color) = 0;
	__device__ ~IMaterial() = default;
};
#endif //SIMPLE_RAY_TRACING_I_MATERIAL_H
