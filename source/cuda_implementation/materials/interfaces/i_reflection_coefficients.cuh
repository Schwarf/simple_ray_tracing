//
// Created by andreas on 02.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_REFLECTION_COEFFICIENTS_H
#define SIMPLE_RAY_TRACING_I_REFLECTION_COEFFICIENTS_H


class IPhongReflectionCoefficients
{
public:
	__device__ virtual float specular() const = 0;

	__device__ virtual float diffuse() const = 0;

	__device__ virtual float ambient() const = 0;

	__device__ virtual float shininess() const = 0;

	__device__ virtual void set_specular_reflection(float specular_coefficient) = 0;

	__device__ virtual void set_diffuse_reflection(float diffuse_coefficient) = 0;

	__device__ virtual void set_ambient_reflection(float ambient_coefficient)  = 0;

	__device__ virtual void set_shininess(float shininess) = 0;

	__device__ virtual ~IPhongReflectionCoefficients() = default;
};

#endif //SIMPLE_RAY_TRACING_I_REFLECTION_COEFFICIENTS_H
