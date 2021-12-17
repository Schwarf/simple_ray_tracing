//
// Created by andreas on 02.10.21.
//

#ifndef SIMPLE_RAY_TRACING_MATERIAL_H
#define SIMPLE_RAY_TRACING_MATERIAL_H

#include "interfaces/i_material.cuh"
#include "./../miscellaneous/templates/c_vector.cuh"

class Material final: public IMaterial
{
public:
	__device__ Material() = default;
	__device__ FloatTriple rgb_color() const final;
	__device__ float specular() const final;
	__device__ float diffuse() const final;
	__device__ float ambient() const final;
	__device__ float shininess() const final;
	__device__ float refraction_index() const final;
	__device__ float transparency() const final;
	__device__ void set_rgb_color(FloatTriple color) final;
	__device__ void set_specular_reflection(float specular_coefficient) final;
	__device__ void set_diffuse_reflection(float diffuse_coefficient) final; 
	__device__ void set_ambient_reflection(float ambient_coefficient) final;
	__device__ void set_refraction_coefficient(float refraction_coefficient) final;
	__device__ void set_specular_exponent(float specular_exponent) final;
	__device__ void set_shininess(float shininess) final;
	__device__ ~Material() override = default;

private:
	float specular_reflection_{-1.0};
	float diffuse_reflection_{-1.0};
	float ambient_reflection_{-1.0};
	float shininess_{-1.0};
	float specular_exponent_{-1.0};
	FloatTriple rgb_color_{0., 0., 0.};
	float refraction_coefficient_{-1.0};
};


#endif //SIMPLE_RAY_TRACING_MATERIAL_H

