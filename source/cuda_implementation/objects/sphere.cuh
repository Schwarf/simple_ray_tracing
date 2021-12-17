//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_SPHERE_H
#define SIMPLE_RAY_TRACING_SPHERE_H
#include <memory>
#include <miscellaneous/templates/n_tuple.h>
#include "interfaces/i_sphere.cuh"
#include "./../materials/interfaces/i_material.cuh"

class Sphere final: public ISphere
{
public:
	__device__ Sphere(float_triple &center, float radius, IMaterial * material);
	__device__ Sphere(const Sphere &) = default;
	__device__ Sphere(Sphere &&) = default;
	__device__ Point3D center() const final;

	__device__ float radius() const final;

	__device__ ~Sphere() override = default;

	__device__ bool does_ray_intersect(std::shared_ptr<IRay> &ray, float_triple &hit_normal, float_triple &hit_point) const final;

	__device__ IMaterial * material() const final;

private:
	void init() const;
	float_triple center_;
	float radius_;
	IMaterial * material_;
};


#endif //SIMPLE_RAY_TRACING_SPHERE_H
