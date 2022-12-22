//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_SPHERE_H
#define SIMPLE_RAY_TRACING_SPHERE_H
#include "objects/interfaces/i_sphere.h"
#include "rays/hit_record.h"
#include "miscellaneous/validate.h"
#include <algorithm>

class Sphere final: public ISphere
{
public:
	Sphere(Point3D &center, float radius);
	Sphere(const Sphere &) = default;
	Sphere(Sphere &&) = default;
	Point3D center() const final;

	float radius() const final;

	~Sphere() override = default;

	bool does_ray_intersect(const IRayPtr &ray, const IHitRecordPtr &hit_record) const final;

	void set_material(const IMaterialPtr &material) final;

	IMaterialPtr get_material() const final;

	size_t object_id() const final;

private:
	void init();
	Point3D center_;
	float radius_;
	IMaterialPtr material_;
	Validate<float> validate_;
	size_t object_id_;
};

#endif //SIMPLE_RAY_TRACING_SPHERE_H
