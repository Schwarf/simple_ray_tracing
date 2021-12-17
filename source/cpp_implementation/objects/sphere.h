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

	bool does_ray_intersect(const std::shared_ptr<IRay> &ray, const std::shared_ptr<IHitRecord> &hit_record) const final;

	void set_material(std::shared_ptr<IMaterial> material) final;

	std::shared_ptr<IMaterial> get_material() const final;

private:
	void init() const;
	Point3D center_;
	float radius_;
	std::shared_ptr<IMaterial> material_;
	Validate<float> validate_;
};


#endif //SIMPLE_RAY_TRACING_SPHERE_H
