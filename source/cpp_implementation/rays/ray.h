//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_RAY_H
#define SIMPLE_RAY_TRACING_RAY_H

#include "rays/interfaces/i_ray.h"

class Ray final: public IRay
{
public:
	Ray() = default;
	Ray(const Point3D &origin, Vector3D &direction);
	Vector3D direction_normalized() const final;

	Point3D origin() const final;

	~Ray() final = default;
	void set_direction(const Vector3D & direction) final;
	void set_origin(const Point3D &origin) final;
private:
	Vector3D direction_normalized_;
	Point3D origin_;
};


#endif //SIMPLE_RAY_TRACING_RAY_H
