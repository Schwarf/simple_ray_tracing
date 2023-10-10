//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_RAY_H
#define SIMPLE_RAY_TRACING_RAY_H

#include "miscellaneous/templates/fixed_sized_array.h"

class Ray final
{
public:
	Ray() = default;
	Ray(const Point3D &origin, const Vector3D &direction);
	Vector3D direction_normalized() const ;

	Point3D origin() const ;

	~Ray()  = default;
	void set_direction(const Vector3D & direction) ;
	void set_origin(const Point3D &origin) ;
private:
	Vector3D direction_normalized_;
	Point3D origin_;
};


#endif //SIMPLE_RAY_TRACING_RAY_H
