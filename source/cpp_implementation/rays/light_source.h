//
// Created by andreas on 04.10.21.
//

#ifndef SIMPLE_RAY_TRACING_LIGHT_SOURCE_H
#define SIMPLE_RAY_TRACING_LIGHT_SOURCE_H

#include "miscellaneous/templates/fixed_sized_array.h"

class LightSource final
{
public:
	LightSource(const Point3D &position, float intensity);
	Point3D position() const;
	float intensity() const;
private:
	Point3D position_;
	float intensity_;
};


#endif //SIMPLE_RAY_TRACING_LIGHT_SOURCE_H
