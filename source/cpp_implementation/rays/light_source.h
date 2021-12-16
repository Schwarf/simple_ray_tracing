//
// Created by andreas on 04.10.21.
//

#ifndef SIMPLE_RAY_TRACING_LIGHT_SOURCE_H
#define SIMPLE_RAY_TRACING_LIGHT_SOURCE_H

#include "rays/interfaces/i_light_source.h"

class LightSource final: public ILightSource
{
public:
	LightSource(const Point3D &position, float intensity);

	Point3D position() const final;

	float intensity() const final;

private:
	Point3D position_;
	float intensity_;
};


#endif //SIMPLE_RAY_TRACING_LIGHT_SOURCE_H
