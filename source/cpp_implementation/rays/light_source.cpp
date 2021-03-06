//
// Created by andreas on 04.10.21.
//

#include "light_source.h"

Point3D LightSource::position() const
{
	return position_;
}

float LightSource::intensity() const
{
	return intensity_;
}

LightSource::LightSource(const Point3D &position, float intensity)
{
	position_ = position;
	intensity_ = intensity;
}
