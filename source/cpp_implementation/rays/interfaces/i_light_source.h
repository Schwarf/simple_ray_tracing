//
// Created by andreas on 04.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_LIGHT_SOURCE_H
#define SIMPLE_RAY_TRACING_I_LIGHT_SOURCE_H
#include "miscellaneous/templates/n_tuple.h"

class ILightSource
{
public:
	virtual Point3D position() const = 0;
	virtual float intensity() const = 0;
	virtual ~ILightSource() = default;
};

#endif //SIMPLE_RAY_TRACING_I_LIGHT_SOURCE_H
