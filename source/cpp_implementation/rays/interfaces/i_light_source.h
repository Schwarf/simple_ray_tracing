//
// Created by andreas on 04.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_LIGHT_SOURCE_H
#define SIMPLE_RAY_TRACING_I_LIGHT_SOURCE_H
#include <memory>
#include "miscellaneous/templates/fixed_sized_array.h"

class ILightSource
{
public:
	virtual Point3D position() const = 0;
	virtual float intensity() const = 0;
	virtual ~ILightSource() = default;
};

using ILightSourcePtr = std::shared_ptr<ILightSource>;
#endif //SIMPLE_RAY_TRACING_I_LIGHT_SOURCE_H
