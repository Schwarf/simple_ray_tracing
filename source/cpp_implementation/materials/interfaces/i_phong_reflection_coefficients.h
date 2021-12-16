//
// Created by andreas on 02.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_REFLECTION_COEFFICIENTS_H
#define SIMPLE_RAY_TRACING_I_REFLECTION_COEFFICIENTS_H


class IPhongReflectionCoefficients
{
public:
	virtual float specular() const = 0;

	virtual float diffuse() const = 0;

	virtual float ambient() const = 0;

	virtual float shininess() const = 0;

	virtual ~IPhongReflectionCoefficients() = default;
};

#endif //SIMPLE_RAY_TRACING_I_REFLECTION_COEFFICIENTS_H
