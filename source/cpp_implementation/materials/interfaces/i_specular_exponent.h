//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_SPECULAR_EXPONENT_H
#define SIMPLE_RAY_TRACING_I_SPECULAR_EXPONENT_H
class ISpecularExponent
{
public:
	virtual float specular_exponent() const = 0;
	virtual ~ISpecularExponent() = default;
};
#endif //SIMPLE_RAY_TRACING_I_SPECULAR_EXPONENT_H
