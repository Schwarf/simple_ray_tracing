//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_REFRACTION_COEFFICIENT_H
#define SIMPLE_RAY_TRACING_I_REFRACTION_COEFFICIENT_H
class IRefractionCoefficients
{
public:
	virtual float refraction_index() const = 0;
	virtual float transparency() const = 0;
	virtual ~IRefractionCoefficients() = default;
};
#endif //SIMPLE_RAY_TRACING_I_REFRACTION_COEFFICIENT_H
