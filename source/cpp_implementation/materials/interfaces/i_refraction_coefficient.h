//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_REFRACTION_COEFFICIENT_H
#define SIMPLE_RAY_TRACING_I_REFRACTION_COEFFICIENT_H
class IRefractionCoefficient
{
public:
	virtual float refraction_coefficient() const = 0;
	virtual ~IRefractionCoefficient() = default;
};
#endif //SIMPLE_RAY_TRACING_I_REFRACTION_COEFFICIENT_H
