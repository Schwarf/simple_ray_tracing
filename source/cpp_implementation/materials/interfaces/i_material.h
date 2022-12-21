//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_MATERIAL_H
#define SIMPLE_RAY_TRACING_I_MATERIAL_H
#include "i_phong_reflection_coefficients.h"
#include "miscellaneous/templates/n_tuple.h"
#include "i_refraction_coefficients.h"
#include <string>


class IMaterial: public IRefractionCoefficients, public IPhongReflectionCoefficients
{
public:
	virtual std::string name() const = 0;
	virtual Color rgb_color() const = 0;
	~IMaterial() = default;
};


#endif //SIMPLE_RAY_TRACING_I_MATERIAL_H
