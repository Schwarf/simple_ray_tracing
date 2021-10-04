//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_MATERIAL_H
#define SIMPLE_RAY_TRACING_I_MATERIAL_H
#include "i_reflection_coefficients.h"
#include "i_specular_exponent.h"
#include "c_vector.h"
#include "i_refraction_coefficient.h"
#include <string>


class IMaterial: public IRefractionCoefficient, public ISpecularExponent, public IReflectionCoefficients
{
public:
    virtual std::string name () const= 0;
    virtual c_vector3 rgb_color() const = 0;
    ~IMaterial()=default;
};
#endif //SIMPLE_RAY_TRACING_I_MATERIAL_H
