//
// Created by andreas on 02.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_MATERIAL_SETTER_H
#define SIMPLE_RAY_TRACING_I_MATERIAL_SETTER_H
#include "i_reflection_coefficients.h"

class IMaterialBuilder: public IReflectionCoefficients {
public:
    virtual void set_specular_reflection(const float &specular_reflection) = 0;

    virtual void set_diffuse_reflection(const float &diffuse_reflection) = 0;

    virtual void set_ambient_reflection(const float &ambient_reflection) = 0;

    virtual void set_shininess(const float &shininess) = 0;

    virtual ~IMaterialBuilder() = default;
};

#endif //SIMPLE_RAY_TRACING_I_MATERIAL_SETTER_H
