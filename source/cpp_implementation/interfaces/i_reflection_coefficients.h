//
// Created by andreas on 02.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_REFLECTION_COEFFICIENTS_H
#define SIMPLE_RAY_TRACING_I_REFLECTION_COEFFICIENTS_H


class IReflectionCoefficients  {
public:
    virtual float specular_reflection() const = 0;

    virtual float diffuse_reflection() const = 0;

    virtual float ambient_reflection() const = 0;

    virtual float shininess() const = 0;

    virtual ~IReflectionCoefficients() = default;
};

#endif //SIMPLE_RAY_TRACING_I_REFLECTION_COEFFICIENTS_H
