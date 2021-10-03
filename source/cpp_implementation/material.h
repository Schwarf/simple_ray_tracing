//
// Created by andreas on 02.10.21.
//

#ifndef SIMPLE_RAY_TRACING_MATERIAL_H
#define SIMPLE_RAY_TRACING_MATERIAL_H

#include <memory>
#include "interfaces/i_reflection_coefficients.h"
#include "interfaces/i_check.h"
#include "interfaces/i_material_builder.h"

class Material : public IReflectionCoefficients {
public:
    explicit Material(const IMaterialBuilder & builder);

    Material(const Material &rhs) = default;

    float specular_reflection() const override;

    float diffuse_reflection() const override;

    float ambient_reflection() const override;

    float shininess() const override;

    virtual ~Material() = default;

private:
    float specular_reflection_{-1.0};
    float diffuse_reflection_{-1.0};
    float ambient_reflection_{-1.0};
    float shininess_{-1.0};

};


#endif //SIMPLE_RAY_TRACING_MATERIAL_H
