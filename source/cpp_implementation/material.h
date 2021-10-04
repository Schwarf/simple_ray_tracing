//
// Created by andreas on 02.10.21.
//

#ifndef SIMPLE_RAY_TRACING_MATERIAL_H
#define SIMPLE_RAY_TRACING_MATERIAL_H

#include <memory>
#include "interfaces/i_material_builder.h"

class Material : public IMaterial {
public:
    explicit Material(const std::string & name, const IMaterialBuilder & builder);

    Material(const Material &rhs) = default;

    float specular_reflection() const final;

    float diffuse_reflection() const final;

    float ambient_reflection() const final;

    float shininess() const final;

    virtual ~Material() = default;

    std::string name() const final;

    float refraction_coefficient() const final;

    float red_value() const final;

    float green_value() const final;

    float blue_value() const final;

    float specular_exponent() const final;

private:
    float specular_reflection_{-1.0};
    float diffuse_reflection_{-1.0};
    float ambient_reflection_{-1.0};
    float shininess_{-1.0};
    float specular_exponent_{-1.0};
    float red_value_{-1.0};
    float green_value_{-1.0};
    float blue_value_{-1.0};
    float refraction_coefficient_{-1.0};
    std::string name_;
};


#endif //SIMPLE_RAY_TRACING_MATERIAL_H

