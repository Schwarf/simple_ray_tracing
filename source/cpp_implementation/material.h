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

    float specular_reflection() const override;

    float diffuse_reflection() const override;

    float ambient_reflection() const override;

    float shininess() const override;

    virtual ~Material() = default;

    std::string name() const override;

    float refraction_coefficient() const override;

    float red_value() const override;

    float green_value() const override;

    float blue_value() const override;

    float specular_exponent() const override;

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

