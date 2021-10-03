//
// Created by andreas on 02.10.21.
//

#ifndef SIMPLE_RAY_TRACING_MATERIAL_BUILDER_H
#define SIMPLE_RAY_TRACING_MATERIAL_BUILDER_H

#include "interfaces/i_material_builder.h"
#include "interfaces/i_reflection_coefficients.h"
#include <stdexcept>

class MaterialBuilder : public IMaterialBuilder {
public:
    MaterialBuilder() = default;

    void set_specular_reflection(const float &specular_reflection) override;

    void set_diffuse_reflection(const float &diffuse_reflection) override;

    void set_ambient_reflection(const float &ambient_reflection) override;

    void set_shininess(const float &shininess) override;

    ~MaterialBuilder() override = default;

    float specular_reflection() const override;

    float diffuse_reflection() const override;

    float ambient_reflection() const override;

    float shininess() const override;

    std::string name() const override;

    void set_specular_exponent(const float &specular_exponent) override;

    void set_red_value(const float &red_value) override;

    void set_green_value(const float &green_value) override;

    void set_blue_value(const float &blue_value) override;

    void set_refraction_coefficient(const float &refraction_coefficient) override;

    void set_name(const std::string &name) override;

    float refraction_coefficient() const override;

    float red_value() const override;

    float green_value() const override;

    float blue_value() const override;

    float specular_exponent() const override;

private:
    void
    is_above_threshold(const std::string &variable_name, const float &variable_value, const float &threshold) const override;

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


#endif //SIMPLE_RAY_TRACING_MATERIAL_BUILDER_H
