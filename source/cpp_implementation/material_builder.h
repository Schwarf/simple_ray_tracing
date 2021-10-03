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
    MaterialBuilder();

    void set_specular_reflection(const float &specular_reflection) override;

    void set_diffuse_reflection(const float &diffuse_reflection) override;

    void set_ambient_reflection(const float &ambient_reflection) override;

    void set_shininess(const float &shininess) override;

    ~MaterialBuilder() override = default;

    float specular_reflection() const override;

    float diffuse_reflection() const override;

    float ambient_reflection() const override;

    float shininess() const override;

private:
    float specular_reflection_;
    float diffuse_reflection_;
    float ambient_reflection_;
    float shininess_;
};


#endif //SIMPLE_RAY_TRACING_MATERIAL_BUILDER_H
