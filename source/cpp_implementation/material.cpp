//
// Created by andreas on 02.10.21.
//

#include "material.h"

float Material::specular_reflection() const {
    return specular_reflection_;
}

float Material::diffuse_reflection() const {
    return diffuse_reflection_;
}

float Material::ambient_reflection() const {
    return ambient_reflection_;
}

float Material::shininess() const {
    return shininess_;
}

Material::Material(const IMaterialBuilder &builder) {
    specular_reflection_ = builder.specular_reflection();
    diffuse_reflection_ = builder.diffuse_reflection();
    ambient_reflection_ = builder.ambient_reflection();
    shininess_ = builder.shininess();
}





