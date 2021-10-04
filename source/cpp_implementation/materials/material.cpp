//
// Created by andreas on 02.10.21.
//

#include "material.h"


Material::Material(const std::string &name, const IMaterialBuilder &builder) {
    name_ = name;
    specular_reflection_ = builder.specular_reflection();
    diffuse_reflection_ = builder.diffuse_reflection();
    ambient_reflection_ = builder.ambient_reflection();
    shininess_ = builder.shininess();
    specular_exponent_ = builder.specular_exponent();
    red_value_ = builder.red_value();
    green_value_ = builder.green_value();
    blue_value_ = builder.blue_value();
    refraction_coefficient_ = builder.refraction_coefficient();
}

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

std::string Material::name() const {
    return name_;
}

float Material::refraction_coefficient() const {
    return refraction_coefficient_;
}

float Material::red_value() const {
    return red_value_;
}

float Material::green_value() const {
    return green_value_;
}

float Material::blue_value() const {
    return blue_value_;
}

float Material::specular_exponent() const {
    return specular_exponent_;
}





