//
// Created by andreas on 02.10.21.
//


#include "material_builder.h"

void MaterialBuilder::set_specular_reflection(const float &specular_reflection) {
    specular_reflection_ = specular_reflection;
}

void MaterialBuilder::set_diffuse_reflection(const float &diffuse_reflection) {
    diffuse_reflection_ = diffuse_reflection;
}

void MaterialBuilder::set_ambient_reflection(const float &ambient_reflection) {
    ambient_reflection_ = ambient_reflection;
}

void MaterialBuilder::set_shininess(const float &shininess) {
    shininess_ = shininess;
}

void MaterialBuilder::set_specular_exponent(const float &specular_exponent) {
    specular_exponent_ = specular_exponent;

}

void MaterialBuilder::set_refraction_coefficient(const float &refraction_coefficient) {
    refraction_coefficient_ = refraction_coefficient;
}

void MaterialBuilder::set_name(const std::string &name) {
    name_ = name;
}


float MaterialBuilder::specular_reflection() const {
    is_above_threshold("specular_reflection", specular_reflection_, 0.0);
    return specular_reflection_;
}

float MaterialBuilder::diffuse_reflection() const {
    is_above_threshold("diffuse_reflection", diffuse_reflection_, 0.0);
    return diffuse_reflection_;
}

float MaterialBuilder::ambient_reflection() const {
    is_above_threshold("ambient_reflection_", ambient_reflection_, 0.0);
    return ambient_reflection_;
}

float MaterialBuilder::shininess() const {
    is_above_threshold("shininess", shininess_, 0.0);
    return shininess_;
}

std::string MaterialBuilder::name() const {
    if (name_.empty()) {
        throw std::invalid_argument("In MaterialBuilder name_ is empty. ");
    }
    return name_;
}


float MaterialBuilder::refraction_coefficient() const {
    is_above_threshold("refraction_coefficient", refraction_coefficient_, 0.0);
    return refraction_coefficient_;
}

float MaterialBuilder::specular_exponent() const {
    is_above_threshold("specular_exponent_", specular_exponent_, 0.0);
    return specular_exponent_;
}

void MaterialBuilder::is_above_threshold(const std::string &variable_name, const float &variable_value,
                                         const float &threshold) const {
    const std::string message_part1 = "In MaterialBuilder ";
    const std::string message_part2 = " is less than ";
    const std::string threshold_part = std::to_string(threshold);
    std::string message = message_part1 + variable_name + message_part2 + threshold_part;
    if (variable_value < threshold) {
        throw std::out_of_range(message);
    }
}

void MaterialBuilder::set_rgb_color(const c_vector3 &rgb_color) {
    rgb_color_ = rgb_color;
}

c_vector3 MaterialBuilder::rgb_color() const {
    is_above_threshold("color value red", rgb_color_[0], 0.0);
    is_above_threshold("color value green", rgb_color_[1], 0.0);
    is_above_threshold("color value blue", rgb_color_[2], 0.0);
    return rgb_color_;
}


