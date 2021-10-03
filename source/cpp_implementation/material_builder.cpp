//
// Created by andreas on 02.10.21.
//


#include "material_builder.h"

MaterialBuilder::MaterialBuilder():
        specular_reflection_(-1.0),
        diffuse_reflection_(-1.0),
        ambient_reflection_(-1.0),
        shininess_(-1.0)
{}

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

float MaterialBuilder::specular_reflection() const {
    if(specular_reflection_ < 0.0)
    {
        throw std::out_of_range("In MaterialBuilder specular_reflection_ is less than 0. ");
    }
    return specular_reflection_;
}

float MaterialBuilder::diffuse_reflection() const{
    if(diffuse_reflection_ < 0.0)
    {
        throw std::out_of_range("In MaterialBuilder diffuse_reflection_ is less than 0. ");
    }
    return diffuse_reflection_;
}

float MaterialBuilder::ambient_reflection() const{
    if(ambient_reflection_ < 0.0)
    {
        throw std::out_of_range("In MaterialBuilder ambient_reflection_ is less than 0. ");
    }
    return ambient_reflection_;
}

float MaterialBuilder::shininess() const{
    if(shininess_ < 0.0)
    {
        throw std::out_of_range("In MaterialBuilder shininess_ is less than 0. ");
    }

    return shininess_;
}

