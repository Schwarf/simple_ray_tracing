//
// Created by andreas on 02.10.21.
//

#include "material.h"


Material::Material(const std::string &name, const IMaterialBuilder &builder)
{
	name_ = name;
	specular_reflection_ = builder.specular_reflection();
	diffuse_reflection_ = builder.diffuse_reflection();
	ambient_reflection_ = builder.ambient_reflection();
	shininess_ = builder.shininess();
	specular_exponent_ = builder.specular_exponent();
	rgb_color_ = builder.rgb_color();
	refraction_coefficient_ = builder.refraction_coefficient();
}

float Material::specular_reflection() const
{
	return specular_reflection_;
}

float Material::diffuse_reflection() const
{
	return diffuse_reflection_;
}

float Material::ambient_reflection() const
{
	return ambient_reflection_;
}

float Material::shininess() const
{
	return shininess_;
}

std::string Material::name() const
{
	return name_;
}

float Material::refraction_coefficient() const
{
	return refraction_coefficient_;
}

float Material::specular_exponent() const
{
	return specular_exponent_;
}

c_vector3 Material::rgb_color() const
{
	return rgb_color_;
}
