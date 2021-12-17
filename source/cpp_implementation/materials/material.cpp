//
// Created by andreas on 02.10.21.
//

#include "material.h"


Material::Material(const std::string &name, const IMaterialBuilder &builder)
{
	name_ = name;
	specular_reflection_ = builder.specular();
	diffuse_reflection_ = builder.diffuse();
	ambient_reflection_ = builder.ambient();
	shininess_ = builder.shininess();
	transparency_ = builder.transparency();
	rgb_color_ = builder.rgb_color();
	refraction_index_ = builder.refraction_index();
}

float Material::specular() const
{
	return specular_reflection_;
}

float Material::diffuse() const
{
	return diffuse_reflection_;
}

float Material::ambient() const
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

float Material::refraction_index() const
{
	return refraction_index_;
}

float Material::transparency() const
{
	return transparency_;
}

Color Material::rgb_color() const
{
	return rgb_color_;
}
