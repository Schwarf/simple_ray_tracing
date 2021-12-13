//
// Created by andreas on 02.10.21.
//


#include "material_builder.h"

void MaterialBuilder::set_specular_reflection(const float &specular_reflection)
{
	std::string class_name = "MaterialBuilder";
	Validate<float>::is_above_threshold("specular_reflection", specular_reflection, 0.0, class_name);
	specular_reflection_ = specular_reflection;
}

void MaterialBuilder::set_diffuse_reflection(const float &diffuse_reflection)
{
	std::string class_name = "MaterialBuilder";
	Validate<float>::is_above_threshold("diffuse_reflection", diffuse_reflection, 0.0, class_name);
	diffuse_reflection_ = diffuse_reflection;
}

void MaterialBuilder::set_ambient_reflection(const float &ambient_reflection)
{
	std::string class_name = "MaterialBuilder";
	Validate<float>::is_above_threshold("ambient_reflection", ambient_reflection, 0.0, class_name);
	ambient_reflection_ = ambient_reflection;
}

void MaterialBuilder::set_shininess(const float &shininess)
{
	std::string class_name = "MaterialBuilder";
	Validate<float>::is_above_threshold("shininess", shininess, 0.0, class_name);

	shininess_ = shininess;
}

void MaterialBuilder::set_specular_exponent(const float &specular_exponent)
{
	std::string class_name = "MaterialBuilder";
	Validate<float>::is_above_threshold("specular_exponent", specular_exponent, 0.0, class_name);
	specular_exponent_ = specular_exponent;
}

void MaterialBuilder::set_refraction_coefficient(const float &refraction_coefficient)
{
	std::string class_name = "MaterialBuilder";
	Validate<float>::is_above_threshold("refraction_coefficient", refraction_coefficient, 0.0, class_name);

	refraction_coefficient_ = refraction_coefficient;
}

void MaterialBuilder::set_name(const std::string &name)
{
	name_ = name;
}

float MaterialBuilder::specular_reflection() const
{
	return specular_reflection_;
}

float MaterialBuilder::diffuse_reflection() const
{
	return diffuse_reflection_;
}

float MaterialBuilder::ambient_reflection() const
{
	return ambient_reflection_;
}

float MaterialBuilder::shininess() const
{
	return shininess_;
}

std::string MaterialBuilder::name() const
{
	if (name_.empty()) {
		throw std::invalid_argument("In MaterialBuilder name_ is empty. ");
	}
	return name_;
}

float MaterialBuilder::refraction_coefficient() const
{
	return refraction_coefficient_;
}

float MaterialBuilder::specular_exponent() const
{
	return specular_exponent_;
}

void MaterialBuilder::set_rgb_color(const c_vector3 &rgb_color)
{
	std::string class_name = "MaterialBuilder";
	Validate<float>::is_in_open_interval("color value red", rgb_color[0], 0.0, 1.0, class_name);
	Validate<float>::is_in_open_interval("color value green", rgb_color[1], 0.0, 1.0, class_name);
	Validate<float>::is_in_open_interval("color value blue", rgb_color[2], 0.0, 1.0, class_name);
	rgb_color_ = rgb_color;
}

c_vector3 MaterialBuilder::rgb_color() const
{
	return rgb_color_;
}
