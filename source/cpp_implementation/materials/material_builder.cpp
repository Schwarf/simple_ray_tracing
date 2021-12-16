//
// Created by andreas on 02.10.21.
//


#include "material_builder.h"

void MaterialBuilder::set_specular_coefficient(const float &specular_coefficient)
{
	std::string class_name = "MaterialBuilder";
	Validate<float>::is_above_threshold("specular", specular_coefficient, 0.0, class_name);
	specular_reflection_ = specular_coefficient;
}

void MaterialBuilder::set_diffuse_coefficient(const float &diffuse_coefficient)
{
	std::string class_name = "MaterialBuilder";
	Validate<float>::is_above_threshold("diffuse", diffuse_coefficient, 0.0, class_name);
	diffuse_reflection_ = diffuse_coefficient;
}

void MaterialBuilder::set_ambient_coefficient(const float &ambient_coefficient)
{
	std::string class_name = "MaterialBuilder";
	Validate<float>::is_above_threshold("ambient", ambient_coefficient, 0.0, class_name);
	ambient_reflection_ = ambient_coefficient;
}

void MaterialBuilder::set_shininess(const float &shininess)
{
	std::string class_name = "MaterialBuilder";
	Validate<float>::is_above_threshold("shininess", shininess, 0.0, class_name);

	shininess_ = shininess;
}

void MaterialBuilder::set_transparency(const float &transparency)
{
	std::string class_name = "MaterialBuilder";
	Validate<float>::is_above_threshold("transparency", transparency, 0.0, class_name);
	transparency_ = transparency;
}

void MaterialBuilder::set_refraction_index(const float &refraction_index)
{
	std::string class_name = "MaterialBuilder";
	Validate<float>::is_above_threshold("refraction_index", refraction_index, 0.0, class_name);

	refraction_index_ = refraction_index;
}

void MaterialBuilder::set_name(const std::string &name)
{
	name_ = name;
}

float MaterialBuilder::specular() const
{
	return specular_reflection_;
}

float MaterialBuilder::diffuse() const
{
	return diffuse_reflection_;
}

float MaterialBuilder::ambient() const
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

float MaterialBuilder::refraction_index() const
{
	return refraction_index_;
}

float MaterialBuilder::transparency() const
{
	return transparency_;
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
