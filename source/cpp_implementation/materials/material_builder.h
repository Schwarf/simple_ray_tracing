//
// Created by andreas on 02.10.21.
//

#ifndef SIMPLE_RAY_TRACING_MATERIAL_BUILDER_H
#define SIMPLE_RAY_TRACING_MATERIAL_BUILDER_H

#include "materials/interfaces/i_material_builder.h"
#include "materials/interfaces/i_reflection_coefficients.h"
#include <stdexcept>
#include "miscellaneous/validate.h"

class MaterialBuilder final: public IMaterialBuilder
{
public:
	MaterialBuilder() = default;

	void set_specular_reflection(const float &specular_reflection) final;

	void set_diffuse_reflection(const float &diffuse_reflection) final;

	void set_ambient_reflection(const float &ambient_reflection) final;

	void set_shininess(const float &shininess) final;

	void set_specular_exponent(const float &specular_exponent) final;

	void set_rgb_color(const c_vector3 &rgb_color) final;

	void set_refraction_coefficient(const float &refraction_coefficient) final;

	void set_name(const std::string &name) final;

	~MaterialBuilder() override = default;

	float specular_reflection() const final;

	float diffuse_reflection() const final;

	float ambient_reflection() const final;

	float shininess() const final;

	std::string name() const final;

	float refraction_coefficient() const final;

	float specular_exponent() const final;

	c_vector3 rgb_color() const final;

private:
	float specular_reflection_{-1.0};
	float diffuse_reflection_{-1.0};
	float ambient_reflection_{-1.0};
	float shininess_{-1.0};
	float specular_exponent_{-1.0};
	float refraction_coefficient_{-1.0};
	c_vector3 rgb_color_{-1, -1, -1};
	std::string name_;
};


#endif //SIMPLE_RAY_TRACING_MATERIAL_BUILDER_H
