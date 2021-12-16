//
// Created by andreas on 02.10.21.
//

#ifndef SIMPLE_RAY_TRACING_MATERIAL_BUILDER_H
#define SIMPLE_RAY_TRACING_MATERIAL_BUILDER_H

#include "materials/interfaces/i_material_builder.h"
#include "materials/interfaces/i_phong_reflection_coefficients.h"
#include <stdexcept>
#include "miscellaneous/validate.h"

class MaterialBuilder final: public IMaterialBuilder
{
public:
	MaterialBuilder() = default;

	void set_specular_coefficient(const float &specular_coefficient) final;

	void set_diffuse_coefficient(const float &diffuse_coefficient) final;

	void set_ambient_coefficient(const float &ambient_coefficient) final;

	void set_shininess(const float &shininess) final;

	void set_transparency(const float &transparency) final;

	void set_rgb_color(const c_vector3 &rgb_color) final;

	void set_refraction_index(const float &refraction_index) final;

	void set_name(const std::string &name) final;

	~MaterialBuilder() override = default;

	float specular() const final;

	float diffuse() const final;

	float ambient() const final;

	float shininess() const final;

	std::string name() const final;

	float refraction_index() const final;

	float transparency() const final;

	c_vector3 rgb_color() const final;

private:
	float specular_reflection_{-1.0};
	float diffuse_reflection_{-1.0};
	float ambient_reflection_{-1.0};
	float shininess_{-1.0};
	float transparency_{-1.0};
	float refraction_index_{-1.0};
	c_vector3 rgb_color_{-1, -1, -1};
	std::string name_;
};


#endif //SIMPLE_RAY_TRACING_MATERIAL_BUILDER_H
