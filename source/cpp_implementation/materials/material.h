//
// Created by andreas on 02.10.21.
//

#ifndef SIMPLE_RAY_TRACING_MATERIAL_H
#define SIMPLE_RAY_TRACING_MATERIAL_H

#include <memory>
#include "materials/interfaces/i_material_builder.h"


class Material final: public IMaterial
{
public:
	explicit Material(const std::string &name, const IMaterialBuilder &builder);

	Material(const Material &rhs) = default;

	float specular_reflection() const final;

	float diffuse_reflection() const final;

	float ambient_reflection() const final;

	float shininess() const final;

	virtual ~Material() = default;

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
	c_vector3 rgb_color_;
	float refraction_coefficient_{-1.0};
	std::string name_;
};


#endif //SIMPLE_RAY_TRACING_MATERIAL_H

