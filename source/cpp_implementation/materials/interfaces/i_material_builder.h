//
// Created by andreas on 02.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_MATERIAL_SETTER_H
#define SIMPLE_RAY_TRACING_I_MATERIAL_SETTER_H
#include "i_material.h"
#include "miscellaneous/interfaces/i_validate.h"

class IMaterialBuilder: public IMaterial
{
public:
	virtual void set_specular_reflection(const float &specular_reflection) = 0;

	virtual void set_diffuse_reflection(const float &diffuse_reflection) = 0;

	virtual void set_ambient_reflection(const float &ambient_reflection) = 0;

	virtual void set_shininess(const float &shininess) = 0;

	virtual void set_specular_exponent(const float &specular_exponent) = 0;

	virtual void set_rgb_color(const c_vector3 &color) = 0;

	virtual void set_refraction_coefficient(const float &refraction_coefficient) = 0;

	virtual void set_name(const std::string &name) = 0;

	virtual ~IMaterialBuilder() = default;
};

#endif //SIMPLE_RAY_TRACING_I_MATERIAL_SETTER_H


