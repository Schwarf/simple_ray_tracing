//
// Created by andreas on 02.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_MATERIAL_SETTER_H
#define SIMPLE_RAY_TRACING_I_MATERIAL_SETTER_H
#include "i_material.h"

class IMaterialBuilder: public IMaterial
{
public:
	virtual void set_specular_coefficient(const float &specular_coefficient) = 0;

	virtual void set_diffuse_coefficient(const float &diffuse_coefficient) = 0;

	virtual void set_ambient_coefficient(const float &ambient_coefficient) = 0;

	virtual void set_shininess(const float &shininess) = 0;

	virtual void set_transparency(const float &transparency) = 0;

	virtual void set_rgb_color(const c_vector3 &color) = 0;

	virtual void set_refraction_index(const float &refraction_index) = 0;

	virtual void set_name(const std::string &name) = 0;

	virtual ~IMaterialBuilder() = default;
};

#endif //SIMPLE_RAY_TRACING_I_MATERIAL_SETTER_H


