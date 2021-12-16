//
// Created by andreas on 11.10.21.
//

#include "gmock/gmock.h"
#include "materials/interfaces/i_material_builder.h"

class MockMaterialBuilder: public IMaterialBuilder
{
public:
	MOCK_METHOD(float, specular, (), (const final));
	MOCK_METHOD(float, diffuse, (), (const final));
	MOCK_METHOD(float, ambient, (), (const final));
	MOCK_METHOD(float, shininess, (), (const final));
	MOCK_METHOD(float, specular_exponent, (), (const final));
	MOCK_METHOD(c_vector3, rgb_color, (), (const final));
	MOCK_METHOD(float, refraction_coefficient, (), (const final));
	MOCK_METHOD(std::string, name, (), (const final));
	MOCK_METHOD(void, set_specular_coefficient, (const float &), (final));
	MOCK_METHOD(void, set_diffuse_coefficient, (const float &), (final));
	MOCK_METHOD(void, set_ambient_coefficient, (const float &), (final));
	MOCK_METHOD(void, set_shininess, (const float &), (final));
	MOCK_METHOD(void, set_transparency, (const float &), (final));
	MOCK_METHOD(void, set_rgb_color, (const c_vector3 &), (final));
	MOCK_METHOD(void, set_refraction_index, (const float &), (final));
	MOCK_METHOD(void, set_name, (const std::string &), (final));

};


