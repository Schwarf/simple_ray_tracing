//
// Created by andreas on 11.10.21.
//

#include "materials/material.h"
#include "mock_material_builder.h"
#include "gtest/gtest.h"


class SetupMaterial: public testing::Test
{
protected:
	float specular_reflection{0.1};
	float diffuse_reflection{0.2};
	float ambient_reflection{0.3};
	float shininess{0.4};
	float specular_exponent{0.5};
	float refraction_coefficient{0.9};
	c_vector3 rgb_color{0.6, 0.7, 0.8};
};

TEST_F(SetupMaterial, test_specular_reflection)
{
	MockMaterialBuilder builder;
	EXPECT_CALL(builder, specular_reflection).WillOnce(testing::Return(specular_reflection));
	Material material("specular", builder);
	EXPECT_FLOAT_EQ(material.specular_reflection(), specular_reflection);
}

TEST_F(SetupMaterial, test_diffuse_reflection)
{
	MockMaterialBuilder builder;
	EXPECT_CALL(builder, diffuse_reflection).WillOnce(testing::Return(diffuse_reflection));
	Material material("specular", builder);
	EXPECT_FLOAT_EQ(material.diffuse_reflection(), diffuse_reflection);
}

TEST_F(SetupMaterial, test_ambient_reflection)
{
	MockMaterialBuilder builder;
	EXPECT_CALL(builder, ambient_reflection).WillOnce(testing::Return(ambient_reflection));
	Material material("specular", builder);
	EXPECT_FLOAT_EQ(material.ambient_reflection(), ambient_reflection);
}

TEST_F(SetupMaterial, test_shininess)
{
	MockMaterialBuilder builder;
	EXPECT_CALL(builder, shininess).WillOnce(testing::Return(shininess));
	Material material("specular", builder);
	EXPECT_FLOAT_EQ(material.shininess(), shininess);
}


TEST_F(SetupMaterial, test_specular_exponent)
{
	MockMaterialBuilder builder;
	EXPECT_CALL(builder, specular_exponent).WillOnce(testing::Return(specular_exponent));
	Material material("specular", builder);
	EXPECT_FLOAT_EQ(material.specular_exponent(), specular_exponent);
}

TEST_F(SetupMaterial, test_refraction_coefficient)
{
	MockMaterialBuilder builder;
	EXPECT_CALL(builder, refraction_coefficient).WillOnce(testing::Return(refraction_coefficient));
	Material material("specular", builder);
	EXPECT_FLOAT_EQ(material.refraction_coefficient(), refraction_coefficient);
}


TEST_F(SetupMaterial, test_rgb_color)
{
	MockMaterialBuilder builder;
	EXPECT_CALL(builder, rgb_color).WillOnce(testing::Return(rgb_color));
	Material material("specular", builder);
	for (int i = 0; i < 3; ++i) {
		EXPECT_FLOAT_EQ(material.rgb_color()[i], rgb_color[i]);
	}
}
