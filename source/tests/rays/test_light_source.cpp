//
// Created by andreas on 07.10.21.
//
#include "gtest/gtest.h"
#include "rays/light_source.h"


class SetupLightSource: public testing::Test
{
protected:
    c_vector3 position{0.2, 3.0, 4.0};
    float intensity =0.5;
};

TEST_F(SetupLightSource, test_light_source_position) {
    auto light_source = LightSource(position, intensity);
    for (int i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(light_source.position()[i], position[i]);
    }
}

TEST_F(SetupLightSource, test_light_source_intensity) {
    auto light_source = LightSource(position, intensity);
    EXPECT_FLOAT_EQ(light_source.intensity(), intensity);
}
