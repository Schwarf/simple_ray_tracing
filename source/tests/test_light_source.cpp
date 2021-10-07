//
// Created by andreas on 07.10.21.
//
#include "gtest/gtest.h"
#include "light_source.h"


TEST(VectorRefTestSuite, destroy_vector)
{
    auto position = c_vector3{0,0,0};
    LightSource light_source(position, 0.5);
    EXPECT_FLOAT_EQ(light_source.intensity(), 0.5);
}
