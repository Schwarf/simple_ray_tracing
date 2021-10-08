//
// Created by andreas on 08.10.21.
//

#include "gtest/gtest.h"
#include "rays/ray.h"


class SetupRay: public testing::Test{
protected:
    c_vector3 origin{10.6, 10.,7.};
    c_vector3 direction{-2, 1.2,2.5};
};

TEST_F(SetupRay, test_orgin)
{
    auto ray = Ray(origin, direction);
    for(int i=0; i< 3; ++i)
    {
        EXPECT_FLOAT_EQ(origin[i], ray.origin()[i]);
    }
}

TEST_F(SetupRay, test_direction)
{
    auto ray = Ray(origin, direction);
    for(int i=0; i< 3; ++i)
    {
        EXPECT_FLOAT_EQ(direction.normalize()[i], ray.direction_normalized()[i]);
    }
}
