//
// Created by andreas on 08.10.21.
//

#include "gtest/gtest.h"
#include "rays/ray.h"


class SetupRay: public testing::Test
{
protected:
	Point3D origin{10.6, 10., 7.};
	Vector3D direction{-2, 1.2, 2.5};
};

TEST_F(SetupRay, test_orgin)
{
	auto ray = Ray(origin, direction);
	for (int i = 0; i < 3; ++i) {
		EXPECT_FLOAT_EQ(origin[i], ray.origin()[i]);
	}
}

TEST_F(SetupRay, test_direction)
{
	auto ray = Ray(origin, direction);
	for (int i = 0; i < 3; ++i) {
		EXPECT_FLOAT_EQ(direction.normalize()[i], ray.direction_normalized()[i]);
	}
}


TEST_F(SetupRay, test_set_orgin)
{
	auto ray = Ray(origin, direction);
	for (int i = 0; i < 3; ++i) {
		EXPECT_FLOAT_EQ(origin[i], ray.origin()[i]);
	}
	ray.set_origin(direction);
	for (int i = 0; i < 3; ++i) {
		EXPECT_FLOAT_EQ(direction[i], ray.origin()[i]);
	}
}

TEST_F(SetupRay, test_set_direction)
{
	auto ray = Ray(origin, direction);
	for (int i = 0; i < 3; ++i) {
		EXPECT_FLOAT_EQ(direction.normalize()[i], ray.direction_normalized()[i]);
	}
	ray.set_direction(origin);
	for (int i = 0; i < 3; ++i) {
		EXPECT_FLOAT_EQ(origin.normalize()[i], ray.direction_normalized()[i]);
	}
}
