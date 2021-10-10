//
// Created by andreas on 08.10.21.
//


#include "gtest/gtest.h"
#include "objects/sphere.h"
#include "mock_ray.h"

class SetupSphere: public testing::Test
{
protected:
	float radius{10.5};
	c_vector3 center{-6.0, 3.678, -17.78};
	bool ray_intersection(IRay & ray, float &closest_hit_distance, c_vector3 & hit_point)
	{
		// see e.g. here
		// https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
		auto center_minus_origin = center - ray.origin();
		float t_ca = center_minus_origin * ray.direction_normalized();
		if (t_ca < 0.0)
			return false;
		float distance_center_ray = std::sqrt(center_minus_origin * center_minus_origin - t_ca * t_ca);
		if (distance_center_ray < 0)
			return false;
		float t_hc = std::sqrt(((radius * radius) - distance_center_ray * distance_center_ray));
		auto t0 = t_ca - t_hc;
		auto t1 = t_ca + t_hc;
		if (t0 < 0) t0 = t1;
		if (t0 < 0) {
			return false;
		}
		hit_point = ray.origin() + t0 * ray.direction_normalized();
		closest_hit_distance = t0;
		return true;
	}
	c_vector3 ray_direction{100.5, 100.5, -500.0};
	c_vector3 ray_origin{-1.1212,1.313, 2.331};
};


TEST_F(SetupSphere, test_radius)
{
	auto sphere = Sphere(center, radius);
	EXPECT_FLOAT_EQ(sphere.radius(), radius);
}

TEST_F(SetupSphere, test_center)
{
	auto sphere = Sphere(center, radius);
	for (int i = 0; i < 3; i++) {
		EXPECT_FLOAT_EQ(sphere.center()[i], center[i]);
	}
}

TEST_F(SetupSphere, test_ray_intersection_ray_direction_is_called)
{
	auto sphere = Sphere(center, radius);
	MockRay mock_ray;
	float closest_hit_distance = 0;
	c_vector3 hit_point{0., 0., 0.};
	using ::testing::Exactly;
	EXPECT_CALL(mock_ray, direction_normalized()).Times(Exactly(1));
	sphere.does_ray_intersect(mock_ray, closest_hit_distance, hit_point);
}

TEST_F(SetupSphere, test_ray_intersection_ray_origin_is_called)
{
	auto sphere = Sphere(center, radius);
	MockRay mock_ray;
	float closest_hit_distance = 0;
	c_vector3 hit_point{0., 0., 0.};
	using ::testing::Exactly;
	EXPECT_CALL(mock_ray, origin()).Times(Exactly(1));
	sphere.does_ray_intersect(mock_ray, closest_hit_distance, hit_point);
}

TEST_F(SetupSphere, test_ray_intersection_closest_distance)
{
	auto sphere = Sphere(center, radius);
	MockRay mock_ray;
	ray_direction = ray_direction.normalize();
	EXPECT_CALL(mock_ray, origin()).WillRepeatedly(testing::Return(ray_origin));
	EXPECT_CALL(mock_ray, direction_normalized()).WillRepeatedly(testing::Return(ray_direction));

	float expected_closest_distance = 0;
	c_vector3 expected_hit_point = c_vector3{0., 0., 0.};
	ray_intersection(mock_ray, expected_closest_distance, expected_hit_point);

	float closest_distance = 0;
	c_vector3 hit_point = c_vector3{0., 0., 0.};

	sphere.does_ray_intersect(mock_ray, closest_distance, hit_point);
	EXPECT_FLOAT_EQ(closest_distance, expected_closest_distance);
}

TEST_F(SetupSphere, test_ray_intersection_hit_point)
{
	auto sphere = Sphere(center, radius);
	MockRay mock_ray;
	ray_direction = ray_direction.normalize();

	EXPECT_CALL(mock_ray, origin()).WillRepeatedly(testing::Return(ray_origin));
	EXPECT_CALL(mock_ray, direction_normalized()).WillRepeatedly(testing::Return(ray_direction));

	float expected_closest_distance = 0;
	c_vector3 expected_hit_point = c_vector3{0., 0., 0.};
	ray_intersection(mock_ray, expected_closest_distance, expected_hit_point);

	float closest_distance = 0;
	c_vector3 hit_point = c_vector3{0., 0., 0.};

	sphere.does_ray_intersect(mock_ray, closest_distance, hit_point);
	for (int i=0; i < 3; ++i) {
		EXPECT_FLOAT_EQ(expected_hit_point[i], hit_point[i]);
	}
}