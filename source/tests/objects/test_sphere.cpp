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
	Point3D center{-6.0, 3.678, -17.78};
	Vector3D null_vector{0., 0., 0.};
	bool ray_intersection(IRay &ray, HitRecord & hit_record)
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
		hit_record.set_hit_point(ray.origin() + t0 * ray.direction_normalized());
		hit_record.set_hit_normal((hit_record.hit_point() - center).normalize());

		return true;
	}
	Vector3D ray_direction{100.5, 100.5, -500.0};
	Point3D ray_origin{-1.1212, 1.313, 2.331};
	IHitRecordPtr hit_record = std::make_shared<HitRecord>(HitRecord());
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
	IRayPtr mock (std::shared_ptr<MockRay>(), &mock_ray);
	using ::testing::Exactly;
	EXPECT_CALL(mock_ray, direction_normalized()).Times(Exactly(1));
	sphere.does_ray_intersect(mock, hit_record);
}

TEST_F(SetupSphere, test_ray_intersection_ray_origin_is_called)
{
	auto sphere = Sphere(center, radius);
	MockRay mock_ray;
	IRayPtr mock (std::shared_ptr<MockRay>(), &mock_ray);
	using ::testing::Exactly;
	EXPECT_CALL(mock_ray, origin()).Times(Exactly(1));
	sphere.does_ray_intersect(mock,hit_record);
}

TEST_F(SetupSphere, test_ray_intersection_closest_distance)
{
	auto sphere = Sphere(center, radius);
	MockRay mock_ray;
	IRayPtr mock (std::shared_ptr<MockRay>(), &mock_ray);
	ray_direction = ray_direction.normalize();
	EXPECT_CALL(mock_ray, origin()).WillRepeatedly(testing::Return(ray_origin));
	EXPECT_CALL(mock_ray, direction_normalized()).WillRepeatedly(testing::Return(ray_direction));

	HitRecord expected_hit_record;
	ray_intersection(mock_ray, expected_hit_record);
	sphere.does_ray_intersect(mock, hit_record);
	for (int i = 0; i < 3; ++i) {
		std::cout << expected_hit_record.hit_normal()[i] << "  " <<  hit_record->hit_normal()[i] <<std::endl;
		EXPECT_FLOAT_EQ(expected_hit_record.hit_normal()[i], hit_record->hit_normal()[i]);
	}
}

TEST_F(SetupSphere, test_ray_intersection_hit_point)
{
	auto sphere = Sphere(center, radius);
	MockRay mock_ray;
	IRayPtr mock (std::shared_ptr<MockRay>(), &mock_ray);
	ray_direction = ray_direction.normalize();

	EXPECT_CALL(mock_ray, origin()).WillRepeatedly(testing::Return(ray_origin));
	EXPECT_CALL(mock_ray, direction_normalized()).WillRepeatedly(testing::Return(ray_direction));

	HitRecord expected_hit_record;
	ray_intersection(mock_ray, expected_hit_record);
	sphere.does_ray_intersect(mock, hit_record);

	for (int i = 0; i < 3; ++i) {
		EXPECT_FLOAT_EQ(expected_hit_record.hit_point()[i], hit_record->hit_point()[i]);
	}
}