//
// Created by andreas on 08.10.21.
//

#include "gtest/gtest.h"
#include "objects/sphere.h"
#include "mock_ray.h"

class SetupSphere: public testing::Test{
protected:
    float radius{1.5};
    c_vector3 center{1.0, 8.678, 17.78};
};


TEST_F(SetupSphere, test_radius)
{
    auto sphere = Sphere(center, radius);
    EXPECT_FLOAT_EQ(sphere.radius(), radius);
}

TEST_F(SetupSphere, test_center)
{
    auto sphere = Sphere(center, radius);
    for(int i =0; i < 3; i++) {
        EXPECT_FLOAT_EQ(sphere.center()[i], center[i]);
    }
}

TEST_F(SetupSphere, test_ray_intersection_ray_direction_is_called)
{
    auto sphere = Sphere(center, radius);
    MockRay mock_ray;
    float closest_hit_distance = 0;
    c_vector3 hit_point{0.,0.,0.};
    using ::testing::Exactly;
    EXPECT_CALL(mock_ray, direction_normalized()).Times(Exactly(1));
    sphere.does_ray_intersect(mock_ray, closest_hit_distance, hit_point);
}

TEST_F(SetupSphere, test_ray_intersection_ray_origin_is_called)
{
    auto sphere = Sphere(center, radius);
    MockRay mock_ray;
    float closest_hit_distance = 0;
    c_vector3 hit_point{0.,0.,0.};
    using ::testing::Exactly;
    EXPECT_CALL(mock_ray, origin()).Times(Exactly(1));
    sphere.does_ray_intersect(mock_ray, closest_hit_distance, hit_point);
}
