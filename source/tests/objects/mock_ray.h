//
// Created by andreas on 09.10.21.
//

#include "gmock/gmock.h"
#include "rays/interfaces/i_ray.h"

class MockRay: public IRay
{
public:
	MOCK_METHOD(Point3D, origin, (), (const, final));
	MOCK_METHOD(Vector3D, direction_normalized, (), (const, final));
	MOCK_METHOD(void, set_origin, (const Point3D &origin), (final));
	MOCK_METHOD(void, set_direction, (const Vector3D &direction), (final));

};


