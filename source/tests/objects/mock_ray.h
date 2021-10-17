//
// Created by andreas on 09.10.21.
//

#include "gmock/gmock.h"
#include "rays/interfaces/i_ray.h"

class MockRay: public IRay
{
public:
	MOCK_METHOD(c_vector3, origin, (), (const, final));
	MOCK_METHOD(c_vector3, direction_normalized, (), (const, final));
};


