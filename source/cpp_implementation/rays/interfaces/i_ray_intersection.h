//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_RAY_INTERSECTION_H
#define SIMPLE_RAY_TRACING_I_RAY_INTERSECTION_H
#include "i_ray.h"
#include "i_hit_record.h"
#include <memory>

class IRayIntersection
{
public:
	virtual bool does_ray_intersect(const IRayPtr &ray, const IHitRecordPtr &hit_record) const = 0;
};

#endif //SIMPLE_RAY_TRACING_I_RAY_INTERSECTION_H
