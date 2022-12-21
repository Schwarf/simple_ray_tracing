//
// Created by andreas on 16.10.21.
//

#ifndef I_OBJECT_LIST_H
#define I_OBJECT_LIST_H
#include "objects/interfaces/i_target_object.h"
#include <memory>

class IObjectList
{
public:
	virtual void add_object(ITargetObjectPtr target_object) = 0;
	virtual ITargetObjectPtr object(size_t index) = 0;
	virtual ITargetObjectPtr get_object_hit_by_ray(const IRayPtr &ray, const std::shared_ptr<
		IHitRecord> &hit_record) = 0;
	virtual size_t number_of_objects() = 0;
	virtual ~IObjectList() = default;
};


#endif //I_OBJECT_LIST_H
