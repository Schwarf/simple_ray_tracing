//
// Created by andreas on 16.10.21.
//

#ifndef I_OBJECT_LIST_H
#define I_OBJECT_LIST_H
#include <memory>
#include "objects/interfaces/i_target_object.h"

class IObjectList
{
public:
	virtual void add_object(const ITargetObjectPtr &target_object) = 0;
	virtual ITargetObjectPtr object(size_t id) = 0;
	virtual ITargetObjectPtr get_object_hit_by_ray(IRay &ray, IHitRecord &hit_record) = 0;
	virtual size_t number_of_objects() = 0;
	virtual ~IObjectList() = default;
};

using IObjectListPtr = std::shared_ptr<IObjectList>;

#endif //I_OBJECT_LIST_H
