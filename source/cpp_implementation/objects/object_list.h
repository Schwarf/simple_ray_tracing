//
// Created by andreas on 16.10.21.
//

#ifndef OBJECT_LIST_H
#define OBJECT_LIST_H

#include "interfaces/i_object_list.h"
#include <vector>

class ObjectList: public IObjectList
{
public:
	ObjectList() = default;
	void add_object(ITargetObjectPtr target_object) final;
	size_t number_of_objects() final;
	~ObjectList() override = default;
	ITargetObjectPtr object(size_t index) final;
	ITargetObjectPtr get_object_hit_by_ray(const IRayPtr &ray, const std::shared_ptr<
		IHitRecord> &hit_record) final;
private:
	std::vector<ITargetObjectPtr> object_vector_;
};


#endif //OBJECT_LIST_H
