//
// Created by andreas on 16.10.21.
//

#ifndef OBJECT_LIST_H
#define OBJECT_LIST_H

#include "interfaces/i_object_list.h"
#include <iostream>
#include <unordered_map>


class ObjectList: public IObjectList
{
public:
	ObjectList() = default;
	void add_object(const ITargetObjectPtr &target_object) final;
	size_t number_of_objects() final;
	~ObjectList() override = default;
	ITargetObjectPtr object(size_t id) final;
	ITargetObjectPtr get_object_hit_by_ray(IRay &ray, IHitRecord &hit_record) final;
private:
	std::unordered_map<size_t, ITargetObjectPtr> id_to_object_map_;
	ITargetObjectPtr most_recently_hit_object_ = nullptr;

};


#endif //OBJECT_LIST_H
