//
// Created by andreas on 16.10.21.
//

#include "object_list.h"
void ObjectList::add_object(ITargetObjectPtr target_object)
{
	object_vector_.push_back(target_object);

}
size_t ObjectList::number_of_objects()
{
	return object_vector_.size();
}
ITargetObjectPtr ObjectList::object(size_t index)
{
	if (index >= object_vector_.size()) {
		return nullptr;
	}
	return object_vector_[index];
}

ITargetObjectPtr ObjectList::get_object_hit_by_ray(const std::shared_ptr<IRay> &ray, const std::shared_ptr<
	IHitRecord> &hit_record)
{
	for(const auto & object : object_vector_)
	{
		if(object->does_ray_intersect(ray, hit_record))
			return object;
	}
	return nullptr;
}
