//
// Created by andreas on 16.10.21.
//

#include "object_list.h"
void ObjectList::add_object(const ITargetObjectPtr &target_object)
{
	id_to_object_map_[target_object->object_id()] = target_object;
}
size_t ObjectList::number_of_objects()
{
	return id_to_object_map_.size();
}
ITargetObjectPtr ObjectList::object(size_t id)
{
	if (id_to_object_map_.find(id) == id_to_object_map_.end()) {
		return nullptr;
	}
	return id_to_object_map_[id];
}

ITargetObjectPtr ObjectList::get_object_hit_by_ray(const IRayPtr &ray, const IHitRecordPtr &hit_record)
{
	if(most_recently_hit_object_  && most_recently_hit_object_->does_ray_intersect(ray, hit_record))
		return most_recently_hit_object_;

	for (const auto &id_object: id_to_object_map_) {
		if (id_object.second->does_ray_intersect(ray, hit_record)) {
			most_recently_hit_object_ = id_object.second;
			return most_recently_hit_object_;
		}
	}
	most_recently_hit_object_ = nullptr;
	return nullptr;
}
