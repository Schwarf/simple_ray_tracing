//
// Created by andreas on 16.10.21.
//

#include "object_list.h"
void ObjectList::add_object(std::shared_ptr<ITargetObject> target_object)
{
	object_vector_.push_back(target_object);

}
size_t ObjectList::number_of_objects()
{
	return object_vector_.size();
}
std::shared_ptr<ITargetObject> ObjectList::object(size_t index)
{
	if(index >= object_vector_.size())
	{
		return nullptr;
	}
	return object_vector_[index];
}
