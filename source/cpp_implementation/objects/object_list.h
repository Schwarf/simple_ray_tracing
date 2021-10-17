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
	void add_object(std::shared_ptr<ITargetObject> target_object) final;
	size_t number_of_objects() final;
	~ObjectList() override = default;
	std::shared_ptr<ITargetObject> object(size_t index) final;
private:
	std::vector<std::shared_ptr<ITargetObject>> object_vector_;
};


#endif //OBJECT_LIST_H
