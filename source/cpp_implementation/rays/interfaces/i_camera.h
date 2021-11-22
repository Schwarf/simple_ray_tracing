//
// Created by andreas on 22.11.21.
//

#ifndef I_CAMERA_H
#define I_CAMERA_H
#include "i_ray.h"
#include <memory>

class ICamera
{
public:
	virtual std::shared_ptr<IRay> get_ray(float width_coordinate, float height_coordinate) = 0;
	virtual int image_width() = 0;
	virtual int image_height() = 0;
	virtual float aspect_ratio() = 0;
	virtual float focal_length() = 0;

};

#endif //I_CAMERA_H
