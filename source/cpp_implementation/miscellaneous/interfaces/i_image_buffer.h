//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_IMAGE_BUFFER_H
#define SIMPLE_RAY_TRACING_I_IMAGE_BUFFER_H
#include "miscellaneous/templates/n_tuple.h"
#include <vector>

class IImageBuffer
{
public:
	virtual int width() const = 0;
	virtual int height() const = 0;
	virtual void set_pixel_value(int width_index, int height_index, const Color &pixel_color_value, int samples_per_pixel) = 0;
	virtual Color get_rgb_pixel(int index) = 0;
	virtual Color get_rgb_pixel(int width_index, int height_index) = 0;
	virtual std::unique_ptr<std::vector<Color>> buffer() = 0;

};
#endif //SIMPLE_RAY_TRACING_I_IMAGE_BUFFER_H
