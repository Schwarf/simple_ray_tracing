//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_IMAGE_BUFFER_H
#define SIMPLE_RAY_TRACING_IMAGE_BUFFER_H

#include <memory>
#include "miscellaneous/interfaces/i_image_buffer.h"
#include "miscellaneous/templates/c_vector.h"
class ImageBuffer final: public IImageBuffer
{

public:
	ImageBuffer(int width, int height);
	~ImageBuffer() = default;
	int width() const final;
	int height() const final;
	std::unique_ptr<std::vector<c_vector3>> buffer() final;
	void set_pixel_value(int width_index, int height_index, const c_vector3 &pixel_color_value, int samples_per_pixel) final;
	c_vector3 get_rgb_pixel(int index) final;
	c_vector3 get_rgb_pixel(int width_index, int height_index) final;

private:
	std::vector<c_vector3> buffer_;
	int width_;
	int height_;
};


#endif //SIMPLE_RAY_TRACING_IMAGE_BUFFER_H
