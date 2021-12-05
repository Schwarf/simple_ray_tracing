//
// Created by andreas on 03.10.21.
//

#include "image_buffer.h"

int ImageBuffer::width() const
{
	return width_;
}

int ImageBuffer::height() const
{
	return height_;
}

std::unique_ptr<std::vector<c_vector3>> ImageBuffer::buffer()
{
	return std::make_unique<std::vector<c_vector3>>(buffer_);
}

ImageBuffer::ImageBuffer(const int width, const int height)
	:
	width_(width),
	height_(height)
{
	buffer_.resize(height_ * width_);
}

void ImageBuffer::set_pixel_value(int width_index, int height_index, const c_vector3 &pixel_color_value, int samples_per_pixel)
{
	auto scale = 1.f/float(samples_per_pixel);
	auto pixel_color_mean_value = pixel_color_value*scale;
	buffer_[width_index + height_index * width_] = pixel_color_mean_value;
}

c_vector3 ImageBuffer::get_rgb_pixel(int index)
{
	auto red = (255 * std::max(0.f, std::min(1.f, buffer_[index][0])));
	auto green = (255 * std::max(0.f, std::min(1.f, buffer_[index][1])));
	auto blue = (255 * std::max(0.f, std::min(1.f, buffer_[index][2])));
	return c_vector3{red, green, blue};
}

c_vector3 ImageBuffer::get_rgb_pixel(int width_index, int height_index)
{
	int index = width_index + height_index * width_;
	return get_rgb_pixel(index);
}
