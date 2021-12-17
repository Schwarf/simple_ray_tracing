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

std::unique_ptr<std::vector<Color>> ImageBuffer::buffer()
{
	return std::make_unique<std::vector<Color>>(buffer_);
}

ImageBuffer::ImageBuffer(const int width, const int height)
	:
	width_(width),
	height_(height)
{
	buffer_.resize(height_ * width_);
}

void ImageBuffer::set_pixel_value(int width_index, int height_index, const Color &pixel_color_value, int samples_per_pixel)
{
	auto scale = 1.f/float(samples_per_pixel);
	auto pixel_color_mean_value = pixel_color_value*scale;
	// Use gamma correction with gamma =2 --> color^(1/gamma) = sqrt
	// auto pixel_color_mean_value = Color{std::sqrt(pixel_color_value[0]*scale),
	//										std::sqrt(pixel_color_value[1]*scale),
	//										std::sqrt(pixel_color_value[2]*scale)};

	buffer_[width_index + height_index * width_] = pixel_color_mean_value;
}

Color ImageBuffer::get_rgb_pixel(int index)
{
	auto red = (255 * std::max(0.f, std::min(1.f, buffer_[index][0])));
	auto green = (255 * std::max(0.f, std::min(1.f, buffer_[index][1])));
	auto blue = (255 * std::max(0.f, std::min(1.f, buffer_[index][2])));
	return Color{red, green, blue};
}

Color ImageBuffer::get_rgb_pixel(int width_index, int height_index)
{
	int index = width_index + height_index * width_;
	return get_rgb_pixel(index);
}
