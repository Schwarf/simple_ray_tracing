//
// Created by andreas on 03.10.21.
//

#include "image_buffer.h"

int ImageBuffer::width() const {
    return width_;
}

int ImageBuffer::height() const {
    return height_;
}

std::unique_ptr<std::vector<c_vector3>> ImageBuffer::buffer() {
    return std::make_unique<std::vector<c_vector3>>(buffer_);
}

ImageBuffer::ImageBuffer(const int width, const int height) {

    width_ = width;
    height_ = height;

    buffer_.resize(height_ * width_);
/*
    for (size_t height_index = 0; height_index < height; ++height_index) {
        for (size_t width_index = 0; width_index < width; width_index++) {
            buffer_[width_index + height_index * width] = c_vector3{float(height_index) / float(height),
                                                                    float(width_index) / float(width) *
                                                                    float(height_index) / float(height),
                                                                    float(width_index) / float(width)};

        }
    }
*/
}

void ImageBuffer::set_pixel_value(size_t width_index, size_t height_index, const c_vector3 pixel_color_value) {
    buffer_[width_index + height_index* width_] = pixel_color_value;
}

c_vector3 ImageBuffer::get_pixel(size_t index) {
    auto red = (255 * std::max(0.f, std::min(1.f, buffer_[index][0])));
    auto green = (255 * std::max(0.f, std::min(1.f, buffer_[index][1])));
    auto blue = (255 * std::max(0.f, std::min(1.f, buffer_[index][2])));
    return c_vector3 {red, green, blue};
}
