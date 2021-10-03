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

std::vector<c_vector3> ImageBuffer::buffer() {
    return buffer_;
}

ImageBuffer::ImageBuffer(const int width, const int height) {

    width_ = width;
    height_ = height;

    buffer_.resize(height_ * width_);

    for (size_t height_index = 0; height_index < height; ++height_index) {
        for (size_t width_index = 0; width_index < width; width_index++) {
            buffer_[width_index + height_index * width] = c_vector3{float(height_index) / float(height),
                                                                    float(width_index) / float(width) *
                                                                    float(height_index) / float(height),
                                                                    float(width_index) / float(width)};

        }
    }

}
