//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_IMAGE_BUFFER_H
#define SIMPLE_RAY_TRACING_I_IMAGE_BUFFER_H
#include "c_vector.h"
#include <vector>
class IImageBuffer{
public:
    virtual int width() const = 0;
    virtual int height() const= 0;
    virtual std::vector<c_vector3> buffer() = 0;
};
#endif //SIMPLE_RAY_TRACING_I_IMAGE_BUFFER_H
