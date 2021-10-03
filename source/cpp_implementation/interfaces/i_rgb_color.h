//
// Created by andreas on 03.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_RGB_COLOR_H
#define SIMPLE_RAY_TRACING_I_RGB_COLOR_H

class IRGBColor {
public:
    virtual float red_value() const = 0;

    virtual float green_value() const = 0;

    virtual float blue_value() const = 0;
    ~IRGBColor()=default;
};

#endif //SIMPLE_RAY_TRACING_I_RGB_COLOR_H

