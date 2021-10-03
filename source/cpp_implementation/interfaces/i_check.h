//
// Created by andreas on 02.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_CHECK_H
#define SIMPLE_RAY_TRACING_I_CHECK_H
#include <string>

class ICheck {
public:
    virtual void is_above_threshold(const std::string & variable_name, const float & variable_value, const float & threshold) const= 0;
//    virtual bool is_below_threshold(const std::string & variable_name, const float & variable_value, const float & threshold) = 0;
    virtual ~ICheck()= default;
};

#endif //SIMPLE_RAY_TRACING_I_CHECK_H
