//
// Created by andreas on 02.10.21.
//

#ifndef SIMPLE_RAY_TRACING_I_CHECK_H
#define SIMPLE_RAY_TRACING_I_CHECK_H

class ICheck {
public:
    virtual bool check() = 0;
    virtual ~ICheck()= default;
};

#endif //SIMPLE_RAY_TRACING_I_CHECK_H
