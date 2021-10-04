//
// Simpl
// Created by andreas on 02.10.21.
//

#ifndef SOURCE_C_VECTOR_H
#define SOURCE_C_VECTOR_H

#include <cmath>
#include <cassert>
#include <iostream>

// Definition of c_vector
template<size_t dimension, typename T>
struct c_vector {
    T &operator[](const size_t index) {
        static_assert(index < dimension);
        return elements[index];
    }

    const T &operator[](const size_t index) const {
        static_assert(index < dimension);
        return elements[index];
    }

    T elements[dimension] = {};
};

// Multiply c_vector with number
template<size_t dimension, typename T>
c_vector<dimension, T> operator*(const c_vector<dimension, T> &lhs, const float rhs) {
    c_vector<dimension, T> result;
    for (size_t index = dimension; index--; result[index] = lhs[index] * rhs);
    return result;
}

// Dot-product
template<size_t dimension, typename T>
T operator*(const c_vector<dimension, T> &lhs, const c_vector<dimension, T> &rhs) {
    T result{};
    for (size_t index = dimension; index--; result += lhs[index] * rhs[index]);
    return result;
}

// Vector addition
template<size_t dimension,  typename T>
c_vector<dimension, T> operator+(c_vector <dimension, T> lhs, const c_vector<dimension, T> &rhs) {
    for (size_t index = dimension; index--; lhs[index] += rhs[index]);
    return lhs;
}

// Vector subtraction
template<size_t dimension, typename T>
c_vector<dimension, T> operator-(c_vector <dimension, T> lhs, const c_vector<dimension, T> &rhs) {
    for (size_t index = dimension; index--; lhs[index] -= rhs[index]);
    return lhs;
}

// Invert direction_normalized
template<size_t dimension, typename T>
c_vector<dimension, T> operator-(const c_vector<dimension, T> &lhs) {
    return lhs * (-1.f);
}

// Specialization for 3-dimensional vector (provided x,y,z access and norm)
template<>
struct c_vector<3, float> {
    float &operator[](const size_t index) {
        assert(index < 3);
        return index == 0 ? x : (1 == index ? y : z);
    }

    const float &operator[](const size_t index) const {
        assert(index < 3);
        return index == 0 ? x : (1 == index ? y : z);
    }

    float norm() { return std::sqrt(x * x + y * y + z * z); }

    c_vector<3, float> &normalize(float l = 1.0) {
        *this = (*this) * (l / norm());
        return *this;
    }

    float x = 0;
    float y = 0;
    float z = 0;
};

typedef c_vector<3, float> c_vector3;
//typedef c_vector<3, char> c_char_vector3;

template<size_t dimension, typename T>
std::ostream &operator<<(std::ostream &out, const c_vector<dimension, T> &v) {
    for (size_t index = 0; index < dimension; index++)
        out << v[index] << " ";
    return out;

}

#endif //SOURCE_C_VECTOR_H
