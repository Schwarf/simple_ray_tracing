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
template<size_t dimension>
struct c_vector {
    float &operator[](const size_t index) {
        static_assert(index < dimension);
        return elements[index];
    }

    const float &operator[](const size_t index) const {
        static_assert(index < dimension);
        return elements[index];
    }

    float elements[dimension] = {};
};

// Multiply c_vector with number
template<size_t dimension>
c_vector<dimension> operator*(const c_vector<dimension> &lhs, const float rhs) {
    c_vector<dimension> result;
    for (size_t index = dimension; index--; result[index] = lhs[index] * rhs);
    return result;
}

// Dot-product
template<size_t dimension>
float operator*(const c_vector<dimension> &lhs, const c_vector<dimension> &rhs) {
    float result = 0;
    for (size_t index = dimension; index--; result += lhs[index] * rhs[index]);
    return result;
}

// Vector addition
template<size_t dimension>
c_vector<dimension> operator+(c_vector <dimension> lhs, const c_vector<dimension> &rhs) {
    for (size_t index = dimension; index--; lhs[index] += rhs[index]);
    return lhs;
}

// Vector subtraction
template<size_t dimension>
c_vector<dimension> operator-(c_vector <dimension> lhs, const c_vector<dimension> &rhs) {
    for (size_t index = dimension; index--; lhs[index] -= rhs[index]);
    return lhs;
}

// Invert direction
template<size_t dimension>
c_vector<dimension> operator-(const c_vector<dimension> &lhs) {
    return lhs * (-1.f);
}

// Specialization for 3-dimensional vector (provided x,y,z access and norm)
template<>
struct c_vector<3> {
    float &operator[](const size_t index) {
        assert(index < 3);
        return index == 0 ? x : (1 == index ? y : z);
    }

    const float &operator[](const size_t index) const {
        assert(index < 3);
        return index == 0 ? x : (1 == index ? y : z);
    }

    float norm() { return std::sqrt(x * x + y * y + z * z); }

    c_vector<3> &normalize(float l = 1.0) {
        *this = (*this) * (l / norm());
        return *this;
    }

    float x = 0;
    float y = 0;
    float z = 0;
};

typedef c_vector<3> c_vector3;
typedef c_vector<4> c_vector4;

// Cross Product only defined for 3d vectors
c_vector3 cross_product(c_vector3 v1, c_vector3 v2) {
    return {v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x};
}

// output vector
template<size_t dimension>
std::ostream &operator<<(std::ostream &out, const c_vector<dimension> &v) {
    for (size_t index = 0; index < dimension; index++)
        out << v[index] << " ";
    return out;
}


#endif //SOURCE_C_VECTOR_H
