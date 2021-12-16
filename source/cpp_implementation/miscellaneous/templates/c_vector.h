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
struct c_vector
{
	T elements[dimension] = {};

	T &operator[](const size_t index)
	{
		if (index >= dimension) {
			const std::string message_part1 =
				"In class c_vector 'index = " + std::to_string(index) + "' is out of range: T & operator[]";
			const std::string message_part2 = "Dimension of c_vector is: " + std::to_string(dimension);
			std::string message = message_part1 + message_part2;
			throw std::out_of_range(message);
		}
		return elements[index];
	}

	const T &operator[](const size_t index) const
	{
		if (index >= dimension) {
			const std::string message_part1 =
				"In class c_vector 'index = " + std::to_string(index) + "' is out of range: const T & operator[]";
			const std::string message_part2 = "Dimension of c_vector is: " + std::to_string(dimension);
			std::string message = message_part1 + message_part2;
			throw std::out_of_range(message);
		}
		return elements[index];
	}

	T norm()
	{
		T result{};
		for (size_t index = dimension; index--; result = *this * (*this));
		return std::sqrt(result);
	}
	c_vector<dimension, T> &normalize()
	{
		T norm = this->norm();
		*this = (*this) / norm;
		return *this;
	}

};

// Multiply c_vector with number
template<size_t dimension, typename T>
c_vector<dimension, T> operator*(const c_vector<dimension, T> &lhs, const T rhs)
{
	c_vector<dimension, T> result;
	for (size_t index = dimension; index--; result[index] = lhs[index] * rhs);
	return result;
}

template<size_t dimension, typename T>
c_vector<dimension, T> operator*(const T lhs, const c_vector<dimension, T> &rhs)
{
	return rhs * lhs;
}

template<size_t dimension, typename T>
c_vector<dimension, T> operator/(const c_vector<dimension, T> &lhs, const T rhs)
{
	c_vector<dimension, T> result;
	for (size_t index = dimension; index--; result[index] = lhs[index] / rhs);
	return result;
}

// Dot-product
template<size_t dimension, typename T>
T operator*(const c_vector<dimension, T> &lhs, const c_vector<dimension, T> &rhs)
{
	T result{};
	for (size_t index = dimension; index--; result += lhs[index] * rhs[index]);
	return result;
}

// Vector addition
template<size_t dimension, typename T>
c_vector<dimension, T> operator+(c_vector<dimension, T> lhs, const c_vector<dimension, T> &rhs)
{
	for (size_t index = dimension; index--; lhs[index] += rhs[index]);
	return lhs;
}

// Vector subtraction
template<size_t dimension, typename T>
c_vector<dimension, T> operator-(c_vector<dimension, T> lhs, const c_vector<dimension, T> &rhs)
{
	for (size_t index = dimension; index--; lhs[index] -= rhs[index]);
	return lhs;
}

// Invert direction_normalized
template<size_t dimension, typename T>
c_vector<dimension, T> operator-(const c_vector<dimension, T> &lhs)
{
	c_vector<dimension, T> result;
	for (size_t index = dimension; index--; result[index] = -lhs[index]);
	return result;
}

typedef c_vector<3, float> c_vector3;

template<size_t dimension, typename T>
std::ostream &operator<<(std::ostream &out, const c_vector<dimension, T> &v)
{
	for (size_t index = 0; index < dimension; index++)
		out << v[index] << " ";
	return out;

}
using Point3D = c_vector3;
using Vector3D = c_vector3;
using Color = c_vector3;

#endif //SOURCE_C_VECTOR_H
