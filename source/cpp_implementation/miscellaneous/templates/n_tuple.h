//
// Simpl
// Created by andreas on 02.10.21.
//

#ifndef SOURCE_C_VECTOR_H
#define SOURCE_C_VECTOR_H

#include <cmath>
#include <cassert>
#include <iostream>

// Definition of n_tuple
template<size_t dimension, typename T>
struct n_tuple
{
	T elements[dimension] = {};

	T &operator[](const size_t index)
	{
		if (index >= dimension) {
			const std::string message_part1 =
				"In class n_tuple 'index = " + std::to_string(index) + "' is out of range: T & operator[]";
			const std::string message_part2 = "Dimension of n_tuple is: " + std::to_string(dimension);
			std::string message = message_part1 + message_part2;
			throw std::out_of_range(message);
		}
		return elements[index];
	}

	const T &operator[](const size_t index) const
	{
		if (index >= dimension) {
			const std::string message_part1 =
				"In class n_tuple 'index = " + std::to_string(index) + "' is out of range: const T & operator[]";
			const std::string message_part2 = "Dimension of n_tuple is: " + std::to_string(dimension);
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
	n_tuple<dimension, T> &normalize()
	{
		T norm = this->norm();
		*this = (*this) / norm;
		return *this;
	}

};

// Multiply n_tuple with number
template<size_t dimension, typename T>
n_tuple<dimension, T> operator*(const n_tuple<dimension, T> &lhs, const T rhs)
{
	n_tuple<dimension, T> result;
	for (size_t index = dimension; index--; result[index] = lhs[index] * rhs);
	return result;
}

template<size_t dimension, typename T>
n_tuple<dimension, T> operator*(const T lhs, const n_tuple<dimension, T> &rhs)
{
	return rhs * lhs;
}

template<size_t dimension, typename T>
n_tuple<dimension, T> operator/(const n_tuple<dimension, T> &lhs, const T rhs)
{
	n_tuple<dimension, T> result;
	for (size_t index = dimension; index--; result[index] = lhs[index] / rhs);
	return result;
}

// Dot-product
template<size_t dimension, typename T>
T operator*(const n_tuple<dimension, T> &lhs, const n_tuple<dimension, T> &rhs)
{
	T result{};
	for (size_t index = dimension; index--; result += lhs[index] * rhs[index]);
	return result;
}

// Vector addition
template<size_t dimension, typename T>
n_tuple<dimension, T> operator+(n_tuple<dimension, T> lhs, const n_tuple<dimension, T> &rhs)
{
	for (size_t index = dimension; index--; lhs[index] += rhs[index]);
	return lhs;
}

// Vector subtraction
template<size_t dimension, typename T>
n_tuple<dimension, T> operator-(n_tuple<dimension, T> lhs, const n_tuple<dimension, T> &rhs)
{
	for (size_t index = dimension; index--; lhs[index] -= rhs[index]);
	return lhs;
}

// Invert direction_normalized
template<size_t dimension, typename T>
n_tuple<dimension, T> operator-(const n_tuple<dimension, T> &lhs)
{
	n_tuple<dimension, T> result;
	for (size_t index = dimension; index--; result[index] = -lhs[index]);
	return result;
}

template<size_t dimension, typename T>
std::ostream &operator<<(std::ostream &out, const n_tuple<dimension, T> &v)
{
	for (size_t index = 0; index < dimension; index++)
		out << v[index] << " ";
	return out;

}
typedef n_tuple<2, float> float_duple;
typedef n_tuple<3, float> float_triple;
typedef n_tuple<4, float> float_quadruple;

using Point3D = float_triple;
using Vector3D = float_triple;
using Color = float_triple;

#endif //SOURCE_C_VECTOR_H
