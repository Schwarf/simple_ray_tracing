//
// Simpl
// Created by andreas on 02.10.21.
//

#ifndef SOURCE_C_VECTOR_H
#define SOURCE_C_VECTOR_H

#include <cmath>
#include <cassert>
#include <iostream>

// Definition of N_Tuple
template<size_t dimension, typename T>
struct N_Tuple
{
	T elements[dimension] = {};

	T &operator[](const size_t index)
	{
		if (index >= dimension) {
			const std::string message_part1 =
				"In class N_Tuple 'index = " + std::to_string(index) + "' is out of range: T & operator[]";
			const std::string message_part2 = "Dimension of N_Tuple is: " + std::to_string(dimension);
			std::string message = message_part1 + message_part2;
			throw std::out_of_range(message);
		}
		return elements[index];
	}

	const T &operator[](const size_t index) const
	{
		if (index >= dimension) {
			const std::string message_part1 =
				"In class N_Tuple 'index = " + std::to_string(index) + "' is out of range: const T & operator[]";
			const std::string message_part2 = "Dimension of N_Tuple is: " + std::to_string(dimension);
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
	N_Tuple<dimension, T> &normalize()
	{
		T norm = this->norm();
		*this = (*this) / norm;
		return *this;
	}

};

// Multiply N_Tuple with number
template<size_t dimension, typename T>
N_Tuple<dimension, T> operator*(const N_Tuple<dimension, T> &lhs, const T rhs)
{
	N_Tuple<dimension, T> result;
	for (size_t index = dimension; index--; result[index] = lhs[index] * rhs);
	return result;
}

template<size_t dimension, typename T>
N_Tuple<dimension, T> operator*(const T lhs, const N_Tuple<dimension, T> &rhs)
{
	return rhs * lhs;
}

template<size_t dimension, typename T>
N_Tuple<dimension, T> operator/(const N_Tuple<dimension, T> &lhs, const T rhs)
{
	N_Tuple<dimension, T> result;
	for (size_t index = dimension; index--; result[index] = lhs[index] / rhs);
	return result;
}

// Dot-product
template<size_t dimension, typename T>
T operator*(const N_Tuple<dimension, T> &lhs, const N_Tuple<dimension, T> &rhs)
{
	T result{};
	for (size_t index = dimension; index--; result += lhs[index] * rhs[index]);
	return result;
}

// Vector addition
template<size_t dimension, typename T>
N_Tuple<dimension, T> operator+(N_Tuple<dimension, T> lhs, const N_Tuple<dimension, T> &rhs)
{
	for (size_t index = dimension; index--; lhs[index] += rhs[index]);
	return lhs;
}

// Vector subtraction
template<size_t dimension, typename T>
N_Tuple<dimension, T> operator-(N_Tuple<dimension, T> lhs, const N_Tuple<dimension, T> &rhs)
{
	for (size_t index = dimension; index--; lhs[index] -= rhs[index]);
	return lhs;
}

// Invert direction_normalized
template<size_t dimension, typename T>
N_Tuple<dimension, T> operator-(const N_Tuple<dimension, T> &lhs)
{
	N_Tuple<dimension, T> result;
	for (size_t index = dimension; index--; result[index] = -lhs[index]);
	return result;
}

template<size_t dimension, typename T>
std::ostream &operator<<(std::ostream &out, const N_Tuple<dimension, T> &v)
{
	for (size_t index = 0; index < dimension; index++)
		out << v[index] << " ";
	return out;

}
using float_duple = N_Tuple<2, float> ;
using FloatTriple =  N_Tuple<3, float>;
using float_quadruple =  N_Tuple<4, float>;

using Point3D = FloatTriple;
using Vector3D = FloatTriple;
using Color = FloatTriple;

#endif //SOURCE_C_VECTOR_H
