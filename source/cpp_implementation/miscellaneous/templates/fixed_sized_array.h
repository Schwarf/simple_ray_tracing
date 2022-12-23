//
// Simpl
// Created by andreas on 02.10.21.
//

#ifndef SOURCE_C_VECTOR_H
#define SOURCE_C_VECTOR_H

#include <cmath>
#include <cassert>
#include <iostream>

// Definition of FixedSizedArray
template<size_t dimension, typename T>
struct FixedSizedArray
{
	T elements[dimension] = {};

	T &operator[](const size_t index)
	{
		if (index >= dimension) {
			const std::string message_part1 =
				"In class FixedSizedArray 'index = " + std::to_string(index) + "' is out of range: T & operator[]";
			const std::string message_part2 = "Dimension of FixedSizedArray is: " + std::to_string(dimension);
			std::string message = message_part1 + message_part2;
			throw std::out_of_range(message);
		}
		return elements[index];
	}

	const T &operator[](const size_t index) const
	{
		if (index >= dimension) {
			const std::string message_part1 =
				"In class FixedSizedArray 'index = " + std::to_string(index) + "' is out of range: const T & operator[]";
			const std::string message_part2 = "Dimension of FixedSizedArray is: " + std::to_string(dimension);
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
	FixedSizedArray<dimension, T> &normalize()
	{
		T norm = this->norm();
		*this = (*this) / norm;
		return *this;
	}
	FixedSizedArray<dimension, T> & operator*=(const T rhs)
	{
		for (size_t index = dimension; index--; elements[index] *= rhs);
		return *this;
	}

};


// Multiply FixedSizedArray with number
template<size_t dimension, typename T>
FixedSizedArray<dimension, T> operator*(FixedSizedArray<dimension, T> lhs, const T rhs)
{
	lhs*=rhs;
	return lhs;
}

template<size_t dimension, typename T>
FixedSizedArray<dimension, T> operator*(const T lhs, FixedSizedArray<dimension, T> rhs)
{
	rhs*=lhs;
	return rhs;
}

template<size_t dimension, typename T>
FixedSizedArray<dimension, T> operator/(const FixedSizedArray<dimension, T> &lhs, const T rhs)
{
	FixedSizedArray<dimension, T> result;
	for (size_t index = dimension; index--; result[index] = lhs[index] / rhs);
	return result;
}

// Dot-product
template<size_t dimension, typename T>
T operator*(const FixedSizedArray<dimension, T> &lhs, const FixedSizedArray<dimension, T> &rhs)
{
	T result{};
	for (size_t index = dimension; index--; result += lhs[index] * rhs[index]);
	return result;
}

// Vector addition
template<size_t dimension, typename T>
FixedSizedArray<dimension, T> operator+(FixedSizedArray<dimension, T> lhs, const FixedSizedArray<dimension, T> &rhs)
{
	for (size_t index = dimension; index--; lhs[index] += rhs[index]);
	return lhs;
}

// Vector subtraction
template<size_t dimension, typename T>
FixedSizedArray<dimension, T> operator-(FixedSizedArray<dimension, T> lhs, const FixedSizedArray<dimension, T> &rhs)
{
	for (size_t index = dimension; index--; lhs[index] -= rhs[index]);
	return lhs;
}

// Invert direction_normalized
template<size_t dimension, typename T>
FixedSizedArray<dimension, T> operator-(const FixedSizedArray<dimension, T> &lhs)
{
	FixedSizedArray<dimension, T> result;
	for (size_t index = dimension; index--; result[index] = -lhs[index]);
	return result;
}

template<size_t dimension, typename T>
std::ostream &operator<<(std::ostream &out, const FixedSizedArray<dimension, T> &v)
{
	for (size_t index = 0; index < dimension; index++)
		out << v[index] << " ";
	return out;

}
template <size_t dimension = 3, typename T>
FixedSizedArray<dimension, T> cross_product(const FixedSizedArray<dimension, T> &first, const FixedSizedArray<dimension, T> &second)
{
	FixedSizedArray<dimension, T> result;
	result[0] = first[1]*second[2] - first[2]*second[1];
	result[1] = first[2]*second[0] - first[0]*second[2];
	result[2] = first[0]*second[1] - first[1]*second[0];
	return result;
}

using float_duple = FixedSizedArray<2, float> ;
using FloatTriple =  FixedSizedArray<3, float>;
using float_quadruple =  FixedSizedArray<4, float>;

using Point3D = FloatTriple;
using Vector3D = FloatTriple;
using Color = FloatTriple;

#endif //SOURCE_C_VECTOR_H
