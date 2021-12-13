//
// Created by andreas on 11.10.21.
//

#ifndef QUADRATIC_EQUATION_H
#define QUADRATIC_EQUATION_H

#include "interfaces/i_solve.h"
#include "templates/c_vector.h"
#include "validate.h"
#include "stdexcept"
#include <limits>

template<typename T>
class QuadraticEquation: public ISolve<2, T>
{
public:
	QuadraticEquation(const c_vector<3, T> & coefficients, const T & epsilon);
	virtual c_vector<2, T> solutions() final;
	virtual size_t number_of_solutions() const final;

private:
	c_vector<2, T> solutions_{std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN()};
	size_t number_of_solutions_{};
};

template<typename T>
QuadraticEquation<T>::QuadraticEquation(const c_vector<3, T> & coefficients, const T & epsilon)
{
	T absolute_average{};
	for(size_t index = 0; index < 3; index++)
	{
		absolute_average += std::abs(coefficients[index]);
	}
	absolute_average /= 3.f;
	Validate<T>::is_not_zero("quadratic_coefficient", coefficients[0]/absolute_average, epsilon, "QuadraticEquation");

	T quadratic_coefficient = coefficients[0];
	T linear_coefficient = coefficients[1];
	T constant = coefficients[2];

	if (std::abs(constant) < epsilon) {
		solutions_[0] = T{};
		solutions_[1] = -linear_coefficient / quadratic_coefficient;
		number_of_solutions_ = 2;
		return;
	}

	auto p = linear_coefficient / quadratic_coefficient;
	auto q = constant / quadratic_coefficient;
	auto discriminant = (p / 2.) * (p / 2.) - q;
	if (discriminant < 0) {
		return;
	}
	if (std::abs(discriminant) < epsilon) {
		solutions_[0] = -p / 2.;
		number_of_solutions_ = 1;
		return;
	}
	solutions_[0]  = -p / 2. + std::sqrt(discriminant);
	solutions_[1] = -p / 2. - std::sqrt(discriminant);
	number_of_solutions_ = 2;
}
template<typename T>
c_vector<2, T> QuadraticEquation<T>::solutions()
{
	return solutions_;
}
template<typename T>
size_t QuadraticEquation<T>::number_of_solutions() const
{
	return number_of_solutions_;
}


#endif //QUADRATIC_EQUATION_H
