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
	c_vector<2, T> solutions() final;

private:
	Validate validate_;
	c_vector<2, T> solutions_{};
};

template<typename T>
QuadraticEquation<T>::QuadraticEquation(const c_vector<3, T> & coefficients, const T & epsilon)
{

	validate_
		.is_above_threshold("quadratic_coefficient", std::abs(coefficients[0]), epsilon, "QuadraticEquation");

	T quadratic_coefficient = coefficients[0];
	T linear_coefficient = coefficients[1];
	T constant = coefficients[2];

	if (std::abs(constant) < epsilon) {
		solutions_[0] = T{};
		solutions_[1] = -linear_coefficient / quadratic_coefficient;
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
		solutions_[1] = std::numeric_limits<T>::quiet_NaN();
		return;
	}
	solutions_[0]  = -p / 2. + std::sqrt(discriminant);
	solutions_[1] = -p / 2. - std::sqrt(discriminant);
}
template<typename T>
c_vector<2, T> QuadraticEquation<T>::solutions()
{
	return solutions_;
}


#endif //QUADRATIC_EQUATION_H
