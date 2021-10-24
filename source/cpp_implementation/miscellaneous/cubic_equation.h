//
// Created by andreas on 11.10.21.
//

#ifndef CUBIC_EQUATION_H
#define CUBIC_EQUATION_H

#include "interfaces/i_solve.h"
#include "quadratic_equation.h"
#include "validate.h"
#include "stdexcept"
#include "limits"

template<typename T>
class CubicEquation: public ISolve<3, T>
{
public:
	CubicEquation(const c_vector<4, T> & coefficients, const T & epsilon);
	c_vector<3, T> solutions() final;

private:
	Validate validate_;
	c_vector<3, T> solutions_;
};

template<typename T>
CubicEquation<T>::CubicEquation(const c_vector<4, T> & coefficients, const T & epsilon)
{
	validate_.is_above_threshold("cubic_coefficient", std::abs(coefficients[0]), epsilon, "CubicEquation");
	T cubic_coefficient = coefficients[0];
	T quadratic_coefficient = coefficients[1];
	T linear_coefficient = coefficients[2];
	T constant = coefficients[3];


	if (std::abs(constant) < epsilon) {
		solutions_[0] = T{};
		auto coeff = c_vector<3, T>{cubic_coefficient, quadratic_coefficient, linear_coefficient};
		auto quadratic_equation = QuadraticEquation<T>(coeff, epsilon);
		for (int i = 0; i < 2; ++i) {
			solutions_[i+1] = quadratic_equation.solutions()[i];
		}
	}

	auto reduced_quadratic_coefficient = quadratic_coefficient / cubic_coefficient;
	auto reduced_linear_coefficient = linear_coefficient / cubic_coefficient;
	auto reduced_constant = constant / cubic_coefficient;

	auto p = reduced_quadratic_coefficient - reduced_linear_coefficient * reduced_linear_coefficient / 3.0;
	auto q = 2.0 * reduced_linear_coefficient * reduced_linear_coefficient * reduced_linear_coefficient / 27.0 -
		reduced_linear_coefficient * reduced_quadratic_coefficient / 3.0 + reduced_constant;

	auto discriminant = q* q / 4.0 + p * p * p / 3.0;
	if (discriminant > 0.0)
		return;

	if (std::abs(p) < epsilon )
	{
		if(std::abs(q) < epsilon)
		{
			solutions_[0] = (-reduced_quadratic_coefficient/3.0);
			solutions_[1] = std::numeric_limits<T>::quiet_NaN();
			solutions_[2] = std::numeric_limits<T>::quiet_NaN();
			solutions_[3] = std::numeric_limits<T>::quiet_NaN();
			return;
		}
		auto exponent = 1./3.;
		auto argument = reduced_quadratic_coefficient*reduced_quadratic_coefficient *reduced_quadratic_coefficient-27.0*reduced_constant;
		solutions_[0] = 1./3. * (std::pow(argument, exponent) - reduced_quadratic_coefficient);
		return;
	}




}
template<typename T>
c_vector<3, T> CubicEquation<T>::solutions()
{
	return c_vector<3, T>();
}

#endif //CUBIC_EQUATION_H
