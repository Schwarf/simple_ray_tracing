//
// Created by andreas on 11.10.21.
//

#ifndef CUBIC_EQUATION_H
#define CUBIC_EQUATION_H

#include "interfaces/i_solve.h"
#include "quadratic_equation.h"
#include "validate.h"
#include "stdexcept"

template<typename T>
class CubicEquation: public ISolve
{
public:
	CubicEquation(const T &cubic_coefficient,
				  const T &quadratic_coefficient,
				  const T &linear_coefficient,
				  const T &constant);
	size_t number_of_solutions();
	void set_epsilon(const double &epsilon);
	T get_solution(size_t number_of_solution);
private:
	double epsilon_{0.00001};
	Validate validate_;
	std::vector<T> solutions_;
};

template<typename T>
CubicEquation<T>::CubicEquation(const T &cubic_coefficient,
								const T &quadratic_coefficient,
								const T &linear_coefficient,
								const T &constant)
{
	validate_.is_above_threshold("cubic_coefficient", std::abs(cubic_coefficient), epsilon_, "CubicEquation");
	if (std::abs(constant) < epsilon) {
		solutions_.push_back(0.0);
		QuadraticEquation<T> quadratic_equation(cubic_coefficient, quadratic_coefficient, linear_coefficient);
		for (int i = 0; i < quadratic_equation.number_of_solutions(); ++i) {
			solutions_.push_back(quadratic_equation.get_solution(i));
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

	if (std::abs(p) < epsilon_ )
	{
		if(std::abs(q) < epsilon_)
		{
			solutions_.push_back(-reduced_quadratic_coefficient/3.0);
			return;
		}
		auto exponent = 1./3.;
		auto argument = reduced_quadratic_coefficient*reduced_quadratic_coefficient *reduced_quadratic_coefficient-27.0*reduced_constant;
		auto solution = 1./3. * (std::pow(argument, exponent) - reduced_quadratic_coefficient);
		solutions_.push_back(solution);
		return;
	}




}

template<typename T>
int CubicEquation<T>::number_of_solutions()
{
	return number_of_solutions_;
}

template<typename T>
void CubicEquation<T>::set_epsilon(const double &epsilon)
{
	epsilon_ = epsilon;
}

template<typename T>
T CubicEquation<T>::get_solution(int number_of_solution)
{
	validate_.is_below_threshold("number_of_solution", number_of_solution, 3, "CubicEquation")
	validate_.is_above_threshold("number_of_solution", number_of_solution, -1, "CubicEquation")
	return solutions_[number_of_solution];
}

#endif //CUBIC_EQUATION_H
