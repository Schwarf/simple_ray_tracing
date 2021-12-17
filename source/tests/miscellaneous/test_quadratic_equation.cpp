//
// Created by andreas on 07.11.21.
//

#include "gtest/gtest.h"
#include "miscellaneous/quadratic_equation.h"
#include <regex>

class SetupQuadraticEquation: public testing::Test
{
protected:
	double a1_{1.2111111999};
	double b1_{-1.731131};
	double c1_{2.612123121};

	double a2_{-21.99773131};
	double b2_{-2.151890121};
	double c2_{3.81123191977};
	std::vector<double> solutions2_{-0.46801548, 0.3701922};

	double a3_{-1.99773131e-10};
	double b3_{-2.151890121};
	double c3_{3.81123191977};

	double a4_{3.121};
	double b4_{0};
	double c4_{-18.9009};
	std::vector<double> solutions4_{-2.46090222,  2.46090222};

	double a5_{-179.121};
	double b5_{+189.9071};
	double c5_{0};
	std::vector<double> solutions5_{1.06021684, 0.};

	double epsilon_{1e-8};

	bool check_for_solution(double value, std::vector<double> &solutions) const
	{

		auto epsilon = epsilon_;
		for (const auto &element: solutions) {
			if(std::abs(value) < epsilon)
			{
				if(element < epsilon)
					return true;
			}
			auto check = std::abs(1. - element / value);
			if (check < epsilon) {
				return true;
			}
		}
		return false;
	}

};

TEST_F(SetupQuadraticEquation, quadratic_equation_no_solution)
{
	N_Tuple<3, double> coefficients{a1_, b1_, c1_};
	auto solver = QuadraticEquation<double>(coefficients, epsilon_);
	EXPECT_EQ(solver.number_of_solutions(), 0);
}


TEST_F(SetupQuadraticEquation, regular_quadratic_equation)
{
	N_Tuple<3, double> coefficients{a2_, b2_, c2_};
	auto solver = QuadraticEquation<double>(coefficients, epsilon_);
	EXPECT_EQ(solver.number_of_solutions(), 2);
	EXPECT_TRUE(check_for_solution(solver.solutions()[0], solutions2_));
	EXPECT_TRUE(check_for_solution(solver.solutions()[1], solutions2_));
}


TEST_F(SetupQuadraticEquation, quadratic_equation_very_small_quadratic_coefficient)
{
	N_Tuple<3, double> coefficients{a3_, b3_, c3_};
	auto value = a3_ / (a3_ + b3_ + c3_) / 3.;
	std::string message =
		"In class QuadraticEquation quadratic_coefficient=" + std::to_string(value) + " is zero (within epsilon) "
			+ std::to_string(epsilon_);
	try {
		auto solver = QuadraticEquation<double>(coefficients, epsilon_);
		FAIL() << "Expected std::out_of_range";
	}
	catch (std::out_of_range const &err) {
		EXPECT_TRUE(err.what() == message);
	}
}

TEST_F(SetupQuadraticEquation, quadratic_equation_linear_coeeficient_is_zero)
{
	N_Tuple<3, double> coefficients{a4_, b4_, c4_};
	auto solver = QuadraticEquation<double>(coefficients, epsilon_);
	EXPECT_EQ(solver.number_of_solutions(), 2);
	EXPECT_TRUE(check_for_solution(solver.solutions()[0], solutions4_));
	EXPECT_TRUE(check_for_solution(solver.solutions()[1], solutions4_));
}

TEST_F(SetupQuadraticEquation, quadratic_equation_constant_is_zero)
{
	N_Tuple<3, double> coefficients{a5_, b5_, c5_};
	auto solver = QuadraticEquation<double>(coefficients, epsilon_);
	EXPECT_EQ(solver.number_of_solutions(), 2);
	EXPECT_TRUE(check_for_solution(solver.solutions()[0], solutions5_));
	EXPECT_TRUE(check_for_solution(solver.solutions()[1], solutions5_));
}
