//
// Created by andreas on 07.11.21.
//
#include "gtest/gtest.h"
#include "miscellaneous/quartic_equation.h"
#include <regex>

class SetupQuarticEquation: public testing::Test
{
protected:
	float a1_{3.};
	float b1_{6.};
	float c1_{-123.};
	float d1_{-126.};
	float e1_{1080};
	std::vector<float> solutions1_{-6, -4, 5, 3};

	float a2_{1.89012};
	float b2_{-2.1212};
	float c2_{-0.77878};
	float d2_{1.9089};
	float e2_{-0.94489002};
	std::vector<float> solutions2_{-0.96142702, 1.0274915};

	float a3_{3.121};
	float b3_{-12.};
	float c3_{0.f};
	float d3_{0.f};
	float e3_{-1.20034169};
	std::vector<float> solutions3_{3.85165235, -0.44747665};

	float a4_{2.9871f};
	float b4_{14.13};
	float c4_{-110.44};
	float d4_{-256.12313};
	float e4_{780.01212};
	std::vector<float> solutions4_{-7.51662383, -3.81118118,  4.62778163, 1.96968292};

	float a5_{-2.f};
	float b5_{0.};
	float c5_{16.f};
	float d5_{-291.f};
	float e5_{630.f};
	std::vector<float> solutions5_{-6.26497173, 2.26605988};

	float a6_{5.f};
	float b6_{17.f};
	float c6_{16.f};
	float d6_{67.f};
	float e6_{1630.31f};
	std::vector<float> solutions6_{};

	float epsilon_{1e-5};

	bool check_for_solution(float value, std::vector<float> &solutions) const
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

TEST_F(SetupQuarticEquation, quartic_equation_four_integer_solutions)
{
	N_Tuple<5, float> coefficients{a1_, b1_, c1_, d1_, e1_};
	auto solver = QuarticEquation<float>(coefficients, epsilon_);
	EXPECT_EQ(solver.number_of_solutions(), 4);
	EXPECT_TRUE(check_for_solution(solver.solutions()[0], solutions1_));
	EXPECT_TRUE(check_for_solution(solver.solutions()[1], solutions1_));
	EXPECT_TRUE(check_for_solution(solver.solutions()[2], solutions1_));
	EXPECT_TRUE(check_for_solution(solver.solutions()[3], solutions1_));
}

TEST_F(SetupQuarticEquation, quartic_equation_two_solutions)
{
	N_Tuple<5, float> coefficients{a2_, b2_, c2_, d2_, e2_};
	auto solver = QuarticEquation<float>(coefficients, epsilon_);
	EXPECT_EQ(solver.number_of_solutions(), 2);
	EXPECT_TRUE(check_for_solution(solver.solutions()[0], solutions2_));
	EXPECT_TRUE(check_for_solution(solver.solutions()[1], solutions2_));
}


TEST_F(SetupQuarticEquation, quartic_equation_linear_and_quadratic_coefficient_are_zero_two_solutions)
{
	N_Tuple<5, float> coefficients{a3_, b3_, c3_, d3_, e3_};
	auto solver = QuarticEquation<float>(coefficients, epsilon_);
	EXPECT_EQ(solver.number_of_solutions(), 2);
	EXPECT_TRUE(check_for_solution(solver.solutions()[0], solutions3_));
	EXPECT_TRUE(check_for_solution(solver.solutions()[1], solutions3_));
}

TEST_F(SetupQuarticEquation, quartic_equation_four_solutions)
{
	N_Tuple<5, float> coefficients{a4_, b4_, c4_, d4_, e4_};
	auto solver = QuarticEquation<float>(coefficients, epsilon_);
	EXPECT_EQ(solver.number_of_solutions(), 4);
	EXPECT_TRUE(check_for_solution(solver.solutions()[0], solutions4_));
	EXPECT_TRUE(check_for_solution(solver.solutions()[1], solutions4_));
	EXPECT_TRUE(check_for_solution(solver.solutions()[2], solutions4_));
	EXPECT_TRUE(check_for_solution(solver.solutions()[3], solutions4_));
}

TEST_F(SetupQuarticEquation, quartic_equation_cubic_coefficient_is_zero_two_solutions)
{
	N_Tuple<5, float> coefficients{a5_, b5_, c5_, d5_, e5_};
	auto solver = QuarticEquation<float>(coefficients, epsilon_);
	EXPECT_EQ(solver.number_of_solutions(), 2);
	EXPECT_TRUE(check_for_solution(solver.solutions()[0], solutions5_));
	EXPECT_TRUE(check_for_solution(solver.solutions()[1], solutions5_));
}

TEST_F(SetupQuarticEquation, quartic_equation_cubic_zero_solutions)
{
	N_Tuple<5, float> coefficients{a6_, b6_, c6_, d6_, e6_};
	auto solver = QuarticEquation<float>(coefficients, epsilon_);
	EXPECT_EQ(solver.number_of_solutions(), 0);
}
