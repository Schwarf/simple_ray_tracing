//
// Created by andreas on 07.11.21.
//

#include "gtest/gtest.h"
#include "miscellaneous/cubic_equation.h"
#include <regex>

class SetupCubicEquation: public testing::Test
{
protected:
	float a1_{1.2111111999};
	float b1_{-1.731131};
	float c1_{2.612123121};
	float d1_{-3.9080121};
	std::vector<float> solutions1_{1.4628704};

	float a2_{0.89012};
	float b2_{-2.1212};
	float c2_{-0.77878};
	float d2_{1.9089};
	std::vector<float> solutions2_{2.37049239, -0.94489002,  0.95744711};

	float a3_{3.121};
	float b3_{-12.};
	float c3_{-18.90009};
	float d3_{0.};
	std::vector<float> solutions3_{5.04526319, -1.20034169,  0.};

	float a4_{3.121};
	float b4_{-12.};
	float c4_{18.90009};
	float d4_{0.};
	std::vector<float> solutions4_{0.0};

	float a5_{2.};
	float b5_{-12.};
	float c5_{0.};
	float d5_{12.};
	std::vector<float> solutions5_{5.82305015,  1.1074036,  -0.93045375};

	float a6_{-2.};
	float b6_{0.};
	float c6_{6.};
	float d6_{-2.};
	std::vector<float> solutions6_{-1.87938524, 1.53208889,  0.34729636};

	float a7_{-2.e-9};
	float b7_{0.};
	float c7_{6.};
	float d7_{-2.};

	float epsilon_{1e-4};

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

TEST_F(SetupCubicEquation, cubic_equation_one_solution)
{
	N_Tuple<4, float> coefficients{a1_, b1_, c1_, d1_};
	auto solver = CubicEquation<float>(coefficients, epsilon_);
	EXPECT_EQ(solver.number_of_solutions(), 1);
	EXPECT_TRUE(check_for_solution(solver.solutions()[0], solutions1_));
}

TEST_F(SetupCubicEquation, cubic_equation_three_solutions)
{
	N_Tuple<4, float> coefficients{a2_, b2_, c2_, d2_};
	auto solver = CubicEquation<float>(coefficients, epsilon_);
	EXPECT_EQ(solver.number_of_solutions(), 3);
	EXPECT_TRUE(check_for_solution(solver.solutions()[0], solutions2_));
	EXPECT_TRUE(check_for_solution(solver.solutions()[1], solutions2_));
	EXPECT_TRUE(check_for_solution(solver.solutions()[2], solutions2_));
}

TEST_F(SetupCubicEquation, cubic_equation_three_solutions_constant_is_zero)
{
	N_Tuple<4, float> coefficients{a3_, b3_, c3_, d3_};
	auto solver = CubicEquation<float>(coefficients, epsilon_);
	EXPECT_EQ(solver.number_of_solutions(), 3);
	EXPECT_TRUE(check_for_solution(solver.solutions()[0], solutions3_));
	EXPECT_TRUE(check_for_solution(solver.solutions()[1], solutions3_));
	EXPECT_TRUE(check_for_solution(solver.solutions()[2], solutions3_));
}

TEST_F(SetupCubicEquation, cubic_equation_one_solutions_constant_is_zero)
{
	N_Tuple<4, float> coefficients{a4_, b4_, c4_, d4_};
	auto solver = CubicEquation<float>(coefficients, epsilon_);
	EXPECT_EQ(solver.number_of_solutions(), 1);
	EXPECT_TRUE(check_for_solution(solver.solutions()[0], solutions4_));
}

TEST_F(SetupCubicEquation, cubic_equation_three_solutions_linear_coefficient_is_zero)
{
	N_Tuple<4, float> coefficients{a5_, b5_, c5_, d5_};
	auto solver = CubicEquation<float>(coefficients, epsilon_);
	EXPECT_EQ(solver.number_of_solutions(), 3);
	EXPECT_TRUE(check_for_solution(solver.solutions()[0], solutions5_));
	EXPECT_TRUE(check_for_solution(solver.solutions()[1], solutions5_));
	EXPECT_TRUE(check_for_solution(solver.solutions()[2], solutions5_));
}


TEST_F(SetupCubicEquation, cubic_equation_three_solutions_quadratic_coefficient_is_zero)
{
	N_Tuple<4, float> coefficients{a6_, b6_, c6_, d6_};
	auto solver = CubicEquation<float>(coefficients, epsilon_);
	EXPECT_EQ(solver.number_of_solutions(), 3);
	EXPECT_TRUE(check_for_solution(solver.solutions()[0], solutions6_));
	EXPECT_TRUE(check_for_solution(solver.solutions()[1], solutions6_));
	EXPECT_TRUE(check_for_solution(solver.solutions()[2], solutions6_));
}

TEST_F(SetupCubicEquation, cubic_equation_very_small_cubic_coefficient)
{
	N_Tuple<4, float> coefficients{a7_, b7_, c7_, d7_};
	auto value = a7_ / (a7_ + b7_ + c7_ + d7_) / 4.;
	std::string message =
		"In class CubicEquation cubic_coefficient=" + std::to_string(value) + " is zero (within epsilon) "
			+ std::to_string(epsilon_);
	try {
		auto solver = CubicEquation<float>(coefficients, epsilon_);
		FAIL() << "Expected std::out_of_range";
	}
	catch (std::out_of_range const &err) {
		EXPECT_TRUE(err.what() == message);
	}

}
