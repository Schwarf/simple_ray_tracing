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

	float epsilon_{1e-6};

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
	c_vector<5, float> coefficients{a1_, b1_, c1_, d1_, e1_};
	auto solver = QuarticEquation<float>(coefficients, epsilon_);
	EXPECT_EQ(solver.number_of_solutions(), 4);
	EXPECT_TRUE(check_for_solution(solver.solutions()[0], solutions1_));
	EXPECT_TRUE(check_for_solution(solver.solutions()[1], solutions1_));
	EXPECT_TRUE(check_for_solution(solver.solutions()[2], solutions1_));
	EXPECT_TRUE(check_for_solution(solver.solutions()[3], solutions1_));
}
