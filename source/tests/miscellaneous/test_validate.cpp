//
// Created by andreas on 10.10.21.
//

#include "gtest/gtest.h"
#include "miscellaneous/validate.h"


class SetupValidate: public testing::Test
{
protected:
	float threshold{2.0};
	float below_threshold{1.0};
	float above_threshold{3.0};
};


TEST_F(SetupValidate, test_above_threshold_does_not_throw)
{
	auto validate = Validate();
	std::string variable_name  = "above_threshold";
	std::string class_name = "TestValidate";

	validate.is_above_threshold(variable_name, above_threshold, threshold, class_name);
}

TEST_F(SetupValidate, test_below_threshold_does_not_throw)
{
	auto validate = Validate();
	std::string variable_name  = "below_threshold";
	std::string class_name = "TestValidate";

	validate.is_below_threshold(variable_name, below_threshold, threshold, class_name);
}

TEST_F(SetupValidate, test_above_threshold_does_throw)
{
	auto validate = Validate();
	std::string variable_name  = "below_threshold";
	std::string class_name = "TestValidate";
	std::string message = "In class " + class_name + " " + variable_name + "=" + std::to_string(below_threshold)
		+ " is lower than "+ std::to_string(threshold);

	try {
		validate.is_above_threshold(variable_name, below_threshold, threshold, class_name);
		FAIL() << "Expected std::out_of_range";
	}
	catch (std::out_of_range const &err) {
		EXPECT_TRUE(err.what() ==message);
	}

}

TEST_F(SetupValidate, test_below_threshold_does_throw)
{
	auto validate = Validate();
	std::string variable_name  = "below_threshold";
	std::string class_name = "TestValidate";

	std::string message = "In class " + class_name + " " + variable_name + "=" + std::to_string(above_threshold)
		+ " is greater than "+ std::to_string(threshold);

	try {
		validate.is_below_threshold(variable_name, above_threshold, threshold, class_name);
		FAIL() << "Expected std::out_of_range";
	}
	catch (std::out_of_range const &err) {
		EXPECT_TRUE(err.what() ==message);
	}

}
